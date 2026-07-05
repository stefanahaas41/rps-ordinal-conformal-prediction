import random

import numpy as np
import pandas as pd
import torch
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import LACConformityScore, APSConformityScore
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint
from skorch.dataset import ValidSplit
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset
from torchvision import models

from dlordinal.output_layers import COPOC
from scores.minCPS import MinCPS
from scores.naive import NaiveCDFScore
from scores.rps import RankedProbabilityScore
from util.metrics import calculate_metrics_conf_prediction, calculate_metrics


def filter_conf_levels(conf_levels, n_scores, eps=1e-12):
    # gültiger Bereich: 1/n < c < 1 - 1/n (strikt wegen "must be lower than")
    lower = 1.0 / n_scores + eps
    upper = 1.0 - 1.0 / n_scores - eps
    kept = [c for c in conf_levels if (c > lower) and (c < upper)]
    removed = [c for c in conf_levels if c not in kept]
    return kept, removed, (lower, upper)


def run_image_experiment(device, num_classes, data_type, train_data, test_data,
                         optimizer=None, lr=2e-4, weight_decay=1e-4, batch_size=32,
                         es_patience=30, lr_patience=10, lr_factor=0.5,
                         lr_monitor="valid_loss", stratified=True,
                         use_determinism=True, n_seeds=50,
                         shuffle=True, aps_randomized=True):
    # --------------------------------------------------------------- determinism -
    SEED = 42
    if use_determinism:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if optimizer is None:
        optimizer = torch.optim.AdamW

    result = pd.DataFrame()
    result_cp = pd.DataFrame()
    # CP confidence
    conf = [0.99, 0.98, 0.97, 0.95, 0.92, 0.9, 0.87, 0.85, 0.82, 0.8]
    seeds = [i for i in range(42, 42 + n_seeds)]

    losses = [
        COPOC(),
        CrossEntropyLoss()
    ]

    for loss in losses:

        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        if type(loss).__name__ == "COPOC":
            model = nn.Sequential(model, COPOC())
            loss_function = CrossEntropyLoss()
        else:
            loss_function = loss

        estimator = NeuralNetClassifier(
            module=model,
            criterion=loss_function,
            optimizer=optimizer,
            lr=lr,
            **({"optimizer__weight_decay": weight_decay} if weight_decay > 0 else {}),
            max_epochs=100,
            batch_size=batch_size,
            device=device,
            iterator_train__shuffle=shuffle,
            iterator_train__num_workers=0,
            iterator_valid__num_workers=0,
            train_split=ValidSplit(0.1, random_state=SEED, stratified=stratified),
            callbacks=[
                EarlyStopping(patience=es_patience, monitor="valid_loss"),
                LRScheduler(policy=ReduceLROnPlateau, monitor=lr_monitor,
                            patience=lr_patience, factor=lr_factor, min_lr=1e-6),
                Checkpoint(monitor="valid_loss_best", load_best=True,
                           dirname=f"ckpt_{data_type}_{type(loss).__name__}"),
            ],
        )

        estimator.fit(
            X=train_data, y=torch.tensor(train_data.targets, dtype=torch.long)
        )

        for i, seed in enumerate(seeds):
            # Split test data into calibration and test sets
            test_indices = np.arange(len(test_data))
            test_idx, cal_idx = train_test_split(
                test_indices, test_size=0.5, stratify=test_data.targets, random_state=seed, shuffle=True
            )

            X_test = Subset(test_data, test_idx)
            y_test = np.array([test_data.targets[i] for i in test_idx])
            X_cal = Subset(test_data, cal_idx)
            y_cal = np.array([test_data.targets[i] for i in cal_idx])

            print("Calibration set size:", len(X_cal))
            print("Test set size:", len(X_test))

            n = len(X_cal)
            conf_kept, conf_removed, (lo, hi) = filter_conf_levels(conf, n)
            conf = conf_kept

            # Predict
            y_pred_proba = estimator.predict_proba(X_test)
            df_probas = pd.DataFrame.from_records(y_pred_proba, columns=[k for k in range(num_classes)])
            df_probas.to_csv(f"probas_{data_type}_{type(loss).__name__}.csv")

            metrics = calculate_metrics(y_test, y_pred_proba)
            df = pd.DataFrame([metrics])
            df.insert(0, "iteration", i)
            df.insert(1, "loss", type(loss).__name__)
            result = pd.concat([result, df], ignore_index=True)

            print("Min CPS")
            mapie_classifier = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=conf,
                conformity_score=MinCPS(),
                prefit=True,
            ).conformalize(X_cal, y_cal)
            predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "min-CPS")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

            # RPS
            print("RPSConformityScore")
            mapie_classifier = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=conf,
                conformity_score=RankedProbabilityScore(),
                random_state=42,
                prefit=True,
            ).conformalize(X_cal, y_cal)
            predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            # print(metrics_cp)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "RPS")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

            print("Naive CDF Score")
            mapie_classifier = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=conf,
                conformity_score=NaiveCDFScore(),
                prefit=True,
            ).conformalize(X_cal, y_cal)
            predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "OCDF")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

            # APS
            print("APSConformityScore")
            mapie_classifier = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=conf,
                conformity_score=APSConformityScore(),
                random_state=42,
                prefit=True,
            ).conformalize(X_cal, y_cal)
            if aps_randomized:
                predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test, conformity_score_params={
                    'include_last_label': 'randomized'})
            else:
                predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "APS")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

            # LAC
            print("LACConformityScore")
            mapie_classifier = SplitConformalClassifier(
                estimator=estimator,
                confidence_level=conf,
                conformity_score=LACConformityScore(),
                random_state=42,
                prefit=True,
            ).conformalize(X_cal, y_cal)
            predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "LAC")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

    result.to_csv(f"{data_type}_experiments.csv", index=False)
    result_cp.to_csv(f"{data_type}_experiments_cp.csv", index=False)
