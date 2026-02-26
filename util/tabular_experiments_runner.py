import numpy as np
import pandas as pd
import torch
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import APSConformityScore, LACConformityScore
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skorch import NeuralNetClassifier
from skorch.toy import MLPModule
from torch import nn
from torch.nn import CrossEntropyLoss

from losses.copoc import COPOC
from scores.minCPS import MinCPS
from scores.naive import NaiveCDFScore
from scores.rps import RankedProbabilityScore
from util.metrics import calculate_metrics, calculate_metrics_conf_prediction


def run_experiment(X: pd.DataFrame, y: pd.Series, le, data_type: str):
    """
    Run tabular data experiment for ordinal conformal prediction.
    Args:
        X: pd.DataFrame, features
        y: pd.Series, labels
        le: LabelEncoder
        data_type: str, dataset name
    Returns:
        None
    """
    result = pd.DataFrame()
    result_cp = pd.DataFrame()

    def toarray(X, dtype=np.float32):
        return X.astype(dtype)

    losses = [COPOC(), CrossEntropyLoss()]
    random_seeds = list(range(50))
    X_train, X_cal_test, y_train, y_cal_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=32, shuffle=True
    )
    y_train = le.transform(y_train)

    for loss in losses:
        mlp = MLPModule(
            input_units=X_train.shape[1],
            hidden_units=64,
            num_hidden=1,
            output_units=len(le.classes_),
            nonlin=nn.ReLU()
        )
        if type(loss).__name__ == "COPOC":
            mlp = nn.Sequential(mlp, COPOC())
            loss_function = CrossEntropyLoss()
        else:
            loss_function = loss
        ann = NeuralNetClassifier(
            mlp,
            criterion=loss_function,
            batch_size=min(200, len(y_train)),
            max_epochs=20,
            lr=0.001,
            optimizer=torch.optim.Adam,
            iterator_train__shuffle=True
        )
        model = Pipeline([
            ('scale', StandardScaler()),
            ('toarray', FunctionTransformer(toarray, validate=False)),
            ('net', ann),
        ])
        model.fit(X_train, y_train)
        for i, trial_seed in enumerate(random_seeds):
            # Split calibration and test set
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_cal_test, y_cal_test, test_size=0.5, stratify=y_cal_test, random_state=trial_seed, shuffle=True
            )
            y_cal = le.transform(y_cal)
            y_test = le.transform(y_test)
            # Predict probabilities for test set
            y_pred_proba = model.predict_proba(X_test)
            metrics = calculate_metrics(y_test, y_pred_proba)
            df = pd.DataFrame([metrics])
            df.insert(0, "iteration", i)
            df.insert(1, "loss", type(loss).__name__)
            result = pd.concat([result, df], ignore_index=True)

            # CP confidence
            conf = [0.99, 0.98, 0.97, 0.95, 0.92, 0.9, 0.87, 0.85, 0.82, 0.8]

            # Min CPS
            mapie_classifier = SplitConformalClassifier(
                estimator=model,
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
            mapie_classifier = SplitConformalClassifier(
                estimator=model,
                confidence_level=conf,
                conformity_score=RankedProbabilityScore(),
                random_state=42,
                prefit=True,
            ).conformalize(X_cal, y_cal)
            predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "RPS")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

            # Naive CDF Score
            mapie_classifier = SplitConformalClassifier(
                estimator=model,
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
            mapie_classifier = SplitConformalClassifier(
                estimator=model,
                confidence_level=conf,
                conformity_score=APSConformityScore(),
                random_state=42,
                prefit=True,
            ).conformalize(X_cal, y_cal)
            predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test, conformity_score_params={
                'include_last_label': 'randomized'})
            metrics_cp = calculate_metrics_conf_prediction(y_test, predicted_sets, conf)
            df_cp = pd.DataFrame(metrics_cp)
            df_cp.insert(0, "iteration", i)
            df_cp.insert(1, "loss", type(loss).__name__)
            df_cp.insert(2, "score", "APS")
            result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

            # LAC
            mapie_classifier = SplitConformalClassifier(
                estimator=model,
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

            # Save mean probabilities of last run
            if i == len(random_seeds) - 1:
                df_probas = pd.DataFrame.from_records(y_pred_proba, columns=le.classes_)
                df_probas.to_csv(f"probas_{data_type}_{type(loss).__name__}.csv")

    result.to_csv(f"{data_type}_experiments.csv", index=False)
    result_cp.to_csv(f"{data_type}_experiments_cp.csv", index=False)
