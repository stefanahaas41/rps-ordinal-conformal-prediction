"""Runner for tabular ordinal conformal prediction experiments with LightGBM."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import APSConformityScore, LACConformityScore
from sklearn.model_selection import train_test_split

from scores.minCPS import MinCPS
from scores.naive import NaiveCDFScore
from scores.rps import RankedProbabilityScore
from util.metrics import calculate_metrics, calculate_metrics_conf_prediction


def run_experiment(X: pd.DataFrame, y: pd.Series, le, data_type: str):
    """
    Run tabular data experiment for ordinal conformal prediction using LightGBM.
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

    random_seeds = list(range(50))
    X_train, X_cal_test, y_train, y_cal_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=32, shuffle=True
    )
    y_train = le.transform(y_train)

    model = LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    for i, trial_seed in enumerate(random_seeds):
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_cal_test, y_cal_test, test_size=0.5, stratify=y_cal_test, random_state=trial_seed, shuffle=True
        )
        y_cal = le.transform(y_cal)
        y_test = le.transform(y_test)

        y_pred_proba = model.predict_proba(X_test)
        metrics = calculate_metrics(y_test, y_pred_proba)
        df = pd.DataFrame([metrics])
        df.insert(0, "iteration", i)
        df.insert(1, "loss", "LightGBM")
        result = pd.concat([result, df], ignore_index=True)

        all_conf = [0.99, 0.98, 0.97, 0.95, 0.92, 0.9, 0.87, 0.85, 0.82, 0.8]
        n_cal = len(y_cal)
        conf = [c for c in all_conf if n_cal >= max(1.0 / (1.0 - c), 1.0 / c)]

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
        df_cp.insert(1, "loss", "LightGBM")
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
        df_cp.insert(1, "loss", "LightGBM")
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
        df_cp.insert(1, "loss", "LightGBM")
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
        df_cp.insert(1, "loss", "LightGBM")
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
        df_cp.insert(1, "loss", "LightGBM")
        df_cp.insert(2, "score", "LAC")
        result_cp = pd.concat([result_cp, df_cp], ignore_index=True)

        if i == len(random_seeds) - 1:
            df_probas = pd.DataFrame.from_records(y_pred_proba, columns=le.classes_)
            df_probas.to_csv(f"lgbm_probas_{data_type}.csv")

    result.to_csv(f"lgbm_{data_type}_experiments.csv", index=False)
    result_cp.to_csv(f"lgbm_{data_type}_experiments_cp.csv", index=False)
