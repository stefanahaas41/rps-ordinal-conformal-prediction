"""Evaluation metrics for ordinal classification and conformal prediction."""

import numpy as np
from dlordinal.metrics import ranked_probability_score, accuracy_off1, mmae, amae
from mapie.metrics.classification import classification_coverage_score, classification_mean_width_score
from mapie.metrics.regression import regression_mean_width_score
from pycalib.metrics import ECE, MCE, conf_ECE, conf_MCE
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score, accuracy_score, brier_score_loss
from typing import Dict, List, Union


def empirical_risk(y_pred_proba: np.ndarray, loss: str = "l1") -> np.ndarray:
    """
    Compute empirical risk minimization prediction for each sample.
    Args:
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)
        loss: Loss function to use ('l1' or 'l2')
    Returns:
        np.ndarray: Predicted class indices
    """
    y_pred = np.zeros(y_pred_proba.shape[0])
    for i in range(y_pred_proba.shape[0]):
        class_risks = np.zeros(y_pred_proba.shape[1])
        for k in range(y_pred_proba.shape[1]):
            for j in range(y_pred_proba.shape[1]):
                dist = np.abs(k - j) ** 2 if loss == "l2" else np.abs(k - j)
                class_risks[k] += dist * y_pred_proba[i, j]
        y_pred[i] = np.argmin(class_risks)
    return y_pred.astype(int)


def is_unimodal(probs: np.ndarray) -> bool:
    """
    Check if a 1D array is unimodal (increases to a peak, then decreases).
    Args:
        probs: 1D array of probabilities
    Returns:
        bool: True if unimodal, False otherwise
    """
    peak_idx = np.argmax(probs)
    inc = np.all(np.diff(probs[: peak_idx + 1]) >= 0)
    dec = np.all(np.diff(probs[peak_idx:]) <= 0)
    return inc and dec


def check_unimodality(y_pred: np.ndarray) -> float:
    """
    Check unimodality for each row in y_pred and return the proportion.
    Args:
        y_pred: 2D array of predicted probabilities
    Returns:
        float: Proportion of unimodal rows
    """
    unimodal_flags = np.array([is_unimodal(row) for row in y_pred])
    proportion = np.mean(unimodal_flags)
    return proportion


def variance_for_probabilities_binary(p: np.ndarray) -> float:
    """
    Compute variance for binary probabilities.
    Args:
        p: 1D array of probabilities (length 2)
    Returns:
        float: Variance
    """
    return p[0] * (1 - p[0])


def mean_for_probabilities(probabilities: np.ndarray) -> float:
    """
    Compute expected value for discrete probabilities.
    Args:
        probabilities: 1D array of probabilities
    Returns:
        float: Expected value
    """
    values = np.arange(len(probabilities))
    expected_value = probabilities @ values
    return expected_value


def variance_for_probabilities(probabilities: np.ndarray) -> float:
    """
    Calculate variance of a discrete random variable.
    Args:
        probabilities: 1D array of probabilities
    Returns:
        float: Variance
    """
    values = np.arange(len(probabilities))
    expected_value = probabilities @ values
    squared_difference = (expected_value - values) ** 2
    return squared_difference @ probabilities


def calculate_metrics(y_true: Union[np.ndarray, List[int]], y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate a suite of classification metrics for ordinal predictions.
    Args:
        y_true: True class labels
        y_pred: Predicted probabilities or logits
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    if np.allclose(np.sum(y_pred, axis=1), 1):
        y_pred_proba = y_pred
    else:
        y_pred_proba = softmax(y_pred, axis=1)
    y_pred_max = np.argmax(y_pred, axis=1)
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    amae_metric = amae(y_true, y_pred_proba)
    mmae_metric = mmae(y_true, y_pred_proba)
    mae_er = np.mean(np.abs(y_true - empirical_risk(y_pred_proba)))
    mse_er = np.mean((y_true - empirical_risk(y_pred_proba, 'l2')) ** 2)
    mae = np.mean(np.abs(y_true - y_pred_max))
    mse = np.mean((y_true - y_pred_max) ** 2)
    acc = accuracy_score(y_true, y_pred_max)
    acc_1off = accuracy_off1(y_true, y_pred_proba)
    qwk = cohen_kappa_score(y_true, y_pred_max, weights="quadratic")
    rps = ranked_probability_score(y_true, y_pred_proba)
    bs = brier_score_loss(y_true, y_pred_proba)
    ece = ECE(y_true, y_pred_proba, bins=10)
    mce = MCE(y_true, y_pred_proba, bins=10, mce_full=True)
    conf_mce = conf_MCE(y_true, y_pred_proba, bins=10)
    conf_ece = conf_ECE(y_true, y_pred_proba, bins=10)
    var = np.mean(np.apply_along_axis(variance_for_probabilities, 1, y_pred_proba))
    conf = np.mean(np.max(y_pred_proba, axis=1))
    unimodal_prop = check_unimodality(y_pred_proba)
    metrics = {
        "ACC": acc,
        "1OFF": acc_1off,
        "MAE": mae,
        "MSE": mse,
        "MAE_ER": mae_er,
        "MSE_ER": mse_er,
        "QWK": qwk,
        "AMAE": amae_metric,
        "MMAE": mmae_metric,
        "RPS": rps,
        "BS": bs,
        "ECE": ece,
        "MCE": mce,
        "CONF_ECE": conf_ece,
        "CONF_MCE": conf_mce,
        "UMOD": unimodal_prop,
        "CONF": conf,
        "VAR": var
    }
    return metrics


def _check_contiguous(a: np.ndarray) -> float:
    """
    Check if each row in a binary matrix is contiguous (all 1s are together).
    Args:
        a: 2D binary array
    Returns:
        float: Proportion of contiguous rows
    """
    result = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        first_one = False
        zero_after_one = False
        one_after_zero = False
        for j in range(a.shape[1]):
            col = a[i, j]
            if col == 1 and not first_one:
                first_one = True
            elif col == 0 and first_one:
                zero_after_one = True
            elif col == 1 and zero_after_one:
                one_after_zero = True
        result[i] = 0 if one_after_zero else 1
    return result.mean()


def calculate_metrics_conf_prediction(
    y_true: np.ndarray,
    y_pred_set: np.ndarray,
    alphas: np.ndarray
) -> List[Dict[str, float]]:
    """
    Calculate conformal prediction metrics for ordinal classification.
    Args:
        y_true: True class labels
        y_pred_set: Predicted sets (n_samples, n_classes, n_alphas)
        alphas: Array of alpha values
    Returns:
        List[Dict[str, float]]: List of metrics for each alpha
    """
    classes = np.unique(y_true)
    coverage = classification_coverage_score(y_true, y_pred_set)
    set_size = classification_mean_width_score(y_pred_set)
    y_intervals = y_pred_set[:, :2, :].astype(int).copy()
    y_intervals[:, 0, :] = 0
    y_intervals[:, 1, :] = 0
    for i in range(y_pred_set.shape[0]):
        for j in range(y_pred_set.shape[-1]):
            true_indices = np.where(y_pred_set[i, :, j])[0]
            if true_indices.size > 0:
                y_intervals[i, 0, j] = true_indices[0]
                y_intervals[i, 1, j] = true_indices[-1]
    mean_width = regression_mean_width_score(y_intervals)
    mamm = np.zeros_like(alphas)
    wamm = np.zeros_like(alphas)
    mase = np.zeros_like(alphas)
    aisl = np.zeros_like(alphas)
    set_sizes = np.zeros((len(y_true), len(alphas)))
    for idx, alpha in enumerate(alphas):
        intervals_alpha = y_intervals[:, :, idx]
        sizes_interval = intervals_alpha[:, 1] - intervals_alpha[:, 0]
        sizes_set = y_pred_set[:, :, idx].sum(axis=1)
        pred_set_elements = [set() for _ in range(len(y_true))]
        for i in range(len(pred_set_elements)):
            pred_set_bools = y_pred_set[i, :, idx]
            for k in range(len(pred_set_bools)):
                if pred_set_bools[k]:
                    pred_set_elements[i].add(k)
        error_distance_set = np.zeros(len(y_true))
        for i in range(len(pred_set_elements)):
            if y_true[i] in pred_set_elements[i]:
                error_distance_set[i] = 0
            else:
                distances = [abs(y_true[i] - k) for k in pred_set_elements[i]]
                error_distance_set[i] = min(distances) if len(distances) > 0 else (classes[-1] - classes[0])
        error_distance_set_missed = error_distance_set[error_distance_set > 0]
        error_distance_interval = np.array([
            0 if ((y_true[i] >= intervals_alpha[i, 0]) and (y_true[i] <= intervals_alpha[i, 1]))
            else ((intervals_alpha[i, 0] - y_true[i]) if (y_true[i] < intervals_alpha[i, 0]) else (y_true[i] - intervals_alpha[i, 1]))
            for i in range(len(y_pred_set))
        ])
        mamm[idx] = np.mean(error_distance_set_missed) if error_distance_set_missed.size > 0 else 0.0
        wamm[idx] = np.max(error_distance_set_missed) if error_distance_set_missed.size > 0 else 0.0
        aisl[idx] = np.mean(sizes_interval + ((2.0 / (1 - alpha)) * error_distance_interval))
        set_sizes[:, idx] = sizes_set
        mase[idx] = np.mean(error_distance_set)
    metrics_result = []
    for idx, alpha in enumerate(alphas):
        metrics = {
            "ALPHA": (1 - alpha),
            "COV": coverage[idx],
            "PS": set_size[idx],
            "MW": mean_width[idx],
            "MAMM": mamm[idx],
            "WAMM": wamm[idx],
            "AISL": aisl[idx],
            "MASE": mase[idx],
            "CV%": 1 - _check_contiguous(y_pred_set.astype(int)[:, :, idx])
        }
        metrics_result.append(metrics)
    return metrics_result

