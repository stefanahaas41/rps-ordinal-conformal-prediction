"""Min-CPS conformity score for ordinal conformal prediction."""

from typing import Optional, Callable

import numpy as np
from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import check_proba_normalized
from mapie.estimator.classifier import EnsembleClassifier
from numpy.typing import NDArray


class MinCPS(BaseClassificationScore):
    """Min-CPS: Provably minimum-length conformal prediction sets.

    Reference
    ---------
    Zhang, Z., Chen, X., Shi, Y., Ma, L. L., Xu, Z., & Yan, Y. (2026, March).
    Minimum-Length Conformal Prediction Sets for Ordinal Classification.
    In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 40, No. 34, pp. 28662-28670).
    """

    def __init__(self) -> None:
        super().__init__()

    def get_qhat_ordinal_aps(
        self,
        prediction_function: Callable,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
        alpha: float,
        lamda: float,
        tol: float = 1e-6,
    ) -> float:
        n = cal_scores.shape[0]
        left, right = 0.001, 1.999
        best_q = right
        target_coverage = np.ceil((n + 1) * (1 - alpha)) / n
        while right - left > tol:
            mid = (left + right) / 2
            coverage, *_ = self.evaluate_sets(
                prediction_function, np.copy(cal_scores), np.copy(cal_labels), mid, alpha, lamda
            )
            if coverage >= target_coverage:
                best_q = mid
                right = mid
            else:
                left = mid
        return best_q

    def sliding_window_predict_set(self, val_scores: np.ndarray, qhat: float, lamda: float) -> np.ndarray:
        N, K = val_scores.shape
        P = np.zeros((N, K), dtype=bool)
        for i in range(N):
            f = val_scores[i]
            y_star = np.argmax(f)
            prefix = np.zeros(K + 1)
            for j in range(K):
                prefix[j + 1] = prefix[j] + f[j]

            best_len = float('inf')
            best_l = best_u = -1
            l = 0

            for u in range(y_star, K):
                while l <= y_star:
                    prob_sum = prefix[u + 1] - prefix[l]
                    dist_penalty = abs(y_star - l) + abs(y_star - u)
                    score = prob_sum - lamda * dist_penalty
                    if y_star >= l and y_star <= u and score >= qhat:
                        if u - l < best_len:
                            best_len = u - l
                            best_l, best_u = l, u

                    if score >= qhat:
                        l += 1
                    else:
                        break
            if best_l != -1 and best_u != -1:
                P[i, best_l:best_u + 1] = True
            else:
                P[i, 0:K] = True
        return P

    def evaluate_sets(
        self,
        prediction_function: Callable,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        qhat: float,
        alpha: float,
        lamda: float,
        print_bool: bool = False,
    ):
        sets = prediction_function(val_scores, qhat, lamda)
        sizes = sets.sum(axis=1)
        sizes_distribution = np.array([(sizes == i).mean() for i in range(5)])
        covered = sets[np.arange(val_labels.shape[0]), val_labels]
        coverage = covered.mean()
        label_stratified_coverage = [
            covered[val_labels == j].mean() for j in range(np.unique(val_labels).max() + 1)
        ]
        label_distribution = [
            (val_labels == j).mean() for j in range(np.unique(val_labels).max() + 1)
        ]
        if print_bool:
            print(f"alpha: {alpha} | coverage: {coverage:.4f} | avg size: {sizes.mean():.4f} | qhat: {qhat:.4f}")
        return coverage, label_stratified_coverage, sizes_distribution, sizes.mean(), label_distribution

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        y_enc: Optional[NDArray] = None,
        **kwargs
    ) -> NDArray:
        """
        Get the conformity score.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        y_enc: NDArray of shape (n_samples,)
            Target values as normalized encodings.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """

        y_true = np.array(y)
        y_proba = np.array(y_pred)

        self.y_true_ = y_true
        self.y_proba_ = y_proba

        # Dummy return, not used in MinCPS
        return y_proba

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        Get predictions from an EnsembleClassifier.

        Parameters
        -----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of predictions.
        """
        y_pred_proba = estimator.predict(X, agg_scores)
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )

        return y_pred_proba

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        Get the quantiles of the conformity scores for each uncertainty level.

        Parameters
        -----------
        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        quantiles = np.zeros_like(alpha_np)

        for i, alpha in enumerate(alpha_np):
            q = self.get_qhat_ordinal_aps(
                self.sliding_window_predict_set,
                self.y_proba_,
                self.y_true_,
                alpha,
                lamda=0,
                tol=1e-6
            )
            quantiles[i] = q

        return quantiles

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        Generate prediction sets based on the probability predictions,
        the conformity scores and the uncertainty level.

        Parameters
        -----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Target prediction.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n_samples, n_classes, n_alpha = y_pred_proba.shape

        prediction_sets = np.zeros((n_samples, n_classes, n_alpha), dtype=bool)

        if (estimator.cv == "prefit") or (agg_scores == "mean"):
            for q_idx, q in enumerate(self.quantiles_):
                prediction_sets[:, :, q_idx] = self.sliding_window_predict_set(
                    y_pred_proba[:, :, q_idx], q, 0
                )

        else:
            raise NotImplementedError(
                "Min-CPS prediction sets are only supported for prefit or "
                "mean aggregation."
            )

        return prediction_sets
