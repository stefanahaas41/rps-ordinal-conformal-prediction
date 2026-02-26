from typing import Optional
import numpy as np
from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import check_proba_normalized
from mapie.estimator.classifier import EnsembleClassifier
from mapie.utils import _compute_quantiles
from numpy.typing import NDArray


class NaiveCDFScore(BaseClassificationScore):
    """
    Naive CDF score for ordinal conformal prediction.
    Reference: Lu et al. (2022), MICCAI.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_conformity_scores_naive_cdf(self, val_scores, y_true):
        """
        Compute conformity scores for ordinal conformal prediction.

        Parameters:
        - val_scores: np.array, shape (n_samples, n_classes), per-class scores (e.g., softmax)
        - y_true: np.array, shape (n_samples,), true labels as class indices

        Returns:
        - conformity_scores: np.array, shape (n_samples,), conformity scores per sample
        """
        cumsum = val_scores.cumsum(axis=1)  # cumulative sums per sample

        # Get cumulative sum at true label for each sample
        cumsum_true = cumsum[np.arange(len(y_true)), y_true]

        # Get index of max score per sample
        argmaxes = val_scores.argmax(axis=1)
        maxes = cumsum[np.arange(val_scores.shape[0]), argmaxes]

        # Conformity score: absolute difference between cumulative sum at true label and max cumulative sum
        conformity_scores = np.abs(cumsum_true - maxes)

        return conformity_scores

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

        conformity_scores = self.get_conformity_scores_naive_cdf(y_proba, y_true)

        return conformity_scores

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
        n = len(conformity_scores)

        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            quantiles_ = _compute_quantiles(
                conformity_scores,
                alpha_np
            )
        else:
            quantiles_ = (n + 1) * (1 - alpha_np)

        return quantiles_

    def cdf_naive_ordinal_prediction(self, val_scores, qhat):
        cumsum = val_scores.cumsum(axis=1)
        argmaxes = val_scores.argmax(axis=1)
        maxes = cumsum[np.arange(val_scores.shape[0]), argmaxes]
        prediction_set = ((cumsum >= (maxes[:, None] - qhat)) & (cumsum <= (maxes[:, None] + qhat)))
        return prediction_set

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
        # n = len(conformity_scores)
        n_samples, n_classes, n_alpha = y_pred_proba.shape

        # Initialize output
        prediction_sets = np.zeros((n_samples, n_classes, n_alpha), dtype=bool)

        if (estimator.cv == "prefit") or (agg_scores == "mean"):
            # prediction_sets = np.less_equal(
            #     (1 - y_pred_proba) - self.quantiles_, EPSILON
            # )

            # For each class, simulate it being the true label and compute RPS
            for q_idx, q in enumerate(self.quantiles_):
                prediction_sets[:, :, q_idx] = self.cdf_naive_ordinal_prediction(y_pred_proba[:, :, q_idx], q)


        else:
            pass
            # TODO: Implement
            # y_pred_included = np.less_equal(
            #     (1 - y_pred_proba) - conformity_scores.ravel(), EPSILON
            # ).sum(axis=2)
            #
            # prediction_sets = np.stack(
            #     [
            #         np.greater_equal(
            #             y_pred_included - _alpha * (n - 1), -EPSILON
            #         )
            #         for _alpha in alpha_np
            #     ], axis=2
            # )

        return prediction_sets
