from typing import Optional
import numpy as np
from mapie._machine_precision import EPSILON
from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import check_proba_normalized
from mapie.estimator.classifier import EnsembleClassifier
from mapie.utils import _compute_quantiles
from numpy.typing import NDArray


class RankedProbabilityScore(BaseClassificationScore):
    """
    Ranked Probability Score (RPS) for ordinal classification.
    Reference:
    Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories.
    Bulletin of the American Meteorological Society.
    """

    def __init__(self) -> None:
        super().__init__()

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

        y_oh = np.zeros(y_proba.shape)
        y_oh[np.arange(len(y_true)), y_true] = 1

        y_oh = y_oh.cumsum(axis=1)
        y_proba = y_proba.cumsum(axis=1)

        conformity_scores = np.ones(len(y_true))

        for i in range(len(y_true)):
            if y_true[i] in np.arange(y_proba.shape[1]):
                conformity_scores[i] = np.power(y_proba[i] - y_oh[i], 2).sum()

        return conformity_scores

        # Casting
        # y_enc = cast(NDArray, y_enc)
        #
        # # Conformity scores
        # conformity_scores = np.take_along_axis(
        #     1 - y_pred, y_enc.reshape(-1, 1), axis=1
        # )
        #
        # return conformity_scores

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
                for class_idx in range(n_classes):
                    # Binary outcomes with 1 at current class position
                    outcomes = np.zeros((n_samples, n_classes))
                    outcomes[:, class_idx] = 1

                    # Cumulative sums
                    outcomes_cumsum = outcomes.cumsum(axis=1)
                    y_pred_cumsum = y_pred_proba[:, :, q_idx].cumsum(axis=1)
                    # RPS calculation
                    rps = np.power(y_pred_cumsum - outcomes_cumsum, 2).sum(axis=1)

                    # Include class in prediction set if RPS is below threshold
                    prediction_sets[:, class_idx, q_idx] = np.less_equal(
                        rps - q, EPSILON
                    )

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