"""
Inhomogeneous Poisson scoring-intensity model.

Models the probability of a scoring event on any given play as a logistic
regression over game-state features.  By dividing by the average time between
plays we convert play-level probabilities into a continuous-time intensity
λ(t) in units of *events per second*.

Finance analogy: this is the equivalent of modelling the intensity of
order arrivals at a given price level — in this case the "order" is a
scoring event that shifts the mid-price (win probability).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SKPipeline

logger = logging.getLogger(__name__)

# Features used by the logistic regression
_NUMERIC_FEATURES = [
    "ydstogo",
    "yardline_100",
    "score_differential",
    "rolling_epa_5",
    "rolling_success_rate_10",
    "drive_play_count",
]
_NUMERIC_FEATURES_BOOL = [
    "is_two_minute_warning",
]
_CATEGORICAL_FEATURES = ["quarter"]


class ScoringIntensityModel:
    """Inhomogeneous Poisson process model for NFL scoring events.

    The model fits a logistic regression to predict whether a given play
    results in a scoring event (touchdown or made field goal).  The
    play-level probability is then divided by the average seconds-per-play
    to produce a continuous-time intensity λ(t).

    Attributes:
        avg_seconds_per_play: Denominator used when converting probability
            to per-second intensity.
        pipeline_: Fitted sklearn Pipeline (set after ``fit``).
        is_fitted_: Boolean flag.
    """

    def __init__(
        self,
        avg_seconds_per_play: float = 40.0,
        logistic_C: float = 1.0,
        logistic_max_iter: int = 1000,
    ) -> None:
        """Initialise the scoring intensity model.

        Args:
            avg_seconds_per_play: Mean play duration in seconds.  Used to
                convert P(score | play) → λ(t) in events/second.
            logistic_C: Inverse regularisation strength for LogisticRegression.
            logistic_max_iter: Maximum iterations for the logistic solver.
        """
        self.avg_seconds_per_play = avg_seconds_per_play
        self.logistic_C = logistic_C
        self.logistic_max_iter = logistic_max_iter
        self.pipeline_: Optional[SKPipeline] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, pbp_features: pd.DataFrame) -> "ScoringIntensityModel":
        """Fit the logistic regression scoring model.

        Args:
            pbp_features: Feature-augmented PBP DataFrame as produced by
                :func:`src.feature_engineering.game_state_features.build_game_state_features`.
                Must contain ``touchdown`` and (optionally) ``field_goal_result``
                and ``play_type`` columns plus the feature columns.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If no scoring events exist in the training data.
        """
        df = pbp_features.copy()

        # ------------------------------------------------------------------
        # Build target: scoring event = TD or made FG
        # ------------------------------------------------------------------
        scoring = pd.Series(0, index=df.index, dtype=int)
        if "touchdown" in df.columns:
            scoring |= df["touchdown"].fillna(0).astype(int)
        if "field_goal_result" in df.columns:
            scoring |= (df["field_goal_result"] == "made").astype(int)
        elif "play_type" in df.columns:
            scoring |= (df["play_type"] == "field_goal").astype(int)

        if scoring.sum() == 0:
            raise ValueError("No scoring events found in training data.")
        logger.info(
            "Training on %d plays with %d scoring events (%.2f%%).",
            len(df), scoring.sum(), 100.0 * scoring.mean(),
        )

        # ------------------------------------------------------------------
        # Prepare feature matrix
        # ------------------------------------------------------------------
        X = self._prepare_features(df)
        y = scoring.loc[X.index].to_numpy()

        # ------------------------------------------------------------------
        # Build sklearn pipeline
        # ------------------------------------------------------------------
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        logger.info("ScoringIntensityModel fitted.")
        return self

    def predict_intensity(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the per-second scoring intensity for each play.

        Args:
            features: Feature DataFrame with the same schema as the training
                data (produced by ``build_game_state_features``).

        Returns:
            Float array of intensities (events per second) of length
            ``len(features)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_fitted()
        X = self._prepare_features(features)
        prob = self.pipeline_.predict_proba(X)[:, 1]
        return prob / self.avg_seconds_per_play

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict the play-level probability of a scoring event.

        Args:
            features: Feature DataFrame.

        Returns:
            Float array of probabilities in [0, 1].
        """
        self._check_fitted()
        X = self._prepare_features(features)
        return self.pipeline_.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and sanitise the feature columns for modelling.

        Args:
            df: Input DataFrame (may have extra columns).

        Returns:
            DataFrame containing only the model feature columns,
            with NaNs coerced to float where needed.
        """
        cols = []
        for c in _NUMERIC_FEATURES + _NUMERIC_FEATURES_BOOL:
            if c in df.columns:
                cols.append(c)
            else:
                logger.warning("Feature column '%s' not found; skipping.", c)
        for c in _CATEGORICAL_FEATURES:
            if c in df.columns:
                cols.append(c)

        X = df[cols].copy()
        # Coerce booleans to int for the transformer
        for c in _NUMERIC_FEATURES_BOOL:
            if c in X.columns:
                X[c] = X[c].astype(float)
        for c in _CATEGORICAL_FEATURES:
            if c in X.columns:
                X[c] = X[c].astype(str)
        return X

    def _build_pipeline(self) -> SKPipeline:
        """Construct the sklearn preprocessing + logistic regression pipeline.

        Returns:
            Unfitted ``sklearn.pipeline.Pipeline``.
        """
        numeric_cols = [
            c for c in _NUMERIC_FEATURES + _NUMERIC_FEATURES_BOOL
        ]
        categorical_cols = _CATEGORICAL_FEATURES

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.logistic_C,
                        max_iter=self.logistic_max_iter,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        return pipeline

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted.

        Args:
            None

        Returns:
            None

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict methods.")
