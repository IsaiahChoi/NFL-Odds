"""
Tests for src.models.poisson_scoring.ScoringIntensityModel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.poisson_scoring import ScoringIntensityModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_pbp(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic PBP-like DataFrame for testing.

    Scoring events are correlated with yardline_100 (more likely near end
    zone) so that the logistic regression learns the correct direction.

    Args:
        n: Number of rows (plays).
        seed: Random seed.

    Returns:
        DataFrame with all columns expected by ScoringIntensityModel.
    """
    rng = np.random.default_rng(seed)

    yardline = rng.integers(1, 100, size=n)

    # Scoring probability proportional to proximity to end zone
    # yardline_100=1 → very likely; yardline_100=99 → very unlikely
    score_prob = np.clip(0.5 * np.exp(-0.04 * yardline), 0.01, 0.95)
    touchdown = rng.binomial(1, score_prob)

    data = {
        "game_id": ["GAME_001"] * n,
        "elapsed_seconds": np.linspace(0, 3500, n),
        "down": rng.integers(1, 5, size=n),
        "ydstogo": rng.integers(1, 20, size=n),
        "yardline_100": yardline,
        "score_differential": rng.integers(-21, 21, size=n),
        "is_two_minute_warning": rng.choice([False, True], size=n, p=[0.9, 0.1]),
        "rolling_epa_5": rng.normal(0, 0.5, size=n),
        "rolling_success_rate_10": rng.uniform(0.2, 0.8, size=n),
        "drive_play_count": rng.integers(0, 15, size=n),
        "quarter": rng.integers(1, 6, size=n),
        "posteam": ["HOM"] * n,
        "epa": rng.normal(0, 0.5, size=n),
        "touchdown": touchdown.astype(int),
        "field_goal_result": [""] * n,
    }
    df = pd.DataFrame(data)
    # Guarantee at least a few positive labels
    if df["touchdown"].sum() < 3:
        df.loc[df["yardline_100"].nsmallest(3).index, "touchdown"] = 1

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoringIntensityModel:
    """Unit tests for ScoringIntensityModel."""

    def test_fit_returns_self(self) -> None:
        """fit() should return the model instance (for chaining)."""
        df = _make_synthetic_pbp()
        model = ScoringIntensityModel()
        result = model.fit(df)
        assert result is model

    def test_is_fitted_after_fit(self) -> None:
        """is_fitted_ should be True after fit()."""
        df = _make_synthetic_pbp()
        model = ScoringIntensityModel()
        assert not model.is_fitted_
        model.fit(df)
        assert model.is_fitted_

    def test_probabilities_in_unit_interval(self) -> None:
        """predict_proba() must return values in [0, 1]."""
        df = _make_synthetic_pbp(n=50)
        model = ScoringIntensityModel().fit(df)
        probs = model.predict_proba(df)
        assert probs.shape == (len(df),), "Shape mismatch"
        assert float(probs.min()) >= 0.0, "Negative probability"
        assert float(probs.max()) <= 1.0, "Probability > 1"

    def test_intensity_nonnegative(self) -> None:
        """predict_intensity() must return non-negative values."""
        df = _make_synthetic_pbp(n=50)
        model = ScoringIntensityModel(avg_seconds_per_play=40.0).fit(df)
        intensities = model.predict_intensity(df)
        assert float(intensities.min()) >= 0.0

    def test_red_zone_higher_intensity_than_own_territory(self) -> None:
        """A red-zone play should have higher scoring intensity than a play
        deep in own territory, all else being equal.

        Uses a large, structured training set so that the logistic regression
        reliably learns the yardline_100 relationship.
        """
        # Large training set where scoring probability is clearly yardline-dependent
        df = _make_synthetic_pbp(n=500, seed=7)
        model = ScoringIntensityModel().fit(df)

        # Build many red-zone and own-territory test plays
        base = df.iloc[[0]].copy()

        red_zone_rows = []
        own_territory_rows = []
        for yl_red, yl_own in [(3, 85), (5, 90), (8, 80), (10, 75)]:
            rz = base.copy()
            rz["yardline_100"] = yl_red
            rz["down"] = 1
            rz["ydstogo"] = yl_red
            red_zone_rows.append(rz)

            ot = base.copy()
            ot["yardline_100"] = yl_own
            ot["down"] = 1
            ot["ydstogo"] = 10
            own_territory_rows.append(ot)

        red_zone_df = pd.concat(red_zone_rows, ignore_index=True)
        own_territory_df = pd.concat(own_territory_rows, ignore_index=True)

        p_red_mean = model.predict_proba(red_zone_df).mean()
        p_own_mean = model.predict_proba(own_territory_df).mean()

        assert p_red_mean > p_own_mean, (
            f"Expected mean red-zone prob {p_red_mean:.4f} > "
            f"own-territory prob {p_own_mean:.4f}"
        )

    def test_predict_without_fit_raises(self) -> None:
        """Calling predict methods before fit() should raise RuntimeError."""
        df = _make_synthetic_pbp()
        model = ScoringIntensityModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_proba(df)

    def test_no_scoring_events_raises(self) -> None:
        """fit() should raise ValueError when there are no scoring events."""
        df = _make_synthetic_pbp()
        df["touchdown"] = 0
        df["field_goal_result"] = ""
        model = ScoringIntensityModel()
        with pytest.raises(ValueError, match="No scoring events"):
            model.fit(df)
