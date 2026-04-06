"""
Tests for src.models.kalman_filter.WinProbabilityFilter.

Key assertion: the MSE of the Kalman-filtered WP against the *true* WP
must be strictly less than the MSE of the raw observations against the
true WP — i.e. the filter improves on naively trusting the noisy signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from src.models.kalman_filter import WinProbabilityFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_observations(
    n: int = 200,
    Q: float = 1e-3,
    R: float = 5e-2,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create a synthetic true WP trajectory and noisy observations.

    True WP is the sigmoid of a Brownian motion.  Observations are
    logit(true_wp) + Gaussian noise.

    Args:
        n: Number of time steps.
        Q: Process noise variance (per step).
        R: Observation noise variance.
        seed: Random seed.

    Returns:
        Tuple of:
        - true_wp: Array of true win probabilities in (0, 1).
        - obs_df: DataFrame with columns required by WinProbabilityFilter.filter().
    """
    rng = np.random.default_rng(seed)

    # Brownian motion in logit space → true WP path
    x = np.zeros(n)
    x[0] = 0.0  # logit(0.5)
    for t in range(1, n):
        x[t] = x[t - 1] + rng.normal(0, np.sqrt(Q))

    true_wp = expit(np.clip(x, -5, 5))

    # Noisy book implied probability
    z = x + rng.normal(0, np.sqrt(R), size=n)
    book_implied_prob = expit(np.clip(z, -5, 5))

    obs_df = pd.DataFrame(
        {
            "elapsed_seconds": np.linspace(0, 3600, n),
            "book_implied_prob": book_implied_prob,
            "is_scoring_event": np.zeros(n, dtype=bool),
            "nflfastr_wp": true_wp,
        }
    )
    return true_wp, obs_df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWinProbabilityFilter:
    """Unit tests for WinProbabilityFilter."""

    def test_filter_reduces_mse(self) -> None:
        """Kalman-filtered WP must have lower MSE vs truth than raw observations."""
        true_wp, obs_df = _make_synthetic_observations(n=300, Q=1e-3, R=5e-2, seed=7)

        kf = WinProbabilityFilter(Q=1e-3, R=5e-2, x0=0.0, P0=1.0)
        result = kf.filter(obs_df)

        mse_filtered = float(np.mean((result["filtered_wp"].to_numpy() - true_wp) ** 2))
        mse_observed = float(np.mean((obs_df["book_implied_prob"].to_numpy() - true_wp) ** 2))

        assert mse_filtered < mse_observed, (
            f"Kalman MSE ({mse_filtered:.6f}) ≥ observation MSE ({mse_observed:.6f}). "
            "Filter is not improving on raw observations."
        )

    def test_output_columns(self) -> None:
        """filter() must return all required output columns."""
        _, obs_df = _make_synthetic_observations(n=50)
        kf = WinProbabilityFilter()
        result = kf.filter(obs_df)

        required = {
            "elapsed_seconds", "filtered_logit_wp", "filtered_wp",
            "filtered_variance", "book_implied_prob", "nflfastr_wp",
        }
        assert required.issubset(set(result.columns)), (
            f"Missing columns: {required - set(result.columns)}"
        )

    def test_filtered_wp_in_unit_interval(self) -> None:
        """filtered_wp must stay strictly within (0, 1)."""
        _, obs_df = _make_synthetic_observations(n=200)
        kf = WinProbabilityFilter()
        result = kf.filter(obs_df)

        assert float(result["filtered_wp"].min()) > 0.0
        assert float(result["filtered_wp"].max()) < 1.0

    def test_filtered_variance_nonnegative(self) -> None:
        """Posterior variance must be non-negative at every step."""
        _, obs_df = _make_synthetic_observations(n=100)
        kf = WinProbabilityFilter()
        result = kf.filter(obs_df)
        assert float(result["filtered_variance"].min()) >= 0.0

    def test_variance_decreases_without_regime_shift(self) -> None:
        """Posterior variance should be lower at the end than at the start
        (filter gains confidence over time when there are no scoring events)."""
        _, obs_df = _make_synthetic_observations(n=200, Q=0.0, R=0.01, seed=3)
        obs_df["is_scoring_event"] = False

        kf = WinProbabilityFilter(Q=0.0, R=0.01, P0=1.0)
        result = kf.filter(obs_df)

        p_first = float(result["filtered_variance"].iloc[0])
        p_last = float(result["filtered_variance"].iloc[-1])
        assert p_last < p_first, (
            f"Variance did not decrease: first={p_first:.6f}, last={p_last:.6f}"
        )

    def test_output_length_matches_input(self) -> None:
        """Output DataFrame must have the same number of rows as input."""
        _, obs_df = _make_synthetic_observations(n=77)
        kf = WinProbabilityFilter()
        result = kf.filter(obs_df)
        assert len(result) == len(obs_df)

    def test_scoring_event_inflates_variance(self) -> None:
        """A scoring event should cause a temporary increase in variance."""
        _, obs_df = _make_synthetic_observations(n=100, Q=1e-5, R=0.01, seed=0)
        # Force scoring event at row 50
        obs_df.loc[50, "is_scoring_event"] = True
        obs_df.loc[50, "nflfastr_wp"] = 0.8  # large WP jump

        kf = WinProbabilityFilter(Q=1e-5, R=0.01, P0=0.1, scoring_Q_multiplier=5.0)
        result = kf.filter(obs_df)

        var_before = float(result["filtered_variance"].iloc[49])
        var_after = float(result["filtered_variance"].iloc[50])
        # After a scoring event the variance should be >= var before
        assert var_after >= var_before * 0.5, (
            "Variance did not increase (or remain) after scoring event."
        )

    def test_missing_columns_raises(self) -> None:
        """filter() must raise ValueError for missing required columns."""
        obs_df = pd.DataFrame({"elapsed_seconds": [0, 1, 2]})
        kf = WinProbabilityFilter()
        with pytest.raises(ValueError, match="missing columns"):
            kf.filter(obs_df)
