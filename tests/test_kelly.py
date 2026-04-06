"""
Tests for src.strategy.kelly_sizing.

Validates analytical correctness of the fractional Kelly formula and
verifies that all exposure caps are enforced correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.kelly_sizing import fractional_kelly_stake, size_portfolio


# ---------------------------------------------------------------------------
# Tests: fractional_kelly_stake
# ---------------------------------------------------------------------------

class TestFractionalKellyStake:
    """Tests for the fractional_kelly_stake function."""

    def test_analytical_value(self) -> None:
        """Check the exact analytical result for a known set of inputs.

        Given:
            edge = 0.05, odds = 2.0, book_implied_prob = 0.50,
            fraction = 0.25, max_stake_pct = 0.05

        True probability: p = 0.50 + 0.05 = 0.55
        Kelly: f* = (p * d - 1) / (d - 1) = (0.55 * 2 - 1) / (2 - 1) = 0.10
        Fractional (0.25): stake = 0.25 * 0.10 = 0.025
        """
        stake = fractional_kelly_stake(
            edge=0.05,
            odds=2.0,
            book_implied_prob=0.50,
            fraction=0.25,
            max_stake_pct=0.05,
        )
        expected = 0.025
        assert abs(stake - expected) < 1e-10, (
            f"Expected stake={expected:.6f}, got {stake:.6f}"
        )

    def test_zero_edge_returns_zero(self) -> None:
        """Edge of exactly zero should yield a zero stake."""
        stake = fractional_kelly_stake(
            edge=0.0,
            odds=2.0,
            book_implied_prob=0.50,
            fraction=0.25,
        )
        assert stake == 0.0

    def test_negative_kelly_returns_zero(self) -> None:
        """Negative Kelly (unfavourable bet) should return zero, not negative."""
        stake = fractional_kelly_stake(
            edge=-0.10,  # significantly negative edge
            odds=2.0,
            book_implied_prob=0.50,
            fraction=0.25,
        )
        assert stake == 0.0

    def test_max_stake_pct_clamp(self) -> None:
        """Stake must not exceed max_stake_pct regardless of edge."""
        stake = fractional_kelly_stake(
            edge=0.40,    # very large edge → raw Kelly ≈ 0.80
            odds=2.0,
            book_implied_prob=0.50,
            fraction=1.0,  # full Kelly
            max_stake_pct=0.05,
        )
        assert stake <= 0.05, f"Stake {stake:.4f} exceeded max_stake_pct=0.05"

    def test_stake_nonnegative(self) -> None:
        """Stake must never be negative."""
        for edge in [-0.5, -0.1, 0.0, 0.01, 0.1, 0.5]:
            stake = fractional_kelly_stake(
                edge=edge,
                odds=1.5,
                book_implied_prob=0.6,
                fraction=0.25,
            )
            assert stake >= 0.0, f"Negative stake for edge={edge}"

    def test_fraction_scales_linearly(self) -> None:
        """Halving fraction should halve the stake (pre-cap)."""
        base = fractional_kelly_stake(edge=0.03, odds=2.5, book_implied_prob=0.38,
                                      fraction=0.50, max_stake_pct=1.0)
        half = fractional_kelly_stake(edge=0.03, odds=2.5, book_implied_prob=0.38,
                                      fraction=0.25, max_stake_pct=1.0)
        assert abs(2 * half - base) < 1e-10, (
            f"Stake not scaling linearly: base={base:.6f}, half={half:.6f}"
        )

    def test_degenerate_odds_returns_zero(self) -> None:
        """Odds <= 1 should be guarded against — return 0.0."""
        stake = fractional_kelly_stake(
            edge=0.10,
            odds=1.0,   # denominator = 0
            book_implied_prob=0.5,
        )
        assert stake == 0.0


# ---------------------------------------------------------------------------
# Tests: size_portfolio
# ---------------------------------------------------------------------------

class TestSizePortfolio:
    """Tests for the size_portfolio function."""

    def _make_signals(self) -> pd.DataFrame:
        """Create a small signals DataFrame with two active bets on different games."""
        return pd.DataFrame(
            {
                "game_id": ["G1", "G1", "G2"],
                "elapsed_seconds": [600.0, 1200.0, 900.0],
                "signal": ["bet_home", "bet_home", "bet_away"],
                "edge": [0.05, 0.04, 0.06],
                "book_decimal_odds": [2.0, 2.2, 1.9],
                "book_implied_prob": [0.50, 0.45, 0.53],
            }
        )

    def test_output_columns_present(self) -> None:
        """size_portfolio must add stake_fraction, stake_dollars, capped columns."""
        df = self._make_signals()
        result = size_portfolio(df, bankroll=10_000.0)
        assert "stake_fraction" in result.columns
        assert "stake_dollars" in result.columns
        assert "capped" in result.columns

    def test_no_bet_signals_filtered_out(self) -> None:
        """Rows with signal='no_bet' must not appear in the output."""
        df = self._make_signals().copy()
        df.loc[0, "signal"] = "no_bet"
        result = size_portfolio(df, bankroll=10_000.0)
        assert "no_bet" not in result["signal"].values

    def test_stake_dollars_equals_fraction_times_bankroll(self) -> None:
        """stake_dollars = stake_fraction * bankroll (before portfolio caps)."""
        df = pd.DataFrame(
            {
                "game_id": ["G1"],
                "signal": ["bet_home"],
                "edge": [0.05],
                "book_decimal_odds": [2.0],
                "book_implied_prob": [0.50],
            }
        )
        bankroll = 10_000.0
        result = size_portfolio(df, bankroll=bankroll)
        if len(result) > 0:
            frac = float(result["stake_fraction"].iloc[0])
            dollars = float(result["stake_dollars"].iloc[0])
            assert abs(dollars - frac * bankroll) < 1e-6

    def test_max_per_bet_enforced(self) -> None:
        """No individual bet should exceed max_per_bet * bankroll."""
        df = self._make_signals()
        max_pct = 0.03
        result = size_portfolio(df, bankroll=10_000.0, max_per_bet=max_pct)
        assert float(result["stake_fraction"].max()) <= max_pct + 1e-9

    def test_max_total_exposure_enforced(self) -> None:
        """Total exposure must not exceed max_total * bankroll."""
        df = self._make_signals()
        max_total = 0.10
        result = size_portfolio(df, bankroll=10_000.0, max_total=max_total, max_per_bet=0.50)
        total_exposure = float(result["stake_fraction"].sum())
        assert total_exposure <= max_total + 1e-9, (
            f"Total exposure {total_exposure:.4f} exceeded cap {max_total}"
        )

    def test_per_game_exposure_enforced(self) -> None:
        """Per-game exposure must not exceed max_per_game * bankroll."""
        # G1 has two bets; their combined fraction should be capped
        df = self._make_signals()
        max_per_game = 0.04
        result = size_portfolio(
            df, bankroll=10_000.0, max_per_game=max_per_game, max_total=1.0
        )
        g1_exposure = float(result[result["game_id"] == "G1"]["stake_fraction"].sum())
        assert g1_exposure <= max_per_game + 1e-9, (
            f"G1 exposure {g1_exposure:.4f} exceeded per-game cap {max_per_game}"
        )

    def test_empty_signals_returns_empty_df(self) -> None:
        """When all signals are 'no_bet', an empty DataFrame is returned."""
        df = pd.DataFrame(
            {
                "game_id": ["G1"],
                "signal": ["no_bet"],
                "edge": [0.01],
                "book_decimal_odds": [2.0],
                "book_implied_prob": [0.5],
            }
        )
        result = size_portfolio(df, bankroll=10_000.0)
        assert len(result) == 0

    def test_negative_bankroll_raises(self) -> None:
        """size_portfolio must reject non-positive bankroll."""
        df = self._make_signals()
        with pytest.raises(ValueError, match="bankroll"):
            size_portfolio(df, bankroll=-100.0)
