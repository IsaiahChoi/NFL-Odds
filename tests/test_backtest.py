"""
Tests for src.backtesting.backtest_engine.WalkForwardBacktest.

Runs the full engine on two tiny synthetic games and verifies:
- The ledger has the correct columns.
- Bankroll arithmetic is consistent (win → bankroll increases, loss → decreases).
- summary_stats returns a well-formed dict with expected keys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_engine import WalkForwardBacktest
from src.models.poisson_scoring import ScoringIntensityModel
from src.models.hawkes_process import HawkesOddsModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_PLAYS = 60  # plays per synthetic game


def _make_synthetic_pbp(game_id: str, seed: int = 0) -> pd.DataFrame:
    """Build a minimal synthetic PBP DataFrame for one game.

    Args:
        game_id: Game identifier string.
        seed: Random seed.

    Returns:
        DataFrame with all columns required by the pipeline.
    """
    rng = np.random.default_rng(seed)
    n = _N_PLAYS
    elapsed = np.linspace(0, 3500, n)
    wp = np.clip(0.5 + np.cumsum(rng.normal(0, 0.03, n)), 0.05, 0.95)

    df = pd.DataFrame(
        {
            "game_id": [game_id] * n,
            "elapsed_seconds": elapsed,
            "game_seconds_remaining": 3600.0 - elapsed,
            "quarter_seconds_remaining": np.clip(3600.0 - elapsed, 0, 900),
            "play_id": range(n),
            "posteam": ["HOM"] * n,
            "defteam": ["AWY"] * n,
            "home_team": ["HOM"] * n,
            "away_team": ["AWY"] * n,
            "down": rng.integers(1, 5, size=n),
            "ydstogo": rng.integers(1, 20, size=n),
            "yardline_100": rng.integers(1, 100, size=n),
            "score_differential": rng.integers(-14, 14, size=n),
            "posteam_score": np.clip(np.cumsum(rng.integers(0, 2, size=n)), 0, 50).astype(int),
            "defteam_score": np.clip(np.cumsum(rng.integers(0, 2, size=n)), 0, 50).astype(int),
            "ep": rng.uniform(0, 4, size=n),
            "epa": rng.normal(0, 0.5, size=n),
            "wp": wp,
            "wpa": rng.normal(0, 0.03, size=n),
            "play_type": rng.choice(["run", "pass"], size=n),
            "touchdown": (rng.random(n) < 0.05).astype(int),
            "fumble_lost": (rng.random(n) < 0.02).astype(int),
            "interception": (rng.random(n) < 0.02).astype(int),
            "field_goal_result": [""] * n,
            "drive": (np.arange(n) // 10).astype(int),
            "series_success": rng.choice([0, 1], size=n),
            "desc": ["play"] * n,
            "season": [2022] * n,
            "week": [1] * n,
        }
    )
    return df


def _make_synthetic_odds(game_id: str, pbp: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Build synthetic odds from a PBP DataFrame.

    Args:
        game_id: Game identifier.
        pbp: Single-game PBP DataFrame with ``wp`` and ``elapsed_seconds``.
        seed: Random seed.

    Returns:
        Odds DataFrame compatible with the pipeline.
    """
    from src.data_ingestion.odds_simulator import SyntheticOddsGenerator

    gen = SyntheticOddsGenerator(seed=seed)
    return gen.generate(pbp)


def _fit_scoring_model(pbp: pd.DataFrame) -> ScoringIntensityModel:
    """Fit a minimal ScoringIntensityModel on synthetic data.

    Args:
        pbp: PBP DataFrame with feature columns.

    Returns:
        Fitted model instance.
    """
    from src.feature_engineering.game_state_features import build_game_state_features

    # Ensure at least a few scoring events
    pbp = pbp.copy()
    pbp.loc[pbp.index[:3], "touchdown"] = 1

    feat = build_game_state_features(pbp)
    model = ScoringIntensityModel(avg_seconds_per_play=40.0)
    model.fit(feat)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

REQUIRED_LEDGER_COLUMNS = {
    "game_id",
    "elapsed_seconds",
    "signal",
    "edge",
    "stake_dollars",
    "book_decimal_odds",
    "result",
    "pnl",
    "cumulative_bankroll",
}

KF_PARAMS = {"Q": 1e-4, "R": 1e-2, "x0": 0.0, "P0": 1.0, "scoring_Q_multiplier": 3.0}


class TestWalkForwardBacktest:
    """Integration tests for the walk-forward backtest engine."""

    @pytest.fixture
    def two_game_data(self):
        """Synthesise two-game PBP, odds, and scoring model."""
        pbp1 = _make_synthetic_pbp("GAME_001", seed=1)
        pbp2 = _make_synthetic_pbp("GAME_002", seed=2)
        pbp = pd.concat([pbp1, pbp2], ignore_index=True)

        odds1 = _make_synthetic_odds("GAME_001", pbp1, seed=1)
        odds2 = _make_synthetic_odds("GAME_002", pbp2, seed=2)
        odds = pd.concat([odds1, odds2], ignore_index=True)

        scoring_model = _fit_scoring_model(pbp)
        hawkes_model = HawkesOddsModel()
        # Manually set parameters to skip expensive MLE for test speed
        hawkes_model.mu_ = 0.5
        hawkes_model.alpha_ = 0.2
        hawkes_model.beta_ = 1.0
        hawkes_model.is_fitted_ = True

        return {
            "games": ["GAME_001", "GAME_002"],
            "pbp": pbp,
            "odds": odds,
            "scoring_model": scoring_model,
            "hawkes_model": hawkes_model,
        }

    def test_ledger_columns(self, two_game_data) -> None:
        """Ledger DataFrame must contain all required columns."""
        engine = WalkForwardBacktest(initial_bankroll=10_000.0)
        ledger = engine.run(
            games=two_game_data["games"],
            pbp=two_game_data["pbp"],
            odds=two_game_data["odds"],
            scoring_model=two_game_data["scoring_model"],
            hawkes_model=two_game_data["hawkes_model"],
            kf_params=KF_PARAMS,
            mc_sims=100,   # fast for tests
        )
        if len(ledger) == 0:
            pytest.skip("No bets generated — cannot check columns.")
        assert REQUIRED_LEDGER_COLUMNS.issubset(set(ledger.columns)), (
            f"Missing columns: {REQUIRED_LEDGER_COLUMNS - set(ledger.columns)}"
        )

    def test_bankroll_updates_correctly(self, two_game_data) -> None:
        """cumulative_bankroll must equal initial + cumsum(pnl)."""
        engine = WalkForwardBacktest(initial_bankroll=10_000.0)
        ledger = engine.run(
            games=two_game_data["games"],
            pbp=two_game_data["pbp"],
            odds=two_game_data["odds"],
            scoring_model=two_game_data["scoring_model"],
            hawkes_model=two_game_data["hawkes_model"],
            kf_params=KF_PARAMS,
            mc_sims=100,
        )
        if len(ledger) == 0:
            pytest.skip("No bets generated — cannot verify bankroll arithmetic.")

        reconstructed = 10_000.0 + ledger["pnl"].cumsum().to_numpy()
        actual = ledger["cumulative_bankroll"].to_numpy()
        np.testing.assert_allclose(actual, reconstructed, atol=1e-6,
                                   err_msg="Bankroll arithmetic inconsistency")

    def test_summary_stats_keys(self, two_game_data) -> None:
        """summary_stats must return all expected keys."""
        engine = WalkForwardBacktest(initial_bankroll=10_000.0)
        ledger = engine.run(
            games=two_game_data["games"],
            pbp=two_game_data["pbp"],
            odds=two_game_data["odds"],
            scoring_model=two_game_data["scoring_model"],
            hawkes_model=two_game_data["hawkes_model"],
            kf_params=KF_PARAMS,
            mc_sims=100,
        )
        stats = engine.summary_stats(ledger)
        expected_keys = {
            "total_bets", "win_rate", "avg_edge", "total_pnl",
            "roi", "max_drawdown", "sharpe_ratio", "final_bankroll",
        }
        assert expected_keys.issubset(set(stats.keys())), (
            f"Missing stats keys: {expected_keys - set(stats.keys())}"
        )

    def test_result_column_values(self, two_game_data) -> None:
        """result column must only contain 'W', 'L', or 'P'."""
        engine = WalkForwardBacktest(initial_bankroll=10_000.0)
        ledger = engine.run(
            games=two_game_data["games"],
            pbp=two_game_data["pbp"],
            odds=two_game_data["odds"],
            scoring_model=two_game_data["scoring_model"],
            hawkes_model=two_game_data["hawkes_model"],
            kf_params=KF_PARAMS,
            mc_sims=100,
        )
        if len(ledger) == 0:
            pytest.skip("No bets generated.")
        invalid = set(ledger["result"].unique()) - {"W", "L", "P"}
        assert not invalid, f"Invalid result values: {invalid}"

    def test_empty_games_list_returns_empty_ledger(self) -> None:
        """An empty games list should return an empty ledger with correct columns."""
        pbp = _make_synthetic_pbp("X", seed=0)
        odds = _make_synthetic_odds("X", pbp, seed=0)
        scoring_model = _fit_scoring_model(pbp)
        hawkes_model = HawkesOddsModel()
        hawkes_model.mu_ = 0.3
        hawkes_model.alpha_ = 0.1
        hawkes_model.beta_ = 1.0
        hawkes_model.is_fitted_ = True

        engine = WalkForwardBacktest(initial_bankroll=5_000.0)
        ledger = engine.run(
            games=[],
            pbp=pbp,
            odds=odds,
            scoring_model=scoring_model,
            hawkes_model=hawkes_model,
            kf_params=KF_PARAMS,
            mc_sims=50,
        )
        assert len(ledger) == 0
        assert REQUIRED_LEDGER_COLUMNS.issubset(set(ledger.columns)), (
            f"Missing columns in empty ledger: {REQUIRED_LEDGER_COLUMNS - set(ledger.columns)}"
        )
