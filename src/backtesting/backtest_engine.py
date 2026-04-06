"""
Walk-forward backtesting engine.

Simulates a live betting operation over a sequence of historical games,
processing each game sequentially (no look-ahead into future games) and
updating the bankroll after each game resolves.

For each play where the strategy generates a bet signal, the engine:

1. Computes Monte Carlo fair odds from the current game state.
2. Sizes the bet using fractional Kelly.
3. Resolves the bet at full-time (binary win/loss).
4. Updates the bankroll.

Finance analogy: walk-forward optimisation / paper-trading simulation —
identical to how algorithmic trading strategies are backtested on time-series
data with strict temporal ordering to prevent look-ahead bias.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.feature_engineering.game_state_features import build_game_state_features
from src.feature_engineering.odds_path_features import build_odds_path_features
from src.models.kalman_filter import WinProbabilityFilter
from src.simulation.monte_carlo import GameSimulator
from src.strategy.signal_generation import generate_signals
from src.strategy.kelly_sizing import size_portfolio

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """Event-driven walk-forward backtest engine.

    Attributes:
        initial_bankroll: Starting bankroll in dollars.
        bankroll: Current bankroll (updated during :meth:`run`).
    """

    def __init__(self, initial_bankroll: float = 10_000.0) -> None:
        """Initialise the backtesting engine.

        Args:
            initial_bankroll: Starting capital in dollars.
        """
        self.initial_bankroll = float(initial_bankroll)
        self.bankroll = float(initial_bankroll)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        games: list[str],
        pbp: pd.DataFrame,
        odds: pd.DataFrame,
        scoring_model,
        hawkes_model,
        kf_params: dict,
        mc_sims: int = 5_000,
        kelly_fraction: float = 0.25,
        edge_threshold: float = 0.03,
        seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Execute the walk-forward backtest over the specified games.

        Args:
            games: List of ``game_id`` strings in chronological order.
            pbp: Full play-by-play DataFrame covering all games.
            odds: Full synthetic odds DataFrame covering all games.
            scoring_model: Fitted :class:`~src.models.poisson_scoring.ScoringIntensityModel`.
            hawkes_model: Fitted :class:`~src.models.hawkes_process.HawkesOddsModel`
                (used for diagnostics, not bet sizing directly).
            kf_params: Dict of Kalman filter keyword arguments (Q, R, x0, P0,
                scoring_Q_multiplier).
            mc_sims: Number of Monte Carlo paths per in-game pricing.
            kelly_fraction: Fractional Kelly multiplier.
            edge_threshold: Minimum edge to trigger a signal.
            seed: Base random seed for MC simulations.

        Returns:
            Ledger DataFrame with columns:
            - ``game_id``
            - ``elapsed_seconds``
            - ``signal``
            - ``edge``
            - ``stake_dollars``
            - ``book_decimal_odds``
            - ``result`` — ``"W"``, ``"L"``, or ``"P"`` (push/tie)
            - ``pnl`` — profit or loss in dollars for this bet
            - ``cumulative_bankroll``
        """
        self.bankroll = self.initial_bankroll
        ledger_rows: list[dict] = []

        kf = WinProbabilityFilter(**kf_params)
        simulator = GameSimulator(scoring_model=scoring_model, n_sims=mc_sims)

        for game_idx, game_id in enumerate(games):
            logger.info(
                "Backtesting game %d/%d: %s | bankroll=$%.2f",
                game_idx + 1, len(games), game_id, self.bankroll,
            )

            # ------------------------------------------------------------------
            # Extract game data
            # ------------------------------------------------------------------
            game_pbp = pbp[pbp["game_id"] == game_id].copy()
            game_odds = odds[odds["game_id"] == game_id].copy()

            if len(game_pbp) == 0 or len(game_odds) == 0:
                logger.warning("No data for game %s, skipping.", game_id)
                continue

            # ------------------------------------------------------------------
            # Feature engineering
            # ------------------------------------------------------------------
            try:
                game_pbp_feat = build_game_state_features(game_pbp)
            except Exception as exc:
                logger.warning("Feature engineering failed for %s: %s", game_id, exc)
                continue

            game_odds_feat = build_odds_path_features(game_odds)

            # ------------------------------------------------------------------
            # Kalman filter
            # ------------------------------------------------------------------
            filtered_df = kf.filter_game(game_pbp_feat, game_odds_feat)
            filtered_df["game_id"] = game_id

            # ------------------------------------------------------------------
            # Determine game outcome
            # ------------------------------------------------------------------
            home_score, away_score = self._get_final_scores(game_pbp)
            if home_score is None:
                logger.warning("Could not determine outcome for %s, skipping.", game_id)
                continue
            home_wins = home_score > away_score

            # ------------------------------------------------------------------
            # Iterate over plays, generate signals, size bets
            # ------------------------------------------------------------------
            plays = game_pbp_feat.sort_values("elapsed_seconds")
            total_seconds = float(plays["elapsed_seconds"].max()) + 1.0

            # Sample plays at which to run MC (every N plays for speed)
            sample_interval = max(1, len(plays) // 40)
            sampled_plays = plays.iloc[::sample_interval].copy()

            mc_records: list[dict] = []
            for _, play_row in sampled_plays.iterrows():
                elapsed = float(play_row["elapsed_seconds"])
                remaining = max(total_seconds - elapsed, 0.0)
                if remaining < 60:
                    continue

                state = self._build_current_state(play_row, home_score=0, away_score=0)
                # Use running scores from PBP at this point
                plays_so_far = plays[plays["elapsed_seconds"] <= elapsed]
                if "posteam_score" in plays_so_far.columns and len(plays_so_far) > 0:
                    last_play = plays_so_far.iloc[-1]
                    state["home_score"] = int(last_play.get("posteam_score", 0) or 0)
                    state["away_score"] = int(last_play.get("defteam_score", 0) or 0)

                mc_result = simulator.fair_odds(
                    state, remaining, seed=(seed or 0) + game_idx
                )
                mc_records.append(
                    {
                        "elapsed_seconds": elapsed,
                        "mc_home_win_prob": mc_result["home_win_prob"],
                        "mc_away_win_prob": mc_result["away_win_prob"],
                    }
                )

            if not mc_records:
                continue

            mc_df = pd.DataFrame(mc_records)

            # ------------------------------------------------------------------
            # Signal generation
            # ------------------------------------------------------------------
            try:
                signals_df = generate_signals(
                    filtered_df, mc_df, edge_threshold=edge_threshold
                )
                signals_df["game_id"] = game_id
            except Exception as exc:
                logger.warning("Signal generation failed for %s: %s", game_id, exc)
                continue

            # ------------------------------------------------------------------
            # Kelly sizing & bet resolution
            # ------------------------------------------------------------------
            sized_df = size_portfolio(
                signals_df,
                bankroll=self.bankroll,
                fraction=kelly_fraction,
            )

            for _, bet_row in sized_df.iterrows():
                stake = float(bet_row["stake_dollars"])
                if stake <= 0:
                    continue

                signal = str(bet_row["signal"])
                odds_val = float(bet_row["book_decimal_odds"])
                edge = float(bet_row["edge"])

                # Resolve: home bet wins if home_wins, away bet wins if not
                if signal == "bet_home":
                    won = home_wins
                elif signal == "bet_away":
                    won = not home_wins
                else:
                    continue

                if home_score == away_score:
                    result_str = "P"
                    pnl = 0.0
                else:
                    result_str = "W" if won else "L"
                    pnl = stake * (odds_val - 1.0) if won else -stake

                self.bankroll += pnl

                ledger_rows.append(
                    {
                        "game_id": game_id,
                        "elapsed_seconds": float(bet_row["elapsed_seconds"]),
                        "signal": signal,
                        "edge": edge,
                        "stake_dollars": stake,
                        "book_decimal_odds": odds_val,
                        "result": result_str,
                        "pnl": pnl,
                        "cumulative_bankroll": self.bankroll,
                    }
                )

        if not ledger_rows:
            logger.warning("No bets placed during backtest.")
            return pd.DataFrame(
                columns=[
                    "game_id", "elapsed_seconds", "signal", "edge",
                    "stake_dollars", "book_decimal_odds", "result", "pnl",
                    "cumulative_bankroll",
                ]
            )

        ledger = pd.DataFrame(ledger_rows).sort_values(
            ["game_id", "elapsed_seconds"]
        ).reset_index(drop=True)
        logger.info(
            "Backtest complete: %d bets, final bankroll=$%.2f",
            len(ledger), self.bankroll,
        )
        return ledger

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(self, ledger: pd.DataFrame) -> dict:
        """Compute summary performance metrics from the backtest ledger.

        Args:
            ledger: Output DataFrame from :meth:`run`.

        Returns:
            Dict with keys:
            - ``total_bets`` (int)
            - ``win_rate`` (float, fraction)
            - ``avg_edge`` (float)
            - ``total_pnl`` (float, dollars)
            - ``roi`` (float, fraction of initial bankroll)
            - ``max_drawdown`` (float, dollars)
            - ``sharpe_ratio`` (float, annualised if enough data)
            - ``final_bankroll`` (float, dollars)
        """
        if len(ledger) == 0:
            return {
                "total_bets": 0,
                "win_rate": 0.0,
                "avg_edge": 0.0,
                "total_pnl": 0.0,
                "roi": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "final_bankroll": self.initial_bankroll,
            }

        resolved = ledger[ledger["result"].isin(["W", "L"])].copy()

        total_bets = len(resolved)
        win_rate = float((resolved["result"] == "W").mean()) if total_bets > 0 else 0.0
        avg_edge = float(ledger["edge"].mean())
        total_pnl = float(ledger["pnl"].sum())
        roi = total_pnl / self.initial_bankroll

        # Max drawdown
        bankroll_curve = np.array([self.initial_bankroll] + list(ledger["cumulative_bankroll"]))
        running_max = np.maximum.accumulate(bankroll_curve)
        drawdowns = running_max - bankroll_curve
        max_drawdown = float(drawdowns.max())

        # Sharpe ratio (per-bet returns)
        returns = resolved["pnl"] / resolved["stake_dollars"]
        sharpe = float(returns.mean() / returns.std()) if len(returns) > 1 and returns.std() > 0 else 0.0

        final_bankroll = float(ledger["cumulative_bankroll"].iloc[-1])

        return {
            "total_bets": total_bets,
            "win_rate": win_rate,
            "avg_edge": avg_edge,
            "total_pnl": total_pnl,
            "roi": roi,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "final_bankroll": final_bankroll,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_final_scores(game_pbp: pd.DataFrame) -> tuple[Optional[int], Optional[int]]:
        """Extract the final home and away scores from a game's PBP.

        Args:
            game_pbp: Single-game PBP DataFrame.

        Returns:
            Tuple of (home_score, away_score) as integers,
            or (None, None) if scores are unavailable.
        """
        if "posteam_score" not in game_pbp.columns or "defteam_score" not in game_pbp.columns:
            return None, None

        last = game_pbp.sort_values("elapsed_seconds").iloc[-1]
        home = last.get("posteam_score")
        away = last.get("defteam_score")

        if pd.isna(home) or pd.isna(away):
            return None, None

        return int(home), int(away)

    @staticmethod
    def _build_current_state(play_row: pd.Series, home_score: int, away_score: int) -> dict:
        """Construct the current_state dict expected by :class:`~src.simulation.monte_carlo.GameSimulator`.

        Args:
            play_row: A single-row Series from the PBP features DataFrame.
            home_score: Current home score.
            away_score: Current away score.

        Returns:
            Dict compatible with ``GameSimulator.simulate_from_state``.
        """
        return {
            "home_score": home_score,
            "away_score": away_score,
            "possession": "home",  # simplified; full model would track this
            "yardline_100": int(play_row.get("yardline_100", 75) or 75),
            "down": int(play_row.get("down", 1) or 1),
            "ydstogo": int(play_row.get("ydstogo", 10) or 10),
            "elapsed_seconds": float(play_row.get("elapsed_seconds", 0.0) or 0.0),
        }
