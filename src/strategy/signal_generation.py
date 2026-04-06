"""
Mispricing signal generation.

Compares model-derived fair probabilities (from the Kalman filter and
Monte Carlo simulator) against the book's quoted odds to identify
statistically significant edge.

Finance analogy: computing the "basis" between the model mid-price and
the quoted market price, then generating a directional signal when the
basis exceeds the transaction-cost threshold.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_signals(
    filtered_df: pd.DataFrame,
    mc_fair_probs: pd.DataFrame,
    edge_threshold: float = 0.03,
) -> pd.DataFrame:
    """Generate home/away/no-bet signals by comparing fair price to book price.

    Merges the Kalman-filtered win probability estimates with the Monte Carlo
    fair odds, then computes the signed edge (model probability − book
    implied probability) for the home team.

    A signal is generated when::

        |edge| > edge_threshold

    Signal strength is normalised by the Kalman filter's posterior variance,
    so high-confidence model estimates produce stronger signals.

    Args:
        filtered_df: Output of :meth:`~src.models.kalman_filter.WinProbabilityFilter.filter`,
            with columns: ``game_id`` (optional), ``elapsed_seconds``,
            ``filtered_wp``, ``filtered_variance``, ``book_implied_prob``.
        mc_fair_probs: DataFrame with columns: ``elapsed_seconds``,
            ``mc_home_win_prob``, ``mc_away_win_prob``.  Typically one row
            per play where an MC simulation was run.
            If ``game_id`` is present it will be used in the merge key.
        edge_threshold: Minimum absolute edge to trigger a bet signal.
            Defaults to 0.03 (3 percentage points).

    Returns:
        DataFrame with columns:
        - ``game_id`` (if present in inputs)
        - ``elapsed_seconds``
        - ``edge`` — mc_home_win_prob − book_implied_prob
        - ``signal`` — one of ``"bet_home"``, ``"bet_away"``, ``"no_bet"``
        - ``signal_strength`` — ``abs(edge) / filtered_variance``
        - ``book_implied_prob``
        - ``book_decimal_odds`` — 1 / book_implied_prob
        - ``fair_decimal_odds`` — 1 / mc_home_win_prob
        - ``mc_home_win_prob``

    Raises:
        ValueError: If required columns are missing from either DataFrame.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    required_filtered = {"elapsed_seconds", "filtered_wp", "filtered_variance", "book_implied_prob"}
    required_mc = {"elapsed_seconds", "mc_home_win_prob"}

    missing_f = required_filtered - set(filtered_df.columns)
    if missing_f:
        raise ValueError(f"filtered_df missing columns: {missing_f}")
    missing_mc = required_mc - set(mc_fair_probs.columns)
    if missing_mc:
        raise ValueError(f"mc_fair_probs missing columns: {missing_mc}")

    # ------------------------------------------------------------------
    # Determine merge keys
    # ------------------------------------------------------------------
    merge_keys = ["elapsed_seconds"]
    if "game_id" in filtered_df.columns and "game_id" in mc_fair_probs.columns:
        merge_keys = ["game_id", "elapsed_seconds"]

    # ------------------------------------------------------------------
    # Merge (nearest-time match for MC estimates, which are sparser)
    # ------------------------------------------------------------------
    left = filtered_df.sort_values(merge_keys)
    right = mc_fair_probs.sort_values(merge_keys).drop_duplicates(subset=merge_keys)

    merged = pd.merge_asof(
        left,
        right[merge_keys + ["mc_home_win_prob"]],
        on="elapsed_seconds",
        by=[k for k in merge_keys if k != "elapsed_seconds"] or None,  # type: ignore[arg-type]
        direction="nearest",
    )

    # ------------------------------------------------------------------
    # Compute edge and signal
    # ------------------------------------------------------------------
    mc_prob = merged["mc_home_win_prob"].fillna(merged["filtered_wp"])
    book_prob = merged["book_implied_prob"].clip(0.01, 0.99)
    var = merged["filtered_variance"].clip(lower=1e-8)

    edge = mc_prob - book_prob

    signal = np.where(
        edge > edge_threshold,
        "bet_home",
        np.where(edge < -edge_threshold, "bet_away", "no_bet"),
    )

    signal_strength = np.abs(edge) / var

    # Safe decimal odds conversion
    book_decimal_odds = (1.0 / book_prob.clip(1e-6)).clip(1.01, 50.0)
    fair_decimal_odds = (1.0 / mc_prob.clip(1e-6)).clip(1.01, 50.0)

    result = pd.DataFrame(
        {
            "elapsed_seconds": merged["elapsed_seconds"].to_numpy(),
            "edge": edge.to_numpy(),
            "signal": signal,
            "signal_strength": signal_strength.to_numpy(),
            "book_implied_prob": book_prob.to_numpy(),
            "book_decimal_odds": book_decimal_odds.to_numpy(),
            "fair_decimal_odds": fair_decimal_odds.to_numpy(),
            "mc_home_win_prob": mc_prob.to_numpy(),
        }
    )

    if "game_id" in merged.columns:
        result.insert(0, "game_id", merged["game_id"].to_numpy())

    n_bets = int((result["signal"] != "no_bet").sum())
    logger.info(
        "Signals generated: %d total, %d actionable (threshold=%.3f).",
        len(result), n_bets, edge_threshold,
    )

    return result
