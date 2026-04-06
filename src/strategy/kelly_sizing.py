"""
Fractional Kelly position sizing with exposure caps.

Implements the Kelly criterion adapted for binary bets on sports outcomes,
with three layers of exposure management:

1. Per-bet fractional Kelly (quarter-Kelly by default).
2. Per-game exposure cap (prevents overconcentration on one game).
3. Total portfolio exposure cap (prevents excessive leverage).

Finance analogy: this is the portfolio-level leverage optimisation step
— analogous to the Kelly-optimal growth criterion used in algorithmic
trading, constrained by a VaR / drawdown limit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fractional_kelly_stake(
    edge: float,
    odds: float,
    book_implied_prob: float,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """Compute the fractional Kelly optimal stake as a fraction of bankroll.

    The Kelly formula for a binary bet with decimal odds ``d`` and true
    win probability ``p`` is:

    .. math::

        f^* = \\frac{p \\cdot d - 1}{d - 1}

    where ``p = book_implied_prob + edge`` and ``d = odds`` (decimal).

    We return ``fraction * f*`` clipped to ``[0, max_stake_pct]``.

    Args:
        edge: Signed edge: model_probability − book_implied_probability.
            Positive → bet the favoured side; negative → consider reverse bet.
        odds: Decimal odds (e.g. 2.0 for evens).  Must be > 1.
        book_implied_prob: Book's quoted probability for the bet side.
        fraction: Kelly fraction (0.25 = quarter-Kelly).  Lower values reduce
            variance at the cost of expected growth rate.
        max_stake_pct: Hard cap on stake as a fraction of bankroll.

    Returns:
        Stake as a fraction of current bankroll, in ``[0, max_stake_pct]``.
        Returns 0.0 if the bet has non-positive Kelly stake.
    """
    odds = max(odds, 1.01)  # guard against degenerate odds

    # True probability estimate
    p = float(np.clip(book_implied_prob + edge, 1e-6, 1.0 - 1e-6))

    # Kelly formula: f* = (p * odds - 1) / (odds - 1)
    numerator = p * odds - 1.0
    denominator = odds - 1.0

    if denominator <= 0:
        return 0.0

    f_star = numerator / denominator

    if f_star <= 0:
        return 0.0

    stake = fraction * f_star
    return float(np.clip(stake, 0.0, max_stake_pct))


def size_portfolio(
    signals_df: pd.DataFrame,
    bankroll: float,
    fraction: float = 0.25,
    max_per_bet: float = 0.05,
    max_per_game: float = 0.08,
    max_total: float = 0.20,
) -> pd.DataFrame:
    """Size a portfolio of concurrent bets with multi-level exposure caps.

    Processing order:
    1. Compute individual fractional Kelly stakes.
    2. Scale down per-game exposure to ``max_per_game``.
    3. Scale down total portfolio exposure to ``max_total``.
    4. Convert stake fractions to dollar amounts.

    Args:
        signals_df: DataFrame with columns:
            - ``signal`` — ``"bet_home"`` or ``"bet_away"`` (others ignored)
            - ``edge`` — signed edge (positive = bet the stated side)
            - ``book_decimal_odds`` — decimal odds for the signal direction
            - ``book_implied_prob`` — book's quoted probability
            - ``game_id`` — game identifier (required for per-game caps)
        bankroll: Current bankroll in dollars.
        fraction: Fractional Kelly multiplier.
        max_per_bet: Maximum stake fraction per individual bet.
        max_per_game: Maximum total stake fraction across all bets on one game.
        max_total: Maximum total stake fraction across all concurrent bets.

    Returns:
        A copy of ``signals_df`` filtered to actionable signals, with added
        columns:
        - ``stake_fraction`` — stake as a fraction of bankroll after all caps.
        - ``stake_dollars`` — stake in dollar terms.
        - ``capped`` — True if any cap was binding for this bet.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"signal", "edge", "book_decimal_odds", "book_implied_prob"}
    missing = required - set(signals_df.columns)
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    if bankroll <= 0:
        raise ValueError("bankroll must be positive.")

    # Filter to actionable signals only
    active = signals_df[signals_df["signal"].isin(["bet_home", "bet_away"])].copy()
    if len(active) == 0:
        return active.assign(stake_fraction=[], stake_dollars=[], capped=[])

    # ------------------------------------------------------------------
    # Step 1: Raw fractional Kelly per bet
    # ------------------------------------------------------------------
    raw_stakes = np.array([
        fractional_kelly_stake(
            edge=float(row["edge"]),
            odds=float(row["book_decimal_odds"]),
            book_implied_prob=float(row["book_implied_prob"]),
            fraction=fraction,
            max_stake_pct=max_per_bet,
        )
        for _, row in active.iterrows()
    ])

    capped = np.zeros(len(active), dtype=bool)

    # ------------------------------------------------------------------
    # Step 2: Per-game cap
    # ------------------------------------------------------------------
    if "game_id" in active.columns:
        for gid in active["game_id"].unique():
            mask = (active["game_id"] == gid).to_numpy()
            game_total = raw_stakes[mask].sum()
            if game_total > max_per_game:
                scale = max_per_game / game_total
                raw_stakes[mask] *= scale
                capped[mask] = True

    # ------------------------------------------------------------------
    # Step 3: Total exposure cap
    # ------------------------------------------------------------------
    total = raw_stakes.sum()
    if total > max_total:
        scale = max_total / total
        raw_stakes *= scale
        capped[:] = True

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    active = active.copy()
    active["stake_fraction"] = raw_stakes
    active["stake_dollars"] = raw_stakes * bankroll
    active["capped"] = capped

    logger.info(
        "Portfolio sized: %d bets, total exposure=%.2f%%, bankroll=$%.2f",
        len(active), 100.0 * raw_stakes.sum(), bankroll,
    )
    return active
