"""
Odds-path feature engineering.

Transforms the synthetic odds stream into microstructure-style features:
log returns, rolling volatility, momentum, volume surges, and price impact.
These are the analogues of high-frequency trading features computed on an
order book.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_odds_path_features(
    odds_df: pd.DataFrame,
    rolling_vol_window: int = 10,
    momentum_window: int = 5,
    volume_surge_window: int = 20,
    volume_surge_multiplier: float = 2.0,
) -> pd.DataFrame:
    """Compute microstructure features on the synthetic odds time series.

    Features are computed *per game* to avoid look-ahead across game
    boundaries and to ensure rolling windows reset at game start.

    The following columns are added:

    - ``odds_return`` — log return of book implied probability:
      :math:`r_t = \\ln(p_t / p_{t-1})`.
    - ``rolling_odds_vol_10`` — rolling standard deviation of
      ``odds_return`` over the last ``rolling_vol_window`` observations.
    - ``odds_momentum_5`` — sum of ``odds_return`` over the last
      ``momentum_window`` observations (signed momentum).
    - ``volume_surge`` — boolean flag: ``volume > surge_multiplier *
      rolling_mean_volume`` over the last ``volume_surge_window`` obs.
    - ``price_impact`` — ``odds_return / log(volume + 1)`` (signed
      price impact per unit of log-volume).

    Args:
        odds_df: DataFrame as produced by
            :class:`src.data_ingestion.odds_simulator.SyntheticOddsGenerator`.
            Must contain ``game_id``, ``elapsed_seconds``,
            ``book_implied_prob``, and ``volume``.
        rolling_vol_window: Window size for rolling volatility.
        momentum_window: Window size for momentum (sum of returns).
        volume_surge_window: Window size for rolling mean volume comparison.
        volume_surge_multiplier: Volume is a "surge" if it exceeds this
            multiple of the rolling mean.

    Returns:
        A copy of ``odds_df`` augmented with the five feature columns
        described above.

    Raises:
        ValueError: If required columns are absent.
    """
    required = {"game_id", "elapsed_seconds", "book_implied_prob", "volume"}
    missing = required - set(odds_df.columns)
    if missing:
        raise ValueError(f"odds_df is missing required columns: {missing}")

    df = odds_df.copy().sort_values(["game_id", "elapsed_seconds"]).reset_index(drop=True)

    # Pre-allocate output arrays
    odds_return = np.full(len(df), np.nan)
    rolling_vol = np.full(len(df), np.nan)
    momentum = np.full(len(df), np.nan)
    volume_surge = np.zeros(len(df), dtype=bool)
    price_impact = np.full(len(df), np.nan)

    for gid, grp in df.groupby("game_id", sort=False):
        idx = grp.index.to_numpy()
        prob = grp["book_implied_prob"].to_numpy(dtype=float)
        vol = grp["volume"].to_numpy(dtype=float)

        n = len(idx)
        if n < 2:
            continue

        # Log returns (undefined for the first observation → NaN)
        log_prob = np.log(np.clip(prob, 1e-9, 1.0))
        ret = np.diff(log_prob, prepend=np.nan)
        ret[0] = np.nan
        odds_return[idx] = ret

        # Rolling volatility
        ret_series = pd.Series(ret)
        roll_vol = ret_series.rolling(window=rolling_vol_window, min_periods=2).std()
        rolling_vol[idx] = roll_vol.to_numpy()

        # Momentum (sum of returns over window)
        roll_mom = ret_series.rolling(window=momentum_window, min_periods=1).sum()
        momentum[idx] = roll_mom.to_numpy()

        # Volume surge
        vol_series = pd.Series(vol)
        roll_mean_vol = vol_series.rolling(window=volume_surge_window, min_periods=1).mean()
        surge = vol > (volume_surge_multiplier * roll_mean_vol.to_numpy())
        volume_surge[idx] = surge

        # Price impact: return / log(volume + 1)
        log_vol = np.log(vol + 1.0)
        pi = np.where(log_vol > 0, ret / log_vol, np.nan)
        price_impact[idx] = pi

    df["odds_return"] = odds_return
    df["rolling_odds_vol_10"] = rolling_vol
    df["odds_momentum_5"] = momentum
    df["volume_surge"] = volume_surge
    df["price_impact"] = price_impact

    logger.info(
        "Odds-path features built: %d rows, surge rate=%.2f%%",
        len(df),
        100.0 * volume_surge.mean(),
    )
    return df
