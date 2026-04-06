"""
Game-state feature engineering from raw play-by-play data.

Transforms the raw PBP columns into categorical and rolling-window features
that capture the *microstructure context* of each play: field position,
scoring urgency, momentum, and drive progression.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Short-yardage threshold (yards)
_SHORT_DISTANCE = 4


def _down_distance_bucket(df: pd.DataFrame) -> pd.Series:
    """Categorise each play by down and distance bucket.

    Args:
        df: PBP DataFrame with ``down`` and ``ydstogo`` columns.

    Returns:
        Categorical ``pd.Series`` with labels such as
        ``"1st-and-10"``, ``"2nd-short"``, etc.
    """
    result = pd.Series("other", index=df.index, dtype="object")
    if "down" not in df.columns or "ydstogo" not in df.columns:
        return result

    down = df["down"]
    dist = df["ydstogo"]

    result[down == 1] = "1st-and-10"
    result[(down == 2) & (dist <= _SHORT_DISTANCE)] = "2nd-short"
    result[(down == 2) & (dist > _SHORT_DISTANCE)] = "2nd-long"
    result[(down == 3) & (dist <= _SHORT_DISTANCE)] = "3rd-short"
    result[(down == 3) & (dist > _SHORT_DISTANCE)] = "3rd-long"
    result[(down == 4) & (dist <= _SHORT_DISTANCE)] = "4th-short"
    result[(down == 4) & (dist > _SHORT_DISTANCE)] = "4th-long"

    return result.astype("category")


def _field_position_bucket(df: pd.DataFrame) -> pd.Series:
    """Categorise each play by field position.

    Args:
        df: PBP DataFrame with ``yardline_100`` column
            (yards to opponent end zone, 1–99).

    Returns:
        Categorical ``pd.Series`` with one of:
        ``"own_territory"``, ``"midfield"``,
        ``"opponent_territory"``, ``"red_zone"``.
    """
    result = pd.Series("unknown", index=df.index, dtype="object")
    if "yardline_100" not in df.columns:
        return result

    yl = df["yardline_100"]
    result[yl > 50] = "own_territory"
    result[(yl > 40) & (yl <= 50)] = "midfield"
    result[(yl > 20) & (yl <= 40)] = "opponent_territory"
    result[yl <= 20] = "red_zone"

    return result.astype("category")


def _score_diff_bucket(df: pd.DataFrame) -> pd.Series:
    """Bin score differential into game-situation categories.

    Args:
        df: PBP DataFrame with ``score_differential`` column.

    Returns:
        Categorical ``pd.Series`` with string labels.
    """
    result = pd.Series("unknown", index=df.index, dtype="object")
    if "score_differential" not in df.columns:
        return result

    sd = df["score_differential"]
    # 7 intervals, 8 bin edges → 7 labels
    bins = [-99, -14, -7, -3, 3, 7, 14, 99]
    labels = [
        "losing_big",    # <= -14
        "losing_td",     # -13 to -7
        "losing_fg",     # -6 to -3
        "near_tied",     # -2 to +2
        "winning_fg",    # 3 to 6
        "winning_td",    # 7 to 13
        "winning_big",   # >= 14
    ]
    # pd.cut uses (lo, hi] intervals; include_lowest makes the first bin closed on both sides
    bucketed = pd.cut(sd, bins=bins, labels=labels, right=True, include_lowest=True)
    result = bucketed.astype("object").fillna("unknown")
    return result.astype("category")


def _quarter_from_game_seconds(df: pd.DataFrame) -> pd.Series:
    """Derive the game quarter (1–5) from ``game_seconds_remaining``.

    Args:
        df: PBP DataFrame with ``game_seconds_remaining`` column.

    Returns:
        Integer ``pd.Series`` in {1, 2, 3, 4, 5} where 5 = overtime.
    """
    if "game_seconds_remaining" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    gsr = df["game_seconds_remaining"]
    q = pd.Series(5, index=df.index, dtype=int)  # default: OT
    q[gsr > 2700] = 1
    q[(gsr <= 2700) & (gsr > 1800)] = 2
    q[(gsr <= 1800) & (gsr > 900)] = 3
    q[(gsr <= 900) & (gsr > 0)] = 4
    q[gsr <= 0] = 5
    return q


def _is_two_minute_warning(df: pd.DataFrame, quarter: pd.Series) -> pd.Series:
    """Flag plays occurring in the two-minute warning window.

    Args:
        df: PBP DataFrame with ``quarter_seconds_remaining``.
        quarter: Integer series with quarter numbers.

    Returns:
        Boolean ``pd.Series``.
    """
    if "quarter_seconds_remaining" not in df.columns:
        return pd.Series(False, index=df.index)

    qsr = df["quarter_seconds_remaining"]
    return ((qsr <= 120) & (quarter.isin([2, 4]))).astype(bool)


def _rolling_epa_5(df: pd.DataFrame) -> pd.Series:
    """Compute rolling 5-play mean EPA for the possession team, per game.

    Args:
        df: PBP DataFrame with ``game_id``, ``posteam``, and ``epa`` columns.

    Returns:
        Float ``pd.Series`` of rolling EPA (NaN where fewer than 1 play).
    """
    if "epa" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    out = pd.Series(np.nan, index=df.index)
    for (gid, team), grp in df.groupby(["game_id", "posteam"], observed=True):
        roll = grp["epa"].rolling(window=5, min_periods=1).mean()
        out.loc[grp.index] = roll.values
    return out


def _rolling_success_rate_10(df: pd.DataFrame) -> pd.Series:
    """Compute rolling 10-play success rate (EPA > 0) for the possession team.

    Args:
        df: PBP DataFrame with ``game_id``, ``posteam``, and ``epa`` columns.

    Returns:
        Float ``pd.Series`` representing the rolling success rate.
    """
    if "epa" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    success = (df["epa"] > 0).astype(float)
    out = pd.Series(np.nan, index=df.index)
    for (gid, team), grp in df.groupby(["game_id", "posteam"], observed=True):
        roll = success.loc[grp.index].rolling(window=10, min_periods=1).mean()
        out.loc[grp.index] = roll.values
    return out


def _drive_play_count(df: pd.DataFrame) -> pd.Series:
    """Count how many plays have occurred so far in the current drive.

    Args:
        df: PBP DataFrame with ``game_id`` and ``drive`` columns.

    Returns:
        Integer ``pd.Series``.
    """
    if "drive" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    out = pd.Series(0, index=df.index, dtype=int)
    for (gid, drv), grp in df.groupby(["game_id", "drive"], observed=True):
        cumcount = pd.Series(range(len(grp)), index=grp.index)
        out.loc[grp.index] = cumcount.values
    return out


def build_game_state_features(pbp: pd.DataFrame) -> pd.DataFrame:
    """Augment a PBP DataFrame with game-state microstructure features.

    Adds the following columns to ``pbp``:

    - ``down_distance_bucket`` (category)
    - ``field_position_bucket`` (category)
    - ``score_diff_bucket`` (category)
    - ``quarter`` (int, 1–5)
    - ``is_two_minute_warning`` (bool)
    - ``rolling_epa_5`` (float)
    - ``rolling_success_rate_10`` (float)
    - ``drive_play_count`` (int)

    Args:
        pbp: Cleaned play-by-play DataFrame as returned by
            :func:`src.data_ingestion.play_by_play.load_pbp`.

    Returns:
        A new DataFrame containing all original columns plus the eight
        engineered feature columns listed above.

    Raises:
        ValueError: If ``pbp`` is empty.
    """
    if len(pbp) == 0:
        raise ValueError("Input PBP DataFrame is empty.")

    df = pbp.copy()

    logger.info("Building game-state features for %d plays …", len(df))

    df["down_distance_bucket"] = _down_distance_bucket(df)
    df["field_position_bucket"] = _field_position_bucket(df)
    df["score_diff_bucket"] = _score_diff_bucket(df)

    quarter = _quarter_from_game_seconds(df)
    df["quarter"] = quarter

    df["is_two_minute_warning"] = _is_two_minute_warning(df, quarter)
    df["rolling_epa_5"] = _rolling_epa_5(df)
    df["rolling_success_rate_10"] = _rolling_success_rate_10(df)
    df["drive_play_count"] = _drive_play_count(df)

    logger.info("Game-state features built.")
    return df
