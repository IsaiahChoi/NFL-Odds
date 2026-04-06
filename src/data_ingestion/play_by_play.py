"""
Play-by-play data ingestion from nflfastR via nfl_data_py.

Downloads raw PBP data for the requested seasons, filters to
the columns relevant for this project, and returns a clean DataFrame.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we actually need — anything else is dropped to save memory.
_KEEP_COLS: list[str] = [
    "play_id",
    "game_id",
    "posteam",
    "defteam",
    "quarter_seconds_remaining",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "ep",
    "epa",
    "wp",
    "wpa",
    "play_type",
    "touchdown",
    "fumble_lost",            # standardised in nflfastR
    "interception",
    "posteam_score",
    "defteam_score",
    "drive",
    "series_success",
    "desc",
    # field-goal detection
    "field_goal_result",
    # home/away context
    "home_team",
    "away_team",
    "season",
    "week",
]

# Some column names differ slightly across nfl_data_py versions — list aliases.
_COL_ALIASES: dict[str, str] = {
    "fumble": "fumble_lost",   # older alias
}


def load_pbp(seasons: list[int], cache_dir: Optional[str] = None) -> pd.DataFrame:
    """Download and clean NFL play-by-play data for the requested seasons.

    Uses ``nfl_data_py.import_pbp_data`` to fetch the raw data, then
    selects only the columns needed by downstream pipeline modules.

    Args:
        seasons: List of NFL seasons (years) to fetch, e.g. ``[2022, 2023]``.
        cache_dir: If provided, save a parquet copy here and load from cache
            on subsequent calls.  Defaults to ``None`` (no caching).

    Returns:
        A cleaned ``pd.DataFrame`` sorted by ``game_id`` and
        ``elapsed_seconds``, with at least the columns listed in
        ``_KEEP_COLS`` plus the derived ``elapsed_seconds`` column.

    Raises:
        ImportError: If ``nfl_data_py`` is not installed.
        ValueError: If ``seasons`` is empty.
    """
    if not seasons:
        raise ValueError("seasons must be a non-empty list of integers.")

    # -------------------------------------------------------------------------
    # Cache hit?
    # -------------------------------------------------------------------------
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"pbp_{'_'.join(str(s) for s in seasons)}.parquet"
        if cache_path.exists():
            logger.info("Loading PBP data from cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    # -------------------------------------------------------------------------
    # Download
    # -------------------------------------------------------------------------
    try:
        import nfl_data_py as nfl  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "nfl_data_py is not installed.  Run: pip install nfl_data_py"
        ) from exc

    logger.info("Downloading PBP data for seasons: %s", seasons)
    raw: pd.DataFrame = nfl.import_pbp_data(seasons)
    logger.info("Downloaded %d rows, %d columns.", len(raw), raw.shape[1])

    # -------------------------------------------------------------------------
    # Normalise column aliases
    # -------------------------------------------------------------------------
    raw = raw.rename(columns=_COL_ALIASES)

    # -------------------------------------------------------------------------
    # Select columns (skip any that are genuinely absent in this data version)
    # -------------------------------------------------------------------------
    available = [c for c in _KEEP_COLS if c in raw.columns]
    missing = set(_KEEP_COLS) - set(available)
    if missing:
        logger.warning("Columns absent in downloaded data (will skip): %s", missing)

    df: pd.DataFrame = raw[available].copy()

    # -------------------------------------------------------------------------
    # Derived columns
    # -------------------------------------------------------------------------
    df["elapsed_seconds"] = (3600.0 - df["game_seconds_remaining"]).clip(lower=0)

    # -------------------------------------------------------------------------
    # Type coercion & cleaning
    # -------------------------------------------------------------------------
    for col in ["touchdown", "fumble_lost", "interception"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in ["down", "ydstogo", "yardline_100", "drive"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["ep", "epa", "wp", "wpa", "score_differential"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the most critical columns are null (e.g. kickoffs with no WP)
    df = df.dropna(subset=["game_id", "elapsed_seconds"])

    # Filter to proper scrimmage plays only (exclude kick-off, etc.) if play_type present
    if "play_type" in df.columns:
        valid_types = {"pass", "run", "field_goal", "punt", "qb_kneel", "qb_spike", "no_play"}
        df = df[df["play_type"].isin(valid_types) | df["play_type"].isna()]

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------
    df = df.sort_values(["game_id", "elapsed_seconds"]).reset_index(drop=True)

    logger.info("Cleaned PBP: %d rows, %d columns.", len(df), df.shape[1])

    # -------------------------------------------------------------------------
    # Cache write
    # -------------------------------------------------------------------------
    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached PBP data to %s", cache_path)

    return df


def main() -> None:
    """CLI entry-point: download and cache PBP data.

    Example::

        python -m src.data_ingestion.play_by_play --seasons 2022 2023

    Args:
        None — reads from ``sys.argv``.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download nflfastR play-by-play data and cache to parquet."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023],
        help="NFL seasons to download (e.g. 2022 2023).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed",
        help="Directory to save parquet cache files.",
    )
    args = parser.parse_args()

    df = load_pbp(seasons=args.seasons, cache_dir=args.cache_dir)
    print(f"Loaded {len(df):,} plays across {df['game_id'].nunique():,} games.")
    print(df.head())


if __name__ == "__main__":
    main()
