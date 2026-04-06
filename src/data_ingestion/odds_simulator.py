"""
Synthetic in-play odds generator.

Because free historical in-play NFL odds at second-level resolution do not
exist publicly, this module builds a *realistic* synthetic odds stream from
nflfastR win-probability (WP) estimates.  The generator adds:

  * **Microstructure noise** — bid-ask bounce, rounding, and model error.
  * **Momentum drift** — sportsbooks systematically over- or under-react
    to the most recent WP shift.  The direction and magnitude of this
    drift is fixed per game (simulating a consistent book behavioural bias)
    but the size is drawn from a Uniform distribution across games.
  * **Volume spikes** — significant events (TDs, turnovers) cause a large
    jump in betting volume.

The resulting data is deliberately imperfect: the book price is
*systematically* biased in a way the Kalman filter and Monte Carlo
simulator are designed to detect.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticOddsGenerator:
    """Generate realistic synthetic in-play odds from a game's PBP DataFrame.

    The generator converts nflfastR's ``wp`` (win-probability for the
    possession team) into a *book implied probability* by adding:

    .. math::

        p_{\\text{book},t} = p_{\\text{true},t}
            + \\varepsilon_t + \\gamma \\cdot (p_{\\text{true},t} - p_{\\text{true},t-1})

    where :math:`\\varepsilon_t \\sim \\mathcal{N}(0, \\sigma_{\\text{micro}})` and
    :math:`\\gamma \\sim \\text{Uniform}(\\text{momentum_min}, \\text{momentum_max})`.

    Attributes:
        sigma_micro: Standard deviation of the Gaussian microstructure noise.
        momentum_factor_min: Lower bound of the per-game momentum multiplier.
        momentum_factor_max: Upper bound of the per-game momentum multiplier.
        base_volume_lambda: Poisson rate for normal-play volume.
        spike_volume_lambda: Poisson rate for scoring/turnover-play volume.
        min_decimal_odds: Floor for generated decimal odds.
        max_decimal_odds: Ceiling for generated decimal odds.
        rng: NumPy random generator instance.
    """

    def __init__(
        self,
        sigma_micro: float = 0.02,
        momentum_factor_min: float = 0.5,
        momentum_factor_max: float = 1.5,
        base_volume_lambda: float = 100.0,
        spike_volume_lambda: float = 500.0,
        min_decimal_odds: float = 1.01,
        max_decimal_odds: float = 50.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the generator with configurable noise parameters.

        Args:
            sigma_micro: Std dev of microstructure noise (probability scale).
            momentum_factor_min: Min of Uniform range for momentum factor.
            momentum_factor_max: Max of Uniform range for momentum factor.
            base_volume_lambda: Poisson λ for typical-play volume.
            spike_volume_lambda: Poisson λ for high-importance-play volume.
            min_decimal_odds: Minimum clamp for decimal odds output.
            max_decimal_odds: Maximum clamp for decimal odds output.
            seed: Random seed for reproducibility.  ``None`` → unseeded.
        """
        self.sigma_micro = sigma_micro
        self.momentum_factor_min = momentum_factor_min
        self.momentum_factor_max = momentum_factor_max
        self.base_volume_lambda = base_volume_lambda
        self.spike_volume_lambda = spike_volume_lambda
        self.min_decimal_odds = min_decimal_odds
        self.max_decimal_odds = max_decimal_odds
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, game_pbp: pd.DataFrame) -> pd.DataFrame:
        """Simulate a synthetic odds stream for a single game.

        Args:
            game_pbp: Play-by-play DataFrame for **one** game.  Must contain
                at least: ``game_id``, ``elapsed_seconds``, ``wp``,
                ``touchdown``, ``fumble_lost`` / ``interception``.

        Returns:
            A DataFrame indexed to the same plays as ``game_pbp`` with columns:

            - ``game_id``
            - ``elapsed_seconds``
            - ``true_wp`` — the nflfastR home-team WP (used as ground truth)
            - ``book_implied_prob`` — noisy book probability (home team wins)
            - ``book_decimal_odds`` — decimal odds for home team
            - ``volume`` — synthetic betting volume on that play

        Raises:
            ValueError: If ``game_pbp`` is empty or missing required columns.
        """
        required = {"game_id", "elapsed_seconds", "wp"}
        missing = required - set(game_pbp.columns)
        if missing:
            raise ValueError(f"game_pbp is missing columns: {missing}")
        if len(game_pbp) == 0:
            raise ValueError("game_pbp is empty.")

        df = game_pbp.copy().reset_index(drop=True)

        # nflfastR wp is for the *possession team*; we treat it as home-team WP
        # for simplicity (most plays the home team has the ball about half the
        # time anyway, and the Kalman filter will correct for systematic bias).
        true_wp = self._get_true_wp(df)

        # Per-game momentum factor (fixed across all plays in this game)
        momentum_factor = float(
            self.rng.uniform(self.momentum_factor_min, self.momentum_factor_max)
        )

        book_prob = self._apply_noise(true_wp, momentum_factor)
        book_odds = self._prob_to_decimal_odds(book_prob)
        volume = self._generate_volume(df)

        result = pd.DataFrame(
            {
                "game_id": df["game_id"],
                "elapsed_seconds": df["elapsed_seconds"],
                "true_wp": true_wp,
                "book_implied_prob": book_prob,
                "book_decimal_odds": book_odds,
                "volume": volume,
            }
        )
        return result

    def generate_multiple_games(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Apply :meth:`generate` to every game in ``pbp``.

        Args:
            pbp: Full play-by-play DataFrame covering multiple games.

        Returns:
            Concatenated odds DataFrame for all games, sorted by
            ``(game_id, elapsed_seconds)``.
        """
        frames: list[pd.DataFrame] = []
        for gid, gdf in pbp.groupby("game_id"):
            try:
                frames.append(self.generate(gdf))
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping game %s due to error: %s", gid, exc)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["game_id", "elapsed_seconds"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_true_wp(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and impute the true win probability array.

        Args:
            df: Single-game PBP frame with a ``wp`` column.

        Returns:
            A float array of length ``len(df)`` in [0.01, 0.99].
        """
        wp = df["wp"].to_numpy(dtype=float)
        # Forward-fill NaNs (missing WP between plays)
        mask = np.isnan(wp)
        if mask.all():
            wp = np.full(len(wp), 0.5)
        else:
            # Simple forward-fill
            last_valid = wp[~mask][0]
            for i in range(len(wp)):
                if np.isnan(wp[i]):
                    wp[i] = last_valid
                else:
                    last_valid = wp[i]
        # Clamp to avoid logit singularities downstream
        return np.clip(wp, 0.01, 0.99)

    def _apply_noise(self, true_wp: np.ndarray, momentum_factor: float) -> np.ndarray:
        """Add microstructure noise and momentum drift to the true WP.

        Args:
            true_wp: Array of true win probabilities.
            momentum_factor: Per-game momentum multiplier gamma.

        Returns:
            Array of book implied probabilities, clipped to [0.01, 0.99].
        """
        n = len(true_wp)

        # Microstructure noise
        noise = self.rng.normal(0.0, self.sigma_micro, size=n)

        # Momentum drift: gamma * delta_wp
        delta_wp = np.diff(true_wp, prepend=true_wp[0])
        drift = momentum_factor * delta_wp

        book_prob = true_wp + noise + drift
        return np.clip(book_prob, 0.01, 0.99)

    def _prob_to_decimal_odds(self, prob: np.ndarray) -> np.ndarray:
        """Convert implied probabilities to decimal odds.

        Args:
            prob: Array of probabilities in (0, 1).

        Returns:
            Array of decimal odds, clipped to
            ``[min_decimal_odds, max_decimal_odds]``.
        """
        odds = 1.0 / np.clip(prob, 1e-6, 1.0)
        return np.clip(odds, self.min_decimal_odds, self.max_decimal_odds)

    def _generate_volume(self, df: pd.DataFrame) -> np.ndarray:
        """Simulate betting volume per play using a Poisson model.

        Args:
            df: Single-game PBP frame.  Uses ``touchdown``, ``fumble_lost``,
                and ``interception`` columns to detect high-volume plays.

        Returns:
            Integer array of volume values.
        """
        n = len(df)
        lambdas = np.full(n, self.base_volume_lambda)

        # Identify high-importance plays (scoring events or turnovers)
        is_spike = np.zeros(n, dtype=bool)
        for col in ("touchdown", "fumble_lost", "interception"):
            if col in df.columns:
                is_spike |= df[col].fillna(0).astype(bool).to_numpy()

        lambdas[is_spike] = self.spike_volume_lambda

        return self.rng.poisson(lambdas).astype(int)
