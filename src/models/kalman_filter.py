"""
Kalman filter for latent "true" win-probability extraction.

Works in logit space so the state can range over all reals while
the observable (book implied probability) stays in (0, 1).

Finance analogy: extracting the unobservable mid-price from noisy
bid-ask quotes — here the "mid-price" is the true win probability,
and the "quoted price" is the book's offered odds.

State model (scalar):
    x_t  = x_{t-1} + w_t,          w_t ~ N(0, Q)
    z_t  = x_t + v_t,              v_t ~ N(0, R)

where:
    x_t = logit(true_win_prob_t)
    z_t = logit(book_implied_prob_t)

On scoring/turnover events we inject a discrete jump into the state
and temporarily inflate Q to model the regime shift.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit  # expit = sigmoid

logger = logging.getLogger(__name__)


class WinProbabilityFilter:
    """Kalman filter that extracts latent true win probability.

    The filter operates in logit space for numerical stability.
    Scoring events trigger a deterministic state jump (using the
    nflfastR WP change) and a temporary increase in process noise.

    Attributes:
        Q: Baseline process noise variance (per second).
        R: Observation noise variance.
        x0: Initial state (logit of 0.5 = 0.0 by default).
        P0: Initial state covariance.
        scoring_Q_multiplier: Q inflation factor on scoring plays.
    """

    def __init__(
        self,
        Q: float = 1e-4,
        R: float = 1e-2,
        x0: float = 0.0,
        P0: float = 1.0,
        scoring_Q_multiplier: float = 3.0,
    ) -> None:
        """Initialise filter parameters.

        Args:
            Q: Process noise variance per second.  Small Q → trust prior;
                large Q → trust observations.
            R: Observation noise variance.  Represents how noisy the book's
                quoted probability is relative to the true probability.
            x0: Initial state (logit of prior win probability).
                Defaults to logit(0.5) = 0.
            P0: Initial state covariance (uncertainty about x0).
            scoring_Q_multiplier: Factor by which Q is temporarily multiplied
                immediately after a scoring event / turnover.
        """
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.P0 = P0
        self.scoring_Q_multiplier = scoring_Q_multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Run the Kalman filter forward over one game's observations.

        Args:
            observations: DataFrame for a single game with columns:
                - ``elapsed_seconds`` — observation time (seconds from kick-off).
                - ``book_implied_prob`` — book's quoted probability in (0, 1).
                - ``is_scoring_event`` — boolean/int: True when a TD, FG, or
                  turnover occurred at this play.
                - ``nflfastr_wp`` — nflfastR win probability (used for the
                  scoring-event jump and as the ground-truth reference).

        Returns:
            DataFrame with columns:
            - ``elapsed_seconds``
            - ``filtered_logit_wp``
            - ``filtered_wp`` — sigmoid of filtered_logit_wp
            - ``filtered_variance``
            - ``book_implied_prob``
            - ``nflfastr_wp``

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"elapsed_seconds", "book_implied_prob", "is_scoring_event", "nflfastr_wp"}
        missing = required - set(observations.columns)
        if missing:
            raise ValueError(f"observations missing columns: {missing}")

        obs = observations.sort_values("elapsed_seconds").reset_index(drop=True)
        n = len(obs)

        filtered_logit = np.empty(n)
        filtered_var = np.empty(n)

        # Initial state
        x = float(self.x0)
        P = float(self.P0)
        inflate_Q_next = False  # flag to inflate Q on the next step

        for i in range(n):
            row = obs.iloc[i]
            dt = float(row["elapsed_seconds"]) if i == 0 else \
                float(row["elapsed_seconds"]) - float(obs.iloc[i - 1]["elapsed_seconds"])
            dt = max(dt, 0.0)

            # ----------------------------------------------------------
            # Scoring event: inject WP jump before prediction step
            # ----------------------------------------------------------
            if bool(row["is_scoring_event"]) and i > 0:
                wp_now = float(row["nflfastr_wp"])
                wp_prev = float(obs.iloc[i - 1]["nflfastr_wp"])
                # Guard against NaN or boundary WP values
                wp_now = np.clip(wp_now, 0.01, 0.99)
                wp_prev = np.clip(wp_prev, 0.01, 0.99)
                delta_logit = float(logit(wp_now)) - float(logit(wp_prev))
                x = x + delta_logit
                inflate_Q_next = True

            # ----------------------------------------------------------
            # Prediction step
            # ----------------------------------------------------------
            Q_eff = self.Q * dt * (self.scoring_Q_multiplier if inflate_Q_next else 1.0)
            inflate_Q_next = False  # reset after use

            x_pred = x
            P_pred = P + Q_eff

            # ----------------------------------------------------------
            # Update step (observation available?)
            # ----------------------------------------------------------
            book_prob = float(row["book_implied_prob"])
            book_prob = np.clip(book_prob, 0.01, 0.99)
            z = float(logit(book_prob))

            # Innovation
            innov = z - x_pred

            # Kalman gain
            K = P_pred / (P_pred + self.R)

            # State update
            x = x_pred + K * innov
            P = (1.0 - K) * P_pred

            filtered_logit[i] = x
            filtered_var[i] = P

        result = pd.DataFrame(
            {
                "elapsed_seconds": obs["elapsed_seconds"].to_numpy(),
                "filtered_logit_wp": filtered_logit,
                "filtered_wp": expit(filtered_logit),
                "filtered_variance": filtered_var,
                "book_implied_prob": obs["book_implied_prob"].to_numpy(),
                "nflfastr_wp": obs["nflfastr_wp"].to_numpy(),
            }
        )
        return result

    def filter_game(
        self,
        pbp: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convenience wrapper that merges PBP and odds before filtering.

        Aligns the odds stream with PBP elapsed_seconds, detects scoring
        events, and calls :meth:`filter`.

        Args:
            pbp: Single-game PBP DataFrame (from ``load_pbp``).
            odds: Single-game odds DataFrame (from ``SyntheticOddsGenerator``).

        Returns:
            Filtered DataFrame as returned by :meth:`filter`.
        """
        # Scoring events: TD or made FG or turnover
        pbp_ev = pbp.copy()
        is_scoring = pd.Series(0, index=pbp_ev.index)
        for col in ("touchdown", "fumble_lost", "interception"):
            if col in pbp_ev.columns:
                is_scoring |= pbp_ev[col].fillna(0).astype(int)
        if "field_goal_result" in pbp_ev.columns:
            is_scoring |= (pbp_ev["field_goal_result"] == "made").astype(int)
        pbp_ev["is_scoring_event"] = is_scoring.astype(bool)

        # Merge on elapsed_seconds (use nearest-play match for odds)
        merged = pd.merge_asof(
            odds.sort_values("elapsed_seconds"),
            pbp_ev[["elapsed_seconds", "is_scoring_event", "wp"]].sort_values("elapsed_seconds"),
            on="elapsed_seconds",
            direction="nearest",
        )
        merged = merged.rename(columns={"wp": "nflfastr_wp", "true_wp": "true_wp"})

        if "nflfastr_wp" not in merged.columns:
            merged["nflfastr_wp"] = merged.get("true_wp", 0.5)

        merged["nflfastr_wp"] = merged["nflfastr_wp"].fillna(0.5).clip(0.01, 0.99)
        merged["is_scoring_event"] = merged["is_scoring_event"].fillna(False).astype(bool)

        return self.filter(
            merged[["elapsed_seconds", "book_implied_prob", "is_scoring_event", "nflfastr_wp"]]
        )
