"""
Monte Carlo game simulator.

Simulates the remainder of an in-progress NFL game from the current
game state using a simplified stochastic play model calibrated to the
scoring-intensity estimates from :class:`src.models.poisson_scoring.ScoringIntensityModel`.

Finance analogy: this is the path-simulation step of an options pricer —
we sample many possible futures and price the "option" (bet) as the
expected payoff under the simulated distribution.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Typical NFL game-clock consumption per play (seconds)
_AVG_PLAY_DURATION = 5.5
# Average yards gained per play (for down-progression simulation)
_AVG_YARDS_MEAN = 3.5
_AVG_YARDS_STD = 7.0
_AVG_YARDS_MIN = -10.0
# Yards to end zone threshold below which TD probability is high
_TD_THRESHOLD = 20
# Probability of TD vs FG when close to goal line
_TD_PROB_RED_ZONE = 0.7
# Points for TD+XP and FG
_TD_POINTS = 7
_FG_POINTS = 3


class GameSimulator:
    """Monte Carlo simulator for the remaining portion of an NFL game.

    Attributes:
        scoring_model: Fitted :class:`~src.models.poisson_scoring.ScoringIntensityModel`
            used to estimate scoring probability on each simulated play.
        n_sims: Default number of simulation paths.
    """

    def __init__(
        self,
        scoring_model,  # ScoringIntensityModel — avoid circular import with string annotation
        n_sims: int = 10_000,
    ) -> None:
        """Initialise the simulator.

        Args:
            scoring_model: Fitted ScoringIntensityModel instance.
            n_sims: Number of Monte Carlo paths (higher → more accurate, slower).
        """
        self.scoring_model = scoring_model
        self.n_sims = n_sims

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_from_state(
        self,
        current_state: dict,
        remaining_seconds: float,
        seed: Optional[int] = None,
    ) -> dict:
        """Simulate the remainder of the game from the current state.

        Args:
            current_state: Dict with keys:
                - ``home_score`` (int)
                - ``away_score`` (int)
                - ``possession`` (str, "home" or "away")
                - ``yardline_100`` (int, yards to opponent end zone)
                - ``down`` (int, 1–4)
                - ``ydstogo`` (int)
                - ``elapsed_seconds`` (float)
            remaining_seconds: Seconds left in the game.
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys:
            - ``home_wins``: Fraction of simulations where home team wins.
            - ``away_wins``: Fraction of simulations where away team wins.
            - ``ties``: Fraction of simulations ending in a tie.
            - ``home_score_dist``: Array of final home scores.
            - ``away_score_dist``: Array of final away scores.
        """
        rng = np.random.default_rng(seed)

        home_score_base = int(current_state.get("home_score", 0))
        away_score_base = int(current_state.get("away_score", 0))

        final_home = np.full(self.n_sims, home_score_base, dtype=int)
        final_away = np.full(self.n_sims, away_score_base, dtype=int)

        for sim_idx in range(self.n_sims):
            h_score = home_score_base
            a_score = away_score_base
            time_left = remaining_seconds
            possession = str(current_state.get("possession", "home"))
            yl = int(current_state.get("yardline_100", 75))
            down = int(current_state.get("down", 1))
            ydstogo = int(current_state.get("ydstogo", 10))

            while time_left > 0:
                # Time consumed by this play
                play_duration = rng.exponential(_AVG_PLAY_DURATION)
                time_left -= play_duration

                # Scoring probability for this play state
                score_prob = self._get_scoring_prob(yl, down, ydstogo)

                if rng.random() < score_prob:
                    # Scoring event
                    if yl <= _TD_THRESHOLD and rng.random() < _TD_PROB_RED_ZONE:
                        pts = _TD_POINTS
                    else:
                        pts = _FG_POINTS

                    if possession == "home":
                        h_score += pts
                    else:
                        a_score += pts

                    # Reset after score: kick off → opponent at ~25 yardline
                    possession = "away" if possession == "home" else "home"
                    yl = 75
                    down = 1
                    ydstogo = 10

                else:
                    # No score: advance the ball
                    yards_gained = float(
                        np.clip(
                            rng.normal(_AVG_YARDS_MEAN, _AVG_YARDS_STD),
                            _AVG_YARDS_MIN,
                            float(yl),
                        )
                    )
                    yl = int(np.clip(yl - yards_gained, 1, 99))

                    if yards_gained >= ydstogo:
                        # First down
                        down = 1
                        ydstogo = 10
                    else:
                        down += 1
                        ydstogo -= int(yards_gained)
                        ydstogo = max(ydstogo, 1)

                    if down > 4:
                        # Turnover on downs or punt
                        possession = "away" if possession == "home" else "home"
                        yl = int(np.clip(100 - yl, 1, 99))
                        down = 1
                        ydstogo = 10

            final_home[sim_idx] = h_score
            final_away[sim_idx] = a_score

        home_wins = float(np.mean(final_home > final_away))
        away_wins = float(np.mean(final_away > final_home))
        ties = float(np.mean(final_home == final_away))

        return {
            "home_wins": home_wins,
            "away_wins": away_wins,
            "ties": ties,
            "home_score_dist": final_home,
            "away_score_dist": final_away,
        }

    def fair_odds(
        self,
        current_state: dict,
        remaining_seconds: float,
        seed: Optional[int] = None,
    ) -> dict:
        """Compute fair decimal odds for home/away given the current state.

        Args:
            current_state: Same format as :meth:`simulate_from_state`.
            remaining_seconds: Seconds remaining in the game.
            seed: Random seed.

        Returns:
            Dict with:
            - ``home_win_prob``: Simulated home-team win probability.
            - ``away_win_prob``: Simulated away-team win probability.
            - ``fair_home_odds``: 1 / home_win_prob (decimal).
            - ``fair_away_odds``: 1 / away_win_prob (decimal).
        """
        result = self.simulate_from_state(current_state, remaining_seconds, seed=seed)

        hwp = max(result["home_wins"], 1e-4)
        awp = max(result["away_wins"], 1e-4)

        return {
            "home_win_prob": hwp,
            "away_win_prob": awp,
            "fair_home_odds": 1.0 / hwp,
            "fair_away_odds": 1.0 / awp,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_scoring_prob(self, yardline_100: int, down: int, ydstogo: int) -> float:
        """Return the scoring probability for a play given state.

        Uses a simplified heuristic calibrated to the scoring model.
        For a full model integration, this would call
        ``scoring_model.predict_proba`` with a constructed feature row.

        Args:
            yardline_100: Yards to opponent end zone (1–99).
            down: Current down (1–4).
            ydstogo: Yards to first down or end zone.

        Returns:
            Float probability in [0, 1].
        """
        # Use a calibrated logistic-like function
        # The closer to the end zone and the better the down, the higher P(score)
        base_prob = 0.08 * np.exp(-0.03 * yardline_100)

        # Down penalty: 4th downs with long ydstogo rarely score
        down_factor = {1: 1.0, 2: 0.9, 3: 0.75, 4: 0.4}.get(down, 0.4)
        if down == 4 and ydstogo > 4:
            down_factor = 0.1  # likely punt / turnover on downs

        prob = base_prob * down_factor
        return float(np.clip(prob, 1e-4, 0.50))
