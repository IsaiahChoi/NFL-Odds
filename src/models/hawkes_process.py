"""
Univariate Hawkes process for odds-change arrivals.

Models the self-exciting nature of significant odds moves: a large move at
time t_i increases the probability of another large move shortly afterward
(analogous to order-flow clustering in HFT market microstructure).

The intensity is:

.. math::

    \\lambda(t) = \\mu + \\sum_{t_i < t} \\alpha \\, e^{-\\beta(t - t_i)}

Parameters (μ, α, β) are estimated by maximum log-likelihood on a set of
event times.  The closed-form integral of the intensity over [0, T] makes
the optimisation tractable:

.. math::

    \\int_0^T \\lambda(s)\\,ds
    = \\mu T + \\frac{\\alpha}{\\beta}\\sum_i \\left(1 - e^{-\\beta(T-t_i)}\\right)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class HawkesOddsModel:
    """Univariate Hawkes process fitted to significant odds-change events.

    Attributes:
        threshold: Minimum |odds_return| to qualify as a significant event.
        mu_: Estimated baseline intensity (events/second).
        alpha_: Estimated excitation coefficient.
        beta_: Estimated decay rate.
        is_fitted_: True after :meth:`fit` succeeds.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        mu_init: float = 0.5,
        alpha_init: float = 0.3,
        beta_init: float = 1.0,
        mu_bounds: tuple[float, float] = (1e-4, 10.0),
        alpha_bounds: tuple[float, float] = (1e-4, 5.0),
        beta_bounds: tuple[float, float] = (1e-3, 20.0),
    ) -> None:
        """Initialise the Hawkes model.

        Args:
            threshold: |odds_return| threshold defining a significant event.
            mu_init: Starting value for baseline intensity.
            alpha_init: Starting value for excitation coefficient.
            beta_init: Starting value for decay rate.
            mu_bounds: (lower, upper) optimisation bounds for μ.
            alpha_bounds: (lower, upper) optimisation bounds for α.
            beta_bounds: (lower, upper) optimisation bounds for β.
        """
        self.threshold = threshold
        self._mu_init = mu_init
        self._alpha_init = alpha_init
        self._beta_init = beta_init
        self._mu_bounds = mu_bounds
        self._alpha_bounds = alpha_bounds
        self._beta_bounds = beta_bounds

        self.mu_: float = mu_init
        self.alpha_: float = alpha_init
        self.beta_: float = beta_init
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def branching_ratio(self) -> float:
        """Return α/β — must be < 1 for the process to be stationary.

        Returns:
            Float in (0, ∞).  Values ≥ 1 indicate an explosive process.
        """
        return self.alpha_ / self.beta_

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, event_times: np.ndarray, T: float) -> "HawkesOddsModel":
        """Estimate (μ, α, β) by maximum log-likelihood.

        Args:
            event_times: 1-D array of event arrival times (seconds),
                sorted in ascending order.  These should be the times of
                significant odds moves only.
            T: Total observation window length (seconds).

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If ``event_times`` is empty or ``T <= 0``.
        """
        if len(event_times) == 0:
            raise ValueError("event_times must be non-empty.")
        if T <= 0:
            raise ValueError("T must be positive.")

        event_times = np.sort(event_times)

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, alpha, beta = params
            return -self._log_likelihood(mu, alpha, beta, event_times, T)

        x0 = np.array([self._mu_init, self._alpha_init, self._beta_init])
        bounds = [self._mu_bounds, self._alpha_bounds, self._beta_bounds]

        result = minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            logger.warning("Hawkes MLE did not fully converge: %s", result.message)

        self.mu_, self.alpha_, self.beta_ = result.x
        self.is_fitted_ = True

        logger.info(
            "Hawkes fitted: μ=%.4f, α=%.4f, β=%.4f, n=%d, T=%.1f, ratio=%.3f",
            self.mu_, self.alpha_, self.beta_, len(event_times), T,
            self.branching_ratio,
        )
        return self

    # ------------------------------------------------------------------
    # Intensity / simulation
    # ------------------------------------------------------------------

    def intensity(self, t: float, history: np.ndarray) -> float:
        """Compute λ(t) given the history of past events.

        Args:
            t: Current time (seconds).
            history: Sorted array of past event times (all < ``t``).

        Returns:
            Float intensity λ(t) ≥ μ.
        """
        past = history[history < t]
        excitation = np.sum(
            self.alpha_ * np.exp(-self.beta_ * (t - past))
        )
        return float(self.mu_ + excitation)

    def intensity_path(
        self, times: np.ndarray, event_times: np.ndarray
    ) -> np.ndarray:
        """Compute λ(t) for an array of query times.

        Args:
            times: Array of query times at which to evaluate λ.
            event_times: Sorted array of event arrival times.

        Returns:
            Float array of intensities, same length as ``times``.
        """
        result = np.empty(len(times))
        for i, t in enumerate(times):
            result[i] = self.intensity(t, event_times)
        return result

    def simulate(self, T: float, seed: Optional[int] = None) -> np.ndarray:
        """Simulate a Hawkes process up to time T using Ogata's thinning.

        Args:
            T: Simulation horizon (seconds).
            seed: Random seed for reproducibility.

        Returns:
            Sorted array of simulated event times in [0, T].
        """
        rng = np.random.default_rng(seed)
        events: list[float] = []
        t = 0.0

        while t < T:
            # Upper bound on intensity: current λ at t is the max because
            # self-excitation can only decrease from here (exponential decay).
            lam_bar = self.intensity(t, np.array(events))

            # Draw next candidate event from Poisson(lam_bar)
            if lam_bar <= 0:
                break
            dt = rng.exponential(1.0 / lam_bar)
            t_candidate = t + dt

            if t_candidate > T:
                break

            # Thinning step: accept with probability λ(t_candidate) / λ_bar
            lam_candidate = self.intensity(t_candidate, np.array(events))
            u = rng.uniform(0.0, 1.0)
            if u <= lam_candidate / lam_bar:
                events.append(t_candidate)

            t = t_candidate

        return np.array(events)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_likelihood(
        mu: float,
        alpha: float,
        beta: float,
        event_times: np.ndarray,
        T: float,
    ) -> float:
        """Compute the Hawkes log-likelihood for given parameters.

        Args:
            mu: Baseline intensity.
            alpha: Excitation amplitude.
            beta: Decay rate.
            event_times: Sorted array of event times.
            T: Observation window.

        Returns:
            Float log-likelihood value.
        """
        n = len(event_times)
        if n == 0:
            return -mu * T

        # Recursive computation of the sum of past excitation at each event
        # A[i] = sum_{j<i} alpha * exp(-beta * (t_i - t_j))
        # Updated recursively: A[i] = (A[i-1] + alpha) * exp(-beta*(t_i-t_{i-1}))
        A = np.zeros(n)
        for i in range(1, n):
            A[i] = (A[i - 1] + alpha) * np.exp(-beta * (event_times[i] - event_times[i - 1]))

        # λ(t_i) = μ + A[i]
        lam_events = mu + A
        lam_events = np.maximum(lam_events, 1e-300)  # numerical safety

        # Log-likelihood = sum_i log(λ(t_i)) - integral_0^T λ(s) ds
        log_sum = np.sum(np.log(lam_events))

        # Closed-form integral
        integral = mu * T + (alpha / beta) * np.sum(
            1.0 - np.exp(-beta * (T - event_times))
        )

        return log_sum - integral
