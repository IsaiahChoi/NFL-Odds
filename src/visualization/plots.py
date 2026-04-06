"""
Visualization module for the NFL live odds microstructure pipeline.

Provides publication-quality matplotlib figures for:

- Single-game inspection (WP paths, Hawkes intensity, edge).
- Backtest performance (equity curve, PnL histogram).
- Kelly-fraction sensitivity (equity curve overlays).

All functions return the matplotlib ``Figure`` object so callers can
save, display, or embed as needed.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend safe for scripts/tests
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Default style
# ------------------------------------------------------------------
_STYLE = "seaborn-v0_8-darkgrid"
_DPI = 120


def _apply_style() -> None:
    """Apply the default matplotlib style, falling back gracefully."""
    try:
        plt.style.use(_STYLE)
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            pass  # use matplotlib defaults


# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------
_C_TRUE_WP = "#1f77b4"      # blue
_C_FILTERED_WP = "#ff7f0e"  # orange
_C_BOOK = "#2ca02c"         # green
_C_HAWKES = "#9467bd"       # purple
_C_EDGE = "#d62728"         # red
_C_BET_HOME = "#17becf"     # cyan
_C_BET_AWAY = "#bcbd22"     # yellow-green
_C_SCORING = "#e377c2"      # pink


# ==============================================================================
# Single-game figure
# ==============================================================================

def plot_single_game(
    game_id: str,
    filtered_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    hawkes_times: Optional[np.ndarray] = None,
    hawkes_intensity: Optional[np.ndarray] = None,
    scoring_times: Optional[np.ndarray] = None,
    figsize: tuple[float, float] = (14, 12),
) -> plt.Figure:
    """Create a three-panel diagnostic figure for a single game.

    Panel 1 — Win probability paths:
        True WP (nflfastR), Kalman-filtered WP, and book implied
        probability plotted against elapsed game seconds.  Vertical
        dashed lines mark scoring events.

    Panel 2 — Hawkes intensity:
        Self-exciting intensity λ(t) over time, with event arrival
        markers.

    Panel 3 — Edge and signal activity:
        Signed edge (model fair prob − book implied prob) over time,
        shaded regions where a signal is active, and bet-entry markers.

    Args:
        game_id: Game identifier string (used in the figure title).
        filtered_df: Output of :meth:`~src.models.kalman_filter.WinProbabilityFilter.filter`.
            Required columns: ``elapsed_seconds``, ``nflfastr_wp``,
            ``filtered_wp``, ``book_implied_prob``.
        odds_df: Synthetic odds DataFrame with ``elapsed_seconds``,
            ``book_implied_prob``, ``volume``.
        signals_df: Output of :func:`~src.strategy.signal_generation.generate_signals`.
            Required columns: ``elapsed_seconds``, ``edge``, ``signal``.
        hawkes_times: Optional sorted array of Hawkes event times (seconds).
        hawkes_intensity: Optional array of λ values at times corresponding
            to ``hawkes_times`` (or a dense grid).
        scoring_times: Optional array of elapsed seconds at which scoring
            events occurred.
        figsize: Figure dimensions (width, height) in inches.

    Returns:
        Matplotlib ``Figure`` object.
    """
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=_DPI, sharex=True)
    fig.suptitle(f"Game: {game_id}", fontsize=14, fontweight="bold", y=1.01)

    t = filtered_df["elapsed_seconds"].to_numpy()

    # ------------------------------------------------------------------
    # Panel 1: Win probability paths
    # ------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(t, filtered_df["nflfastr_wp"], color=_C_TRUE_WP, lw=1.5, label="nflfastR WP (true)")
    ax1.plot(t, filtered_df["filtered_wp"], color=_C_FILTERED_WP, lw=2.0,
             label="Kalman Filtered WP", zorder=3)
    ax1.plot(t, filtered_df["book_implied_prob"], color=_C_BOOK, lw=1.0,
             alpha=0.7, ls="--", label="Book Implied Prob")

    if scoring_times is not None:
        for st in scoring_times:
            ax1.axvline(st, color=_C_SCORING, lw=0.8, ls="--", alpha=0.6)

    ax1.set_ylabel("Win Probability")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Win Probability Paths", fontsize=10)

    # ------------------------------------------------------------------
    # Panel 2: Hawkes intensity
    # ------------------------------------------------------------------
    ax2 = axes[1]
    if hawkes_times is not None and hawkes_intensity is not None:
        intensity_t = hawkes_times if len(hawkes_times) == len(hawkes_intensity) else \
            np.linspace(0, t.max(), len(hawkes_intensity))
        ax2.plot(intensity_t, hawkes_intensity, color=_C_HAWKES, lw=1.5, label="Hawkes λ(t)")
        # Event markers
        event_idx = np.searchsorted(intensity_t, hawkes_times)
        event_idx = np.clip(event_idx, 0, len(hawkes_intensity) - 1)
        ax2.scatter(hawkes_times, hawkes_intensity[event_idx], color=_C_HAWKES,
                    s=20, zorder=5, label="Events")
    else:
        # Fallback: plot volume from odds_df as a proxy
        if "volume" in odds_df.columns:
            ax2.bar(odds_df["elapsed_seconds"], odds_df["volume"],
                    width=30, color=_C_HAWKES, alpha=0.5, label="Volume (proxy)")

    ax2.set_ylabel("Hawkes Intensity / Volume")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_title("Bet-Arrival Self-Excitation (Hawkes)", fontsize=10)

    # ------------------------------------------------------------------
    # Panel 3: Edge and signals
    # ------------------------------------------------------------------
    ax3 = axes[2]
    sig_t = signals_df["elapsed_seconds"].to_numpy()
    edge = signals_df["edge"].to_numpy()
    signal = signals_df["signal"].to_numpy()

    ax3.plot(sig_t, edge, color=_C_EDGE, lw=1.5, label="Edge (model − book)")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.axhline(0.03, color="grey", lw=0.5, ls=":")
    ax3.axhline(-0.03, color="grey", lw=0.5, ls=":")

    # Shaded signal regions
    home_mask = signal == "bet_home"
    away_mask = signal == "bet_away"
    if home_mask.any():
        ax3.fill_between(sig_t, edge, 0, where=home_mask, alpha=0.3, color=_C_BET_HOME,
                         label="Home bet zone")
    if away_mask.any():
        ax3.fill_between(sig_t, edge, 0, where=away_mask, alpha=0.3, color=_C_BET_AWAY,
                         label="Away bet zone")

    # Entry markers
    for label_sig, col in [("bet_home", _C_BET_HOME), ("bet_away", _C_BET_AWAY)]:
        mask = signal == label_sig
        ax3.scatter(sig_t[mask], edge[mask], color=col, s=40, zorder=5, marker="^")

    ax3.set_xlabel("Elapsed Game Time (seconds)")
    ax3.set_ylabel("Edge")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_title("Model Edge and Signal Activity", fontsize=10)

    fig.tight_layout()
    return fig


# ==============================================================================
# Backtest performance
# ==============================================================================

def plot_backtest_results(
    ledger: pd.DataFrame,
    initial_bankroll: float = 10_000.0,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Two-panel backtest performance figure.

    Panel 1 — Equity curve: cumulative bankroll over bet number.
    Panel 2 — PnL distribution: histogram of per-bet PnL with a vertical
    line at the mean.

    Args:
        ledger: Output of :meth:`~src.backtesting.backtest_engine.WalkForwardBacktest.run`.
        initial_bankroll: Starting bankroll for reference line in Panel 1.
        figsize: Figure dimensions.

    Returns:
        Matplotlib ``Figure`` object.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=_DPI)

    # ------------------------------------------------------------------
    # Panel 1: Equity curve
    # ------------------------------------------------------------------
    ax1 = axes[0]
    curve = np.concatenate([[initial_bankroll], ledger["cumulative_bankroll"].to_numpy()])
    ax1.plot(range(len(curve)), curve, color=_C_TRUE_WP, lw=2)
    ax1.axhline(initial_bankroll, color="grey", ls="--", lw=1, label="Initial bankroll")
    ax1.set_xlabel("Bet Number")
    ax1.set_ylabel("Bankroll ($)")
    ax1.set_title("Equity Curve", fontsize=11)
    ax1.legend(fontsize=9)

    final = float(curve[-1])
    roi = (final - initial_bankroll) / initial_bankroll * 100
    ax1.annotate(
        f"Final: ${final:,.0f}\nROI: {roi:+.1f}%",
        xy=(len(curve) - 1, final),
        xytext=(-60, -30),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "grey"},
    )

    # ------------------------------------------------------------------
    # Panel 2: PnL histogram
    # ------------------------------------------------------------------
    ax2 = axes[1]
    pnl = ledger["pnl"].to_numpy()
    mean_pnl = float(pnl.mean())

    ax2.hist(pnl, bins=30, color=_C_EDGE, alpha=0.7, edgecolor="white")
    ax2.axvline(mean_pnl, color="black", lw=2, ls="--",
                label=f"Mean PnL: ${mean_pnl:+.2f}")
    ax2.axvline(0, color="grey", lw=1)
    ax2.set_xlabel("Per-Bet PnL ($)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Per-Bet PnL Distribution", fontsize=11)
    ax2.legend(fontsize=9)

    n_wins = int((pnl > 0).sum())
    n_bets = len(pnl)
    ax2.set_title(
        f"Per-Bet PnL Distribution\n(Win rate: {n_wins}/{n_bets} = {n_wins/max(n_bets,1):.1%})",
        fontsize=10,
    )

    fig.suptitle("Backtest Performance Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ==============================================================================
# Kelly fraction comparison
# ==============================================================================

def plot_kelly_comparison(
    ledger_full: pd.DataFrame,
    ledger_half: pd.DataFrame,
    ledger_quarter: pd.DataFrame,
    initial_bankroll: float = 10_000.0,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Overlay equity curves for full, half, and quarter Kelly strategies.

    Args:
        ledger_full: Backtest ledger using full Kelly (fraction=1.0).
        ledger_half: Backtest ledger using half Kelly (fraction=0.5).
        ledger_quarter: Backtest ledger using quarter Kelly (fraction=0.25).
        initial_bankroll: Starting bankroll for the initial reference point.
        figsize: Figure dimensions.

    Returns:
        Matplotlib ``Figure`` object.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize, dpi=_DPI)

    def _curve(ledger: pd.DataFrame) -> np.ndarray:
        return np.concatenate(
            [[initial_bankroll], ledger["cumulative_bankroll"].to_numpy()]
        )

    configs = [
        (ledger_full, "Full Kelly (f=1.0)", "#d62728", 1.5),
        (ledger_half, "Half Kelly (f=0.5)", "#ff7f0e", 1.5),
        (ledger_quarter, "Quarter Kelly (f=0.25)", "#1f77b4", 2.0),
    ]

    for ledger, label, color, lw in configs:
        if len(ledger) == 0:
            continue
        c = _curve(ledger)
        ax.plot(range(len(c)), c, color=color, lw=lw, label=label)

    ax.axhline(initial_bankroll, color="grey", ls="--", lw=1, label="Initial bankroll")
    ax.set_xlabel("Bet Number")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Kelly Fraction Sensitivity — Equity Curves", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


# ==============================================================================
# Utility
# ==============================================================================

def save_figure(fig: plt.Figure, path: str) -> None:
    """Save a matplotlib figure to disk.

    Args:
        fig: Figure to save.
        path: Output file path (extension determines format, e.g. ``.png``, ``.pdf``).

    Returns:
        None
    """
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    logger.info("Figure saved to %s", path)
    plt.close(fig)
