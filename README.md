# In-Play Line-Movement Microstructure Model

A production-grade Python framework that models **live NFL odds movements as an order-book microstructure problem**, extracting latent fair prices from noisy sportsbook quotes and generating risk-managed betting signals.

---

## Motivation & Alpha Thesis

Sportsbooks systematically **under- or over-react** to specific in-game events — turnovers near the end of a half, field goals in the red zone, or a sudden gust of momentum after a big play.  These reactions create predictable, short-horizon mispricings in the live betting market that are analogous to the transient price dislocations exploited by high-frequency traders around order-book imbalances.

This project models the live odds stream as if it were a Level-2 order book:

- **Sportsbook quotes** ↔ best bid/ask quotes from a market maker
- **Win probability** ↔ unobservable mid-price
- **Significant odds moves** ↔ large market orders that temporarily move the mid
- **Edge** ↔ basis between fair mid-price and quoted price

We then extract fair value, detect mispricings, and size positions using mathematically principled methods from quantitative finance:

1. **Hawkes process** — models the *self-exciting* clustering of significant odds moves (bets beget bets, information cascades).
2. **Kalman filter** — extracts the *latent* fair win probability from the noisy book-quoted implied probability, with regime jumps on scoring events.
3. **Monte Carlo simulation** — prices the distributional fair value of the remaining game like an options pricer sampling paths.
4. **Fractional Kelly** — sizes positions to maximise log-growth subject to drawdown constraints, with per-game and portfolio-level exposure caps.

---

## Finance Analog

| Sports Betting Component | Quantitative Finance Analog |
|---|---|
| Hawkes process for odds-move clustering | HFT order-flow self-excitation; market impact modelling |
| Kalman filter for latent WP | Mid-price extraction from bid-ask quotes (Glosten-Milgrom) |
| Inhomogeneous Poisson scoring model | Time-varying intensity models (credit risk, trade arrival) |
| Monte Carlo game simulation | Risk-neutral path simulation for options pricing |
| Fractional Kelly sizing | Kelly-optimal leverage with drawdown constraint (portfolio theory) |
| Walk-forward backtest | Out-of-sample time-series validation; regime-aware backtesting |

---

## Architecture

```
┌─────────────────────┐
│  nflfastR PBP Data  │  (via nfl_data_py)
└────────┬────────────┘
         │
         ▼
┌────────────────────────────┐
│   Feature Engineering      │
│  game_state_features.py    │  down/distance, field position,
│  odds_path_features.py     │  rolling EPA, vol, momentum
└────────────┬───────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
┌──────────┐   ┌───────────────────┐
│ Poisson  │   │  Hawkes Process   │
│ Scoring  │   │  (odds-change     │
│ Intensity│   │   self-excitation)│
└────┬─────┘   └────────┬──────────┘
     │                  │
     └───────┬───────────┘
             ▼
    ┌────────────────┐
    │  Kalman Filter │   latent true WP from noisy book implied prob
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │ Monte Carlo    │   simulate remaining game → fair odds distribution
    │  Simulation    │
    └───────┬────────┘
            │
            ▼
    ┌────────────────────┐
    │  Signal Generation │   edge = model fair prob − book implied prob
    └───────┬────────────┘
            │
            ▼
    ┌────────────────────┐
    │  Kelly Sizing      │   fractional Kelly + exposure caps
    └───────┬────────────┘
            │
            ▼
    ┌────────────────────┐
    │  Walk-Forward      │   historical performance ledger
    │  Backtest Engine   │
    └────────────────────┘
```

---

## Methodology

### 1. Inhomogeneous Poisson Scoring Intensity

The probability of a scoring event on play $i$ is modelled as a logistic regression over game-state features:

$$P(\text{score}_i \mid \mathbf{x}_i) = \sigma(\mathbf{w}^\top \mathbf{x}_i)$$

The continuous-time scoring intensity (events per second) is:

$$\lambda(t) = \frac{P(\text{score} \mid \mathbf{x}(t))}{\bar{\tau}}$$

where $\bar{\tau}$ is the average seconds per play (≈ 40 s).

### 2. Hawkes Process Log-Likelihood

The self-exciting intensity for significant odds moves:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \, e^{-\beta(t - t_i)}$$

Log-likelihood (with closed-form integral):

$$\mathcal{L}(\mu, \alpha, \beta) = \sum_i \log \lambda(t_i) - \underbrace{\mu T + \frac{\alpha}{\beta} \sum_i \left(1 - e^{-\beta(T - t_i)}\right)}_{\int_0^T \lambda(s)\,ds}$$

Stationarity requires $\rho = \alpha / \beta < 1$.

### 3. Kalman Filter Recursions

State: $x_t = \text{logit}(\text{true\_wp}_t)$

**Prediction:**
$$\hat{x}_{t|t-1} = x_{t-1|t-1}, \qquad P_{t|t-1} = P_{t-1|t-1} + Q \cdot \Delta t$$

**Update** (observation $z_t = \text{logit}(p_{\text{book},t})$):
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$
$$x_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - \hat{x}_{t|t-1})$$
$$P_{t|t} = (1 - K_t)\,P_{t|t-1}$$

On scoring events, $Q$ is temporarily multiplied by a factor of 3 to accommodate the regime shift.

### 4. Kelly Criterion

For a bet with decimal odds $d$ and estimated true probability $p$:

$$f^* = \frac{p \cdot d - 1}{d - 1}$$

Fractional Kelly: $f = k \cdot f^*$ where $k = 0.25$ (quarter-Kelly).

Stake is further constrained by:
- **Per-bet cap**: $f \leq f_{\max} = 0.05$
- **Per-game cap**: $\sum_{\text{same game}} f_i \leq 0.08$
- **Portfolio cap**: $\sum_{\text{all}} f_i \leq 0.20$

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/isaiahchoi/nfl-live-odds-microstructure
cd nfl-live-odds-microstructure
pip install -r requirements.txt
pip install -e .

# 2. Download play-by-play data (requires internet access)
python -m src.data_ingestion.play_by_play --seasons 2022 2023 --cache-dir data/processed

# 3. Run the full end-to-end pipeline (Jupyter)
jupyter notebook notebooks/01_data_exploration.ipynb

# 4. Or run all three notebooks sequentially in the terminal
jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --output notebooks/01_data_exploration_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_model_fitting.ipynb --output notebooks/02_model_fitting_executed.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_backtest_results.ipynb --output notebooks/03_backtest_results_executed.ipynb

# 5. Run the test suite
pytest tests/ -v
```

---

## Repository Structure

```
nfl-live-odds-microstructure/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── default.yaml              # All hyperparameters and file paths
├── data/
│   ├── raw/                      # gitignored — downloaded data
│   └── processed/                # gitignored — cleaned parquet files
├── src/
│   ├── data_ingestion/
│   │   ├── play_by_play.py       # nfl_data_py wrapper + cleaning
│   │   └── odds_simulator.py     # Synthetic in-play odds generator
│   ├── feature_engineering/
│   │   ├── game_state_features.py
│   │   └── odds_path_features.py
│   ├── models/
│   │   ├── poisson_scoring.py    # Inhomogeneous Poisson model
│   │   ├── hawkes_process.py     # Hawkes process MLE + simulation
│   │   └── kalman_filter.py      # Kalman filter for latent WP
│   ├── simulation/
│   │   └── monte_carlo.py        # MC game simulation
│   ├── strategy/
│   │   ├── signal_generation.py  # Edge computation + signals
│   │   └── kelly_sizing.py       # Fractional Kelly + caps
│   ├── backtesting/
│   │   └── backtest_engine.py    # Walk-forward backtest
│   └── visualization/
│       └── plots.py              # Matplotlib diagnostics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_fitting.ipynb
│   └── 03_backtest_results.ipynb
└── tests/
    ├── test_poisson.py
    ├── test_hawkes.py
    ├── test_kalman.py
    ├── test_kelly.py
    └── test_backtest.py
```

---

## Configuration

All hyperparameters live in `config/default.yaml` and are documented with inline comments.  Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `odds_simulator.sigma_micro` | 0.02 | Microstructure noise std dev |
| `hawkes.significant_move_threshold` | 0.01 | Min \|odds_return\| for event |
| `kalman.Q` | 0.0001 | Process noise variance |
| `kalman.R` | 0.01 | Observation noise variance |
| `signals.edge_threshold` | 0.03 | Min edge to generate signal |
| `kelly.fraction` | 0.25 | Fractional Kelly multiplier |
| `backtest.initial_bankroll` | 10000 | Starting capital |

---
