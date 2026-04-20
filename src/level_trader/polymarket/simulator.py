"""Synthetic Polymarket market simulator.

We model a single binary market as:

* A latent "true" probability ``p_true`` drawn from a Beta distribution. This
  is the probability that the market eventually resolves YES.
* An observed price path. At each step we draw

      price_t = clip(p_true + noise_t, eps, 1 - eps)

  where ``noise_t`` follows an Ornstein–Uhlenbeck (AR(1)) process that is
  mean-reverting to 0. The instantaneous noise std decays linearly from
  ``sigma_start`` at t=0 down to ``sigma_end`` at t=T-1 — real prediction
  markets sharpen as resolution approaches.
* A realized outcome drawn from ``Bernoulli(p_true)``. At t=T the price is
  pinned to 0 or 1 depending on the outcome.

Every random draw uses the ``numpy.random.Generator`` passed in, so both the
price path and the outcome are fully deterministic given a seed. This makes
the backtest reproducible, which is exactly what we want when judging whether
a strategy has an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field


class SimulatorConfig(BaseModel):
    """Parameters of the synthetic market generator."""

    steps: int = Field(200, ge=2, description="Number of price ticks before resolution.")
    p_true_alpha: float = Field(2.0, gt=0, description="Beta(α, β) shape for latent p_true.")
    p_true_beta: float = Field(2.0, gt=0, description="Beta(α, β) shape for latent p_true.")
    ou_theta: float = Field(0.15, ge=0.0, le=1.0, description="OU mean-reversion speed.")
    sigma_start: float = Field(0.10, gt=0.0, description="Noise std at market open.")
    sigma_end: float = Field(0.01, gt=0.0, description="Noise std near resolution.")
    price_floor: float = Field(0.01, gt=0.0, lt=0.5, description="Min tradeable price.")
    price_ceiling: float = Field(0.99, gt=0.5, lt=1.0, description="Max tradeable price.")


@dataclass(frozen=True)
class MarketPath:
    """A single simulated market path.

    ``prices[-1]`` is the resolution and equals ``outcome`` (0.0 or 1.0).
    """

    market_id: str
    prices: np.ndarray = field(repr=False)
    outcome: int
    p_true: float

    def __len__(self) -> int:  # pragma: no cover - trivial passthrough
        return len(self.prices)


def simulate_market(
    cfg: SimulatorConfig,
    rng: np.random.Generator,
    market_id: str = "m0",
) -> MarketPath:
    """Simulate a single market price path ending at resolution."""
    p_true = float(rng.beta(cfg.p_true_alpha, cfg.p_true_beta))
    outcome = int(rng.random() < p_true)

    T = cfg.steps
    sigmas = np.linspace(cfg.sigma_start, cfg.sigma_end, T - 1)
    noise = np.zeros(T - 1, dtype=float)
    x = 0.0
    for t, s in enumerate(sigmas):
        x = (1.0 - cfg.ou_theta) * x + rng.normal(0.0, s)
        noise[t] = x

    prices = np.empty(T, dtype=float)
    prices[: T - 1] = np.clip(p_true + noise, cfg.price_floor, cfg.price_ceiling)
    prices[-1] = float(outcome)
    return MarketPath(market_id=market_id, prices=prices, outcome=outcome, p_true=p_true)


def simulate_markets(
    cfg: SimulatorConfig,
    n_markets: int,
    seed: int = 0,
) -> list[MarketPath]:
    """Simulate ``n_markets`` independent markets from a single seed."""
    rng = np.random.default_rng(seed)
    return [simulate_market(cfg, rng, market_id=f"m{i:05d}") for i in range(n_markets)]
