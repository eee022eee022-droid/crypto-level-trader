"""Polymarket prediction-market strategy and backtester.

Polymarket is a binary prediction market: each market has a YES token and a NO
token that each redeem for $1 if their side wins. Prices live in (0, 1) and
settle to 0 or 1 at resolution. This module provides:

* A synthetic market simulator whose price paths match the stylised facts of
  real prediction markets (mean-reverting noise around a latent true
  probability, with the noise amplitude decaying as resolution approaches).
* A mean-reversion strategy that exploits that noise by trading against
  short-term deviations from a long EWMA of price.
* A backtest engine with taker fees + slippage that settles every open
  position at resolution and produces standard performance metrics.

The strategy never needs to know the latent true probability: its only signal
is the observed price series, which is exactly what the Polymarket CLOB/AMM
would expose in production. Plugging in live market data from
``data_api.polymarket.com`` is a drop-in replacement for the simulator.
"""

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .simulator import MarketPath, SimulatorConfig, simulate_market, simulate_markets
from .strategy import MeanReversionConfig, MeanReversionStrategy, Trade

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "MarketPath",
    "MeanReversionConfig",
    "MeanReversionStrategy",
    "SimulatorConfig",
    "Trade",
    "run_backtest",
    "simulate_market",
    "simulate_markets",
]
