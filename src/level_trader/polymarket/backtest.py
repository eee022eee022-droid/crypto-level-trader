"""Backtest engine for Polymarket strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .simulator import SimulatorConfig, simulate_markets
from .strategy import MeanReversionConfig, MeanReversionStrategy, Trade


class BacktestConfig(BaseModel):
    """Top-level backtest configuration."""

    n_markets: int = Field(500, ge=1)
    seed: int = Field(7, ge=0)
    taker_fee: float = Field(0.01, ge=0.0, le=1.0)
    slippage: float = Field(0.005, ge=0.0, le=1.0)
    starting_bankroll: float = Field(10_000.0, gt=0.0)
    simulator: SimulatorConfig = Field(default_factory=SimulatorConfig)
    strategy: MeanReversionConfig = Field(default_factory=MeanReversionConfig)


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    starting_bankroll: float = 10_000.0
    ending_bankroll: float = 10_000.0
    n_markets: int = 0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)

    @property
    def n_losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fees_paid for t in self.trades)

    @property
    def winrate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def gross_win(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl > 0)

    @property
    def gross_loss(self) -> float:
        return -sum(t.pnl for t in self.trades if t.pnl <= 0)

    @property
    def profit_factor(self) -> float:
        if self.gross_loss > 0:
            return self.gross_win / self.gross_loss
        return float("inf") if self.gross_win > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.n_trades if self.n_trades else 0.0

    @property
    def sharpe(self) -> float:
        """Trade-level Sharpe (mean/std of PnL, un-annualised)."""
        if self.n_trades < 2:
            return 0.0
        mean = self.avg_pnl
        variance = sum((t.pnl - mean) ** 2 for t in self.trades) / (self.n_trades - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            return 0.0
        return mean / std

    @property
    def roi(self) -> float:
        if self.starting_bankroll <= 0:
            return 0.0
        return (self.ending_bankroll - self.starting_bankroll) / self.starting_bankroll

    def summary(self) -> dict[str, float | int]:
        return {
            "n_markets": self.n_markets,
            "n_trades": self.n_trades,
            "wins": self.n_wins,
            "losses": self.n_losses,
            "winrate": round(self.winrate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_pnl_per_trade": round(self.avg_pnl, 4),
            "gross_win": round(self.gross_win, 2),
            "gross_loss": round(self.gross_loss, 2),
            "profit_factor": round(self.profit_factor, 3),
            "sharpe_trade": round(self.sharpe, 3),
            "starting_bankroll": round(self.starting_bankroll, 2),
            "ending_bankroll": round(self.ending_bankroll, 2),
            "roi": round(self.roi, 4),
            "total_fees": round(self.total_fees, 2),
        }


def run_backtest(cfg: BacktestConfig) -> BacktestResult:
    """Simulate ``cfg.n_markets`` markets and run the strategy across all of them."""
    markets = simulate_markets(cfg.simulator, cfg.n_markets, seed=cfg.seed)
    strategy = MeanReversionStrategy(cfg.strategy)
    all_trades: list[Trade] = []
    for market in markets:
        trades = strategy.trade_market(market, taker_fee=cfg.taker_fee, slippage=cfg.slippage)
        all_trades.extend(trades)

    result = BacktestResult(
        trades=all_trades,
        starting_bankroll=cfg.starting_bankroll,
        ending_bankroll=cfg.starting_bankroll + sum(t.pnl for t in all_trades),
        n_markets=cfg.n_markets,
    )
    return result
