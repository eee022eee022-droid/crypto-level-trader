"""Live web dashboard for the Polymarket mean-reversion backtest.

Serves a single-page dashboard that lets the user pick bankroll / stake / seed
/ fees etc., runs the backtest in-process, and streams back the summary
metrics, equity curve, per-trade log and a handful of sample market paths
with entry/exit markers.

No live order routing: this is a synthetic backtest viewer.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .simulator import SimulatorConfig
from .strategy import MeanReversionConfig, Trade

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = (BASE_DIR / "dashboard.html").read_text(encoding="utf-8")


class RunRequest(BaseModel):
    """JSON payload accepted by ``POST /api/run``."""

    bankroll: float = Field(500.0, gt=0.0)
    stake: float = Field(10.0, gt=0.0)
    n_markets: int = Field(200, ge=1, le=5000)
    seed: int = Field(7, ge=0)
    steps: int = Field(200, ge=20, le=2000)
    taker_fee: float = Field(0.01, ge=0.0, le=0.5)
    slippage: float = Field(0.005, ge=0.0, le=0.5)
    ewma_halflife: int = Field(20, ge=1, le=500)
    entry_threshold: float = Field(0.05, gt=0.0, lt=0.5)
    stop_threshold: float = Field(0.12, gt=0.0, lt=0.9)
    max_hold: int = Field(60, ge=1, le=5000)


def _trade_row(t: Trade) -> dict:
    return {
        "market_id": t.market_id,
        "side": t.side,
        "entry_tick": t.entry_tick,
        "entry_price": round(t.entry_price, 4),
        "exit_tick": t.exit_tick,
        "exit_price": round(t.exit_price, 4) if t.exit_price is not None else None,
        "exit_reason": t.exit_reason,
        "stake": round(t.stake, 2),
        "fees_paid": round(t.fees_paid, 4),
        "pnl": round(t.pnl, 4),
    }


def _equity_curve(result: BacktestResult) -> list[float]:
    eq = result.starting_bankroll
    out = [round(eq, 4)]
    for t in result.trades:
        eq += t.pnl
        out.append(round(eq, 4))
    return out


def _drawdown_series(equity: list[float]) -> list[float]:
    peak = equity[0]
    out = []
    for v in equity:
        peak = max(peak, v)
        out.append(round(v - peak, 4))
    return out


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for v in equity:
        peak = max(peak, v)
        worst = min(worst, v - peak)
    return round(worst, 4)


def create_app() -> FastAPI:
    app = FastAPI(title="polymarket-dashboard", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/defaults")
    def defaults() -> dict:
        return RunRequest().model_dump()

    @app.post("/api/run")
    def run(req: RunRequest) -> dict:
        cfg = BacktestConfig(
            n_markets=req.n_markets,
            seed=req.seed,
            taker_fee=req.taker_fee,
            slippage=req.slippage,
            starting_bankroll=req.bankroll,
            simulator=SimulatorConfig(steps=req.steps),
            strategy=MeanReversionConfig(
                ewma_halflife=req.ewma_halflife,
                entry_threshold=req.entry_threshold,
                stop_threshold=req.stop_threshold,
                max_hold_ticks=req.max_hold,
                stake_per_trade=req.stake,
            ),
        )
        result = run_backtest(cfg)
        equity = _equity_curve(result)
        trades = [_trade_row(t) for t in result.trades]
        summary = result.summary()
        summary["max_drawdown"] = _max_drawdown(equity)
        summary["bankroll_ever_wiped"] = bool(min(equity) <= 0.0)
        return {
            "summary": summary,
            "equity": equity,
            "drawdown": _drawdown_series(equity),
            "trades": trades,
        }

    return app


app = create_app()
