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

from .backtest import BacktestConfig, BacktestResult, run_backtest, run_backtest_on_markets
from .live_data import LiveFetchConfig, fetch_resolved_markets
from .simulator import MarketPath, SimulatorConfig
from .strategy import MeanReversionConfig, Trade

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_HTML = (BASE_DIR / "dashboard.html").read_text(encoding="utf-8")


class RunRequest(BaseModel):
    """JSON payload accepted by ``POST /api/run`` and ``POST /api/run_live``."""

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


class LiveRunRequest(RunRequest):
    """Run on real Polymarket markets instead of the synthetic simulator."""

    min_volume: float = Field(10_000.0, ge=0.0)
    min_duration_hours: float = Field(24.0, gt=0.0, le=24.0 * 365)
    max_age_days: int = Field(30, ge=1, le=365)
    fidelity_minutes: int = Field(60, ge=1, le=1440)


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


def _backtest_config(req: RunRequest) -> BacktestConfig:
    return BacktestConfig(
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


def _result_to_payload(result: BacktestResult, extra: dict | None = None) -> dict:
    equity = _equity_curve(result)
    trades = [_trade_row(t) for t in result.trades]
    summary = result.summary()
    summary["max_drawdown"] = _max_drawdown(equity)
    summary["bankroll_ever_wiped"] = bool(min(equity) <= 0.0)
    if extra:
        summary.update(extra)
    return {
        "summary": summary,
        "equity": equity,
        "drawdown": _drawdown_series(equity),
        "trades": trades,
    }


# In-process cache of fetched live markets — Polymarket rate-limits us and
# fetching 20-50 markets takes several seconds. Keyed by the fetch params.
_LIVE_CACHE: dict[tuple, list[MarketPath]] = {}


def _fetch_live_markets_cached(req: LiveRunRequest) -> list[MarketPath]:
    key = (
        req.n_markets,
        req.min_volume,
        req.min_duration_hours,
        req.max_age_days,
        req.fidelity_minutes,
    )
    cached = _LIVE_CACHE.get(key)
    if cached is not None:
        return cached
    markets = fetch_resolved_markets(
        LiveFetchConfig(
            limit=req.n_markets,
            min_volume=req.min_volume,
            min_duration_hours=req.min_duration_hours,
            max_age_days=req.max_age_days,
            fidelity_minutes=req.fidelity_minutes,
        )
    )
    _LIVE_CACHE[key] = markets
    return markets


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
        result = run_backtest(_backtest_config(req))
        return _result_to_payload(result, extra={"data_source": "synthetic"})

    @app.post("/api/run_live")
    def run_live(req: LiveRunRequest) -> dict:
        markets = _fetch_live_markets_cached(req)
        if not markets:
            return {
                "summary": {
                    "n_markets": 0,
                    "n_trades": 0,
                    "data_source": "polymarket_live",
                    "error": "no resolved markets with retained price history matched the filters; relax min_volume / min_duration / max_age",
                },
                "equity": [req.bankroll],
                "drawdown": [0.0],
                "trades": [],
            }
        result = run_backtest_on_markets(markets, _backtest_config(req))
        avg_ticks = sum(len(m.prices) for m in markets) / len(markets)
        return _result_to_payload(
            result,
            extra={
                "data_source": "polymarket_live",
                "avg_ticks_per_market": round(avg_ticks, 1),
                "n_yes_resolutions": sum(1 for m in markets if m.outcome == 1),
                "n_no_resolutions": sum(1 for m in markets if m.outcome == 0),
            },
        )

    return app


app = create_app()
