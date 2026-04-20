"""Live web dashboard for the Polymarket mean-reversion backtest.

Serves a single-page dashboard that lets the user pick bankroll / stake / seed
/ fees etc., runs the backtest in-process, and streams back the summary
metrics, equity curve, per-trade log and a handful of sample market paths
with entry/exit markers.

No live order routing: this is a synthetic backtest viewer.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .backtest import BacktestConfig, BacktestResult, run_backtest, run_backtest_on_markets
from .forward import ForwardConfig, ForwardWorker, load_state
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


# ---------------------------------------------------------------------------
# Forward paper-trading state + controls
# ---------------------------------------------------------------------------


class ForwardStartRequest(BaseModel):
    """Start (or reconfigure) the forward paper-trading worker."""

    bankroll: float = Field(500.0, gt=0.0)
    stake: float = Field(10.0, gt=0.0)
    taker_fee: float = Field(0.01, ge=0.0, le=0.5)
    slippage: float = Field(0.005, ge=0.0, le=0.5)
    poll_interval_s: int = Field(120, ge=30, le=3600)
    max_tracked_markets: int = Field(20, ge=1, le=100)
    min_volume: float = Field(50_000.0, ge=0.0)
    min_hours_to_close: float = Field(6.0, gt=0.0)
    max_hours_to_close: float = Field(24.0 * 30, gt=0.0)
    ewma_halflife_ticks: int = Field(10, ge=1, le=500)
    warmup_ticks: int = Field(5, ge=0, le=500)
    entry_threshold: float = Field(0.05, gt=0.0, lt=0.5)
    stop_threshold: float = Field(0.12, gt=0.0, lt=0.9)
    max_hold_ticks: int = Field(60, ge=1, le=5000)
    cooldown_ticks: int = Field(3, ge=0, le=500)
    reset: bool = Field(
        False, description="If true, wipe state and start fresh even if a state file exists."
    )


def _forward_state_path() -> Path:
    """Resolve where to persist forward-test state.

    Priority:
    1. ``LEVEL_TRADER_FORWARD_STATE`` env var (explicit path).
    2. ``/data/forward_state.json`` when ``/data`` is writable (Fly.io volume).
    3. ``~/.level_trader/forward_state.json`` (local dev).
    """
    override = os.environ.get("LEVEL_TRADER_FORWARD_STATE")
    if override:
        return Path(override)
    fly_volume = Path("/data")
    if fly_volume.is_dir() and os.access(fly_volume, os.W_OK):
        return fly_volume / "forward_state.json"
    return Path.home() / ".level_trader" / "forward_state.json"


def _default_forward_config() -> ForwardConfig:
    """Build a ForwardConfig from LEVEL_TRADER_FORWARD_* env vars (if any)."""

    def fget(name: str, default: float) -> float:
        val = os.environ.get(name)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    def iget(name: str, default: int) -> int:
        return int(fget(name, default))

    return ForwardConfig(
        bankroll=fget("LEVEL_TRADER_FORWARD_BANKROLL", 500.0),
        stake=fget("LEVEL_TRADER_FORWARD_STAKE", 10.0),
        taker_fee=fget("LEVEL_TRADER_FORWARD_TAKER_FEE", 0.01),
        slippage=fget("LEVEL_TRADER_FORWARD_SLIPPAGE", 0.005),
        poll_interval_s=iget("LEVEL_TRADER_FORWARD_POLL_INTERVAL_S", 120),
        max_tracked_markets=iget("LEVEL_TRADER_FORWARD_MAX_MARKETS", 20),
        min_volume=fget("LEVEL_TRADER_FORWARD_MIN_VOLUME", 50_000.0),
        min_hours_to_close=fget("LEVEL_TRADER_FORWARD_MIN_HOURS", 6.0),
        max_hours_to_close=fget("LEVEL_TRADER_FORWARD_MAX_HOURS", 24.0 * 30),
        ewma_halflife_ticks=iget("LEVEL_TRADER_FORWARD_EWMA_HL", 10),
        warmup_ticks=iget("LEVEL_TRADER_FORWARD_WARMUP", 5),
        entry_threshold=fget("LEVEL_TRADER_FORWARD_ENTRY", 0.05),
        stop_threshold=fget("LEVEL_TRADER_FORWARD_STOP", 0.12),
        max_hold_ticks=iget("LEVEL_TRADER_FORWARD_MAX_HOLD", 60),
        cooldown_ticks=iget("LEVEL_TRADER_FORWARD_COOLDOWN", 3),
    )


def _forward_config_from_req(req: ForwardStartRequest) -> ForwardConfig:
    return ForwardConfig(
        bankroll=req.bankroll,
        stake=req.stake,
        taker_fee=req.taker_fee,
        slippage=req.slippage,
        poll_interval_s=req.poll_interval_s,
        max_tracked_markets=req.max_tracked_markets,
        min_volume=req.min_volume,
        min_hours_to_close=req.min_hours_to_close,
        max_hours_to_close=req.max_hours_to_close,
        ewma_halflife_ticks=req.ewma_halflife_ticks,
        warmup_ticks=req.warmup_ticks,
        entry_threshold=req.entry_threshold,
        stop_threshold=req.stop_threshold,
        max_hold_ticks=req.max_hold_ticks,
        cooldown_ticks=req.cooldown_ticks,
    )


class _ForwardRuntime:
    """Module-level holder for the forward worker."""

    worker: ForwardWorker | None = None


def _maybe_autostart_forward() -> None:
    """Auto-start the forward worker on app startup if enabled via env.

    Set ``LEVEL_TRADER_FORWARD_AUTOSTART=1`` to enable. Useful for PaaS
    deployments where we want the worker ticking as soon as the container
    boots.
    """
    flag = os.environ.get("LEVEL_TRADER_FORWARD_AUTOSTART", "").lower()
    if flag in {"0", "false", "no"}:
        return
    # Default: autostart when running on Fly.io (FLY_APP_NAME is auto-set).
    if flag not in {"1", "true", "yes"} and not os.environ.get("FLY_APP_NAME"):
        return
    if _ForwardRuntime.worker is not None and _ForwardRuntime.worker.is_running():
        return
    cfg = _default_forward_config()
    path = _forward_state_path()
    state = load_state(path, cfg=cfg)
    state.cfg = cfg
    worker = ForwardWorker(path, state)
    _ForwardRuntime.worker = worker
    worker.start()


def create_app() -> FastAPI:
    app = FastAPI(title="polymarket-dashboard", version="0.1.0")

    @app.on_event("startup")
    def _on_startup() -> None:
        _maybe_autostart_forward()

    @app.on_event("shutdown")
    def _on_shutdown() -> None:
        worker = _ForwardRuntime.worker
        if worker is not None and worker.is_running():
            worker.stop()

    @app.get("/healthz")
    def healthz() -> dict:
        worker = _ForwardRuntime.worker
        return {
            "ok": True,
            "forward_running": bool(worker and worker.is_running()),
            "poll_count": worker.state.poll_count if worker else 0,
        }

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

    # ---- Forward paper-trading endpoints ----

    @app.get("/api/forward/status")
    def forward_status() -> dict:
        worker = _ForwardRuntime.worker
        if worker is None:
            # Attempt a read-only load of existing state.
            path = _forward_state_path()
            if path.exists():
                from .forward import snapshot as _snap

                state = load_state(path)
                snap = _snap(state)
                snap["running"] = False
                return snap
            return {
                "running": False,
                "started_at": None,
                "last_tick_at": None,
                "poll_count": 0,
                "last_error": None,
                "bankroll_start": 500.0,
                "bankroll_current": 500.0,
                "realized_pnl": 0.0,
                "n_closed_trades": 0,
                "n_tracked_markets": 0,
                "n_resolved_markets": 0,
                "n_open_positions": 0,
                "open_positions": [],
                "tracked_markets": [],
                "closed_trades": [],
                "equity": [],
                "cfg": None,
            }
        snap = worker.snapshot()
        snap["running"] = worker.is_running()
        return snap

    @app.post("/api/forward/start")
    def forward_start(req: ForwardStartRequest) -> dict:
        cfg = _forward_config_from_req(req)
        path = _forward_state_path()
        if req.reset and path.exists():
            path.unlink()
        state = load_state(path, cfg=cfg)
        state.cfg = cfg
        worker = _ForwardRuntime.worker
        if worker is not None and worker.is_running():
            worker.stop()
        worker = ForwardWorker(path, state)
        _ForwardRuntime.worker = worker
        worker.start()
        snap = worker.snapshot()
        snap["running"] = worker.is_running()
        return snap

    @app.post("/api/forward/stop")
    def forward_stop() -> dict:
        worker = _ForwardRuntime.worker
        if worker is None:
            raise HTTPException(404, "forward worker not started")
        worker.stop()
        snap = worker.snapshot()
        snap["running"] = worker.is_running()
        return snap

    @app.post("/api/forward/poll_now")
    def forward_poll_now() -> dict:
        """Force a single poll tick without waiting for the interval.

        Useful for manual testing and fast-forwarding the first tick.
        """
        worker = _ForwardRuntime.worker
        if worker is None:
            raise HTTPException(404, "forward worker not started")
        from .forward import poll_once, save_state

        with worker._lock:  # noqa: SLF001
            poll_once(worker.state)
            save_state(worker.state, worker.state_path)
        snap = worker.snapshot()
        snap["running"] = worker.is_running()
        return snap

    return app


app = create_app()
