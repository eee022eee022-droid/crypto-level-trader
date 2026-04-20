"""Forward paper-trading engine for live Polymarket markets.

Runs the same mean-reversion strategy against currently-open Polymarket
markets in real time, using midpoint quotes from the public CLOB endpoints.
Nothing is ever submitted on-chain — this just snapshots prices, runs the
strategy step-by-step, and records virtual PnL to a JSON state file that
survives process restarts.

The worker is designed so :func:`poll_once` is idempotent: call it on a
schedule (once per poll interval) and the strategy will accrue one tick per
market per call. Interval is typically 60-300 seconds, so the ``halflife``
and ``max_hold_ticks`` knobs are interpreted in *poll ticks*, not minutes.

This module has no FastAPI dependency — the dashboard just calls
:func:`start_worker` / :func:`stop_worker` and reads :func:`snapshot`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
_CLOB_MIDPOINT_URL = "https://clob.polymarket.com/midpoint"
_USER_AGENT = "level-trader-polymarket-adapter/0.1"
_HTTP_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Config / state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ForwardConfig:
    """Static configuration for a forward-paper-trading session."""

    bankroll: float = 500.0
    stake: float = 10.0
    taker_fee: float = 0.01
    slippage: float = 0.005

    poll_interval_s: int = 120
    max_tracked_markets: int = 20
    min_volume: float = 50_000.0
    min_hours_to_close: float = 6.0
    max_hours_to_close: float = 24.0 * 30

    # Strategy knobs — units are *poll ticks*, not minutes.
    ewma_halflife_ticks: int = 10
    warmup_ticks: int = 5
    entry_threshold: float = 0.05
    stop_threshold: float = 0.12
    max_hold_ticks: int = 60
    cooldown_ticks: int = 3
    min_price: float = 0.05
    max_price: float = 0.95


@dataclass
class OpenPosition:
    side: str  # "yes" | "no"
    entry_ts: int
    entry_price: float
    stake: float
    quantity: float
    fees_paid: float


@dataclass
class ForwardTrade:
    market_id: str
    question: str
    side: str
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    exit_reason: str
    stake: float
    fees_paid: float
    pnl: float


@dataclass
class MarketState:
    market_id: str
    token_id: str
    question: str
    slug: str
    end_ts: int | None
    first_seen_ts: int
    last_polled_ts: int | None = None
    price_history: list[tuple[int, float]] = field(default_factory=list)
    ewma: float | None = None
    ticks: int = 0
    open_position: OpenPosition | None = None
    cooldown_until_tick: int = 0
    resolved: bool = False
    outcome: int | None = None  # 0 or 1 when resolved


@dataclass
class ForwardState:
    cfg: ForwardConfig
    started_at: int
    last_tick_at: int | None = None
    markets: dict[str, MarketState] = field(default_factory=dict)
    closed_trades: list[ForwardTrade] = field(default_factory=list)
    equity_snapshots: list[tuple[int, float]] = field(default_factory=list)
    poll_count: int = 0
    last_error: str | None = None

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def bankroll(self) -> float:
        return self.cfg.bankroll + self.realized_pnl


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, params: dict | None = None) -> Any:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_iso(s: str | None) -> int | None:
    if not s:
        return None
    try:
        import datetime as _dt

        return int(_dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


def fetch_open_markets(cfg: ForwardConfig) -> list[dict]:
    """Return currently open Polymarket markets matching ``cfg`` filters."""
    now = int(time.time())
    page = _http_get(
        _GAMMA_URL,
        {
            "closed": "false",
            "active": "true",
            "archived": "false",
            "limit": "200",
            "order": "volumeNum",
            "ascending": "false",
        },
    )
    if not isinstance(page, list):
        return []
    out: list[dict] = []
    for m in page:
        end_ts = _parse_iso(m.get("endDate"))
        if end_ts is None or end_ts <= now:
            continue
        hours_to_close = (end_ts - now) / 3600.0
        if hours_to_close < cfg.min_hours_to_close:
            continue
        if hours_to_close > cfg.max_hours_to_close:
            continue
        try:
            vol = float(m.get("volumeNum") or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol < cfg.min_volume:
            continue
        tokens_raw = m.get("clobTokenIds") or "[]"
        try:
            tokens = json.loads(tokens_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not tokens:
            continue
        out.append(m)
        if len(out) >= cfg.max_tracked_markets * 3:
            break
    return out


def fetch_midpoint(token_id: str) -> float | None:
    try:
        resp = _http_get(_CLOB_MIDPOINT_URL, {"token_id": token_id})
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        log.debug("midpoint %s -> %s", token_id[:10], e)
        return None
    if not isinstance(resp, dict):
        return None
    mid = resp.get("mid")
    if mid is None:
        return None
    try:
        return float(mid)
    except (TypeError, ValueError):
        return None


def fetch_market_resolution(market_id: str) -> tuple[bool, int | None]:
    """Re-query a market by id. Returns ``(resolved, outcome)``.

    ``outcome`` is 1 if YES, 0 if NO, else None.
    """
    try:
        data = _http_get(_GAMMA_URL, {"id": market_id})
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False, None
    if not isinstance(data, list) or not data:
        return False, None
    m = data[0]
    if not m.get("closed") and not m.get("resolved"):
        return False, None
    prices_raw = m.get("outcomePrices") or "[]"
    try:
        prices = json.loads(prices_raw)
    except (json.JSONDecodeError, TypeError):
        return True, None
    if not prices:
        return True, None
    outcome = 1 if str(prices[0]) == "1" else 0
    return True, outcome


# ---------------------------------------------------------------------------
# Strategy step (incremental, per-tick)
# ---------------------------------------------------------------------------


def _ewma_step(prev: float | None, price: float, halflife: int) -> float:
    if prev is None:
        return price
    alpha = 1.0 - 0.5 ** (1.0 / max(halflife, 1))
    return alpha * price + (1.0 - alpha) * prev


def _fill_price(side: str, ref: float, slippage: float, is_entry: bool) -> float:
    if is_entry:
        if side == "yes":
            return min(1.0, ref * (1.0 + slippage))
        return max(0.0, ref * (1.0 - slippage))
    if side == "yes":
        return max(0.0, ref * (1.0 - slippage))
    return min(1.0, ref * (1.0 + slippage))


def _open_position(
    side: str,
    ts: int,
    ref_price: float,
    cfg: ForwardConfig,
) -> OpenPosition:
    fill = _fill_price(side, ref_price, cfg.slippage, is_entry=True)
    token_price = fill if side == "yes" else (1.0 - fill)
    token_price = max(min(token_price, 0.999), 0.001)
    quantity = cfg.stake / token_price
    fee = cfg.stake * cfg.taker_fee
    return OpenPosition(
        side=side,
        entry_ts=ts,
        entry_price=fill,
        stake=cfg.stake,
        quantity=quantity,
        fees_paid=fee,
    )


def _close_position(
    pos: OpenPosition,
    ts: int,
    ref_price: float,
    reason: str,
    cfg: ForwardConfig,
    market_id: str,
    question: str,
) -> ForwardTrade:
    fill = _fill_price(pos.side, ref_price, cfg.slippage, is_entry=False)
    token_exit = fill if pos.side == "yes" else (1.0 - fill)
    proceeds = pos.quantity * token_exit
    exit_fee = proceeds * cfg.taker_fee
    pnl = proceeds - pos.stake - exit_fee - pos.fees_paid
    return ForwardTrade(
        market_id=market_id,
        question=question,
        side=pos.side,
        entry_ts=pos.entry_ts,
        entry_price=pos.entry_price,
        exit_ts=ts,
        exit_price=fill,
        exit_reason=reason,
        stake=pos.stake,
        fees_paid=pos.fees_paid + exit_fee,
        pnl=pnl,
    )


def _step_strategy(
    mkt: MarketState,
    price: float,
    now_ts: int,
    cfg: ForwardConfig,
    closed_trades: list[ForwardTrade],
) -> None:
    """Advance ``mkt`` one tick with ``price``. Append new closed trades."""
    mkt.ticks += 1
    mkt.price_history.append((now_ts, price))
    mkt.ewma = _ewma_step(mkt.ewma, price, cfg.ewma_halflife_ticks)
    mkt.last_polled_ts = now_ts
    ewma = mkt.ewma
    assert ewma is not None

    pos = mkt.open_position
    if pos is not None:
        deviation = price - ewma
        reverted = (pos.side == "yes" and price >= ewma) or (pos.side == "no" and price <= ewma)
        if reverted:
            tr = _close_position(pos, now_ts, price, "reverted", cfg, mkt.market_id, mkt.question)
            closed_trades.append(tr)
            mkt.open_position = None
            mkt.cooldown_until_tick = mkt.ticks + cfg.cooldown_ticks
            return
        adverse = (pos.side == "yes" and deviation <= -cfg.stop_threshold) or (
            pos.side == "no" and deviation >= cfg.stop_threshold
        )
        if adverse:
            tr = _close_position(pos, now_ts, price, "stop", cfg, mkt.market_id, mkt.question)
            closed_trades.append(tr)
            mkt.open_position = None
            mkt.cooldown_until_tick = mkt.ticks + cfg.cooldown_ticks
            return
        held = mkt.ticks - _entry_tick_of(mkt, pos)
        if held >= cfg.max_hold_ticks:
            tr = _close_position(pos, now_ts, price, "timeout", cfg, mkt.market_id, mkt.question)
            closed_trades.append(tr)
            mkt.open_position = None
            mkt.cooldown_until_tick = mkt.ticks + cfg.cooldown_ticks
            return
        return

    if mkt.ticks < cfg.warmup_ticks or mkt.ticks < mkt.cooldown_until_tick:
        return
    if price < cfg.min_price or price > cfg.max_price:
        return
    deviation = price - ewma
    if deviation >= cfg.entry_threshold:
        mkt.open_position = _open_position("no", now_ts, price, cfg)
    elif deviation <= -cfg.entry_threshold:
        mkt.open_position = _open_position("yes", now_ts, price, cfg)


def _entry_tick_of(mkt: MarketState, pos: OpenPosition) -> int:
    """Recover the tick index at which ``pos`` was opened.

    We don't persist the tick — derive it by linear scan. ``price_history``
    is the authoritative time axis.
    """
    target_ts = pos.entry_ts
    for idx, (ts, _p) in enumerate(mkt.price_history):
        if ts >= target_ts:
            return idx + 1  # ticks are 1-based in _step_strategy
    return mkt.ticks


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------


def poll_once(state: ForwardState) -> None:
    """Run one full polling tick: refresh market list, fetch prices, step strategy."""
    now = int(time.time())
    state.poll_count += 1
    state.last_tick_at = now
    state.last_error = None

    try:
        candidates = fetch_open_markets(state.cfg)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        state.last_error = f"gamma fetch: {e}"
        log.warning("gamma fetch failed: %s", e)
        return

    # Add new markets up to the cap.
    tracked_open = sum(1 for m in state.markets.values() if not m.resolved)
    for m in candidates:
        if tracked_open >= state.cfg.max_tracked_markets:
            break
        mid = str(m.get("id"))
        if mid in state.markets:
            continue
        tokens_raw = m.get("clobTokenIds") or "[]"
        try:
            tokens = json.loads(tokens_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not tokens:
            continue
        state.markets[mid] = MarketState(
            market_id=mid,
            token_id=str(tokens[0]),
            question=str(m.get("question", ""))[:200],
            slug=str(m.get("slug", "")),
            end_ts=_parse_iso(m.get("endDate")),
            first_seen_ts=now,
        )
        tracked_open += 1

    # Step each still-open tracked market.
    for mkt in list(state.markets.values()):
        if mkt.resolved:
            continue
        # Market may have closed since we last polled it.
        if mkt.end_ts is not None and now >= mkt.end_ts:
            _settle_resolved(mkt, now, state)
            continue
        price = fetch_midpoint(mkt.token_id)
        if price is None:
            mkt.last_polled_ts = now
            continue
        _step_strategy(mkt, price, now, state.cfg, state.closed_trades)

    state.equity_snapshots.append((now, round(state.bankroll, 4)))
    # Keep at most the last ~2000 equity points.
    if len(state.equity_snapshots) > 2000:
        state.equity_snapshots = state.equity_snapshots[-2000:]


def _settle_resolved(mkt: MarketState, now: int, state: ForwardState) -> None:
    """Mark ``mkt`` resolved and close any open position at the outcome."""
    resolved, outcome = fetch_market_resolution(mkt.market_id)
    mkt.resolved = resolved
    mkt.outcome = outcome
    if not resolved or outcome is None:
        return
    pos = mkt.open_position
    if pos is not None:
        ref_price = float(outcome)
        tr = _close_position(
            pos, now, ref_price, "resolution", state.cfg, mkt.market_id, mkt.question
        )
        state.closed_trades.append(tr)
        mkt.open_position = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def state_to_json(state: ForwardState) -> str:
    def default(o):
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(f"not json-serializable: {type(o)}")

    return json.dumps(asdict(state), default=default, indent=2)


def state_from_json(text: str) -> ForwardState:
    raw = json.loads(text)
    cfg = ForwardConfig(**raw.get("cfg", {}))
    markets_raw = raw.get("markets") or {}
    markets: dict[str, MarketState] = {}
    for mid, mr in markets_raw.items():
        pos_raw = mr.get("open_position")
        pos = OpenPosition(**pos_raw) if pos_raw else None
        markets[mid] = MarketState(
            market_id=mr["market_id"],
            token_id=mr["token_id"],
            question=mr.get("question", ""),
            slug=mr.get("slug", ""),
            end_ts=mr.get("end_ts"),
            first_seen_ts=mr.get("first_seen_ts", 0),
            last_polled_ts=mr.get("last_polled_ts"),
            price_history=[tuple(p) for p in mr.get("price_history") or []],
            ewma=mr.get("ewma"),
            ticks=mr.get("ticks", 0),
            open_position=pos,
            cooldown_until_tick=mr.get("cooldown_until_tick", 0),
            resolved=mr.get("resolved", False),
            outcome=mr.get("outcome"),
        )
    closed_trades = [ForwardTrade(**t) for t in raw.get("closed_trades") or []]
    equity = [tuple(e) for e in raw.get("equity_snapshots") or []]
    return ForwardState(
        cfg=cfg,
        started_at=raw.get("started_at", int(time.time())),
        last_tick_at=raw.get("last_tick_at"),
        markets=markets,
        closed_trades=closed_trades,
        equity_snapshots=equity,
        poll_count=raw.get("poll_count", 0),
        last_error=raw.get("last_error"),
    )


def load_state(path: Path, cfg: ForwardConfig | None = None) -> ForwardState:
    if path.exists():
        try:
            state = state_from_json(path.read_text(encoding="utf-8"))
            if cfg is not None:
                state.cfg = cfg  # allow hot-reconfig on restart
            return state
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("forward state %s is corrupt, starting fresh: %s", path, e)
    return ForwardState(cfg=cfg or ForwardConfig(), started_at=int(time.time()))


def save_state(state: ForwardState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(state_to_json(state), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class ForwardWorker:
    """Background thread that polls Polymarket and updates a ForwardState."""

    def __init__(self, state_path: Path, state: ForwardState) -> None:
        self.state_path = state_path
        self.state = state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="forward-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        # Immediate first tick, then sleep-poll cycle.
        while not self._stop.is_set():
            try:
                with self._lock:
                    poll_once(self.state)
                    save_state(self.state, self.state_path)
            except Exception as e:  # noqa: BLE001 - worker must stay alive
                log.exception("poll_once failed: %s", e)
                self.state.last_error = f"{type(e).__name__}: {e}"
            # Sleep in small chunks so stop() returns quickly.
            remaining = self.state.cfg.poll_interval_s
            while remaining > 0 and not self._stop.is_set():
                chunk = min(remaining, 1.0)
                time.sleep(chunk)
                remaining -= chunk

    def snapshot(self) -> dict:
        with self._lock:
            return snapshot(self.state)


def snapshot(state: ForwardState) -> dict:
    """Read-only snapshot shaped for the dashboard."""
    open_positions = []
    for mkt in state.markets.values():
        if mkt.open_position is None or mkt.resolved:
            continue
        pos = mkt.open_position
        last_price = mkt.price_history[-1][1] if mkt.price_history else pos.entry_price
        # Unrealized PnL at current token price.
        token_exit = last_price if pos.side == "yes" else (1.0 - last_price)
        unreal = pos.quantity * token_exit - pos.stake - pos.fees_paid
        open_positions.append(
            {
                "market_id": mkt.market_id,
                "question": mkt.question,
                "side": pos.side,
                "entry_ts": pos.entry_ts,
                "entry_price": round(pos.entry_price, 4),
                "last_price": round(last_price, 4),
                "stake": round(pos.stake, 2),
                "unrealized_pnl": round(unreal, 4),
            }
        )

    tracked = []
    for mkt in state.markets.values():
        tracked.append(
            {
                "market_id": mkt.market_id,
                "question": mkt.question,
                "ticks": mkt.ticks,
                "last_price": mkt.price_history[-1][1] if mkt.price_history else None,
                "ewma": round(mkt.ewma, 4) if mkt.ewma is not None else None,
                "has_open": mkt.open_position is not None,
                "resolved": mkt.resolved,
                "outcome": mkt.outcome,
                "end_ts": mkt.end_ts,
            }
        )

    closed = [asdict(t) for t in state.closed_trades]
    wins = sum(1 for t in state.closed_trades if t.pnl > 0)
    total = len(state.closed_trades)
    gross_win = sum(t.pnl for t in state.closed_trades if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in state.closed_trades if t.pnl < 0)

    return {
        "running": None,  # filled by worker wrapper
        "started_at": state.started_at,
        "last_tick_at": state.last_tick_at,
        "poll_count": state.poll_count,
        "last_error": state.last_error,
        "bankroll_start": state.cfg.bankroll,
        "bankroll_current": round(state.bankroll, 4),
        "realized_pnl": round(state.realized_pnl, 4),
        "n_closed_trades": total,
        "winrate": round(wins / total, 4) if total else None,
        "profit_factor": round(gross_win / gross_loss, 4)
        if gross_loss > 0
        else None,
        "n_tracked_markets": len(state.markets),
        "n_resolved_markets": sum(1 for m in state.markets.values() if m.resolved),
        "n_open_positions": len(open_positions),
        "open_positions": open_positions,
        "tracked_markets": tracked,
        "closed_trades": closed,
        "equity": [{"ts": ts, "bankroll": bk} for ts, bk in state.equity_snapshots],
        "cfg": asdict(state.cfg),
    }
