"""Tests for the forward paper-trading engine (no network I/O)."""
from __future__ import annotations

import time

import pytest

from level_trader.polymarket.forward import (
    ForwardConfig,
    ForwardState,
    ForwardTrade,
    MarketState,
    _step_strategy,
    load_state,
    save_state,
    snapshot,
    state_from_json,
    state_to_json,
)


def _fresh_state(cfg: ForwardConfig | None = None) -> ForwardState:
    return ForwardState(cfg=cfg or ForwardConfig(), started_at=int(time.time()))


def _add_market(state: ForwardState, mid: str = "m1") -> MarketState:
    mkt = MarketState(
        market_id=mid,
        token_id=f"tok-{mid}",
        question=f"question {mid}",
        slug=f"slug-{mid}",
        end_ts=int(time.time()) + 3600 * 48,
        first_seen_ts=int(time.time()),
    )
    state.markets[mid] = mkt
    return mkt


def test_strategy_opens_and_exits_on_reversion() -> None:
    cfg = ForwardConfig(
        stake=10.0,
        taker_fee=0.0,
        slippage=0.0,
        warmup_ticks=2,
        ewma_halflife_ticks=3,
        entry_threshold=0.05,
        stop_threshold=0.2,
        max_hold_ticks=100,
        cooldown_ticks=1,
    )
    state = _fresh_state(cfg)
    mkt = _add_market(state)
    closed = state.closed_trades

    # Prime with stable price near 0.5 to settle the EWMA.
    for i, p in enumerate([0.5, 0.5, 0.5, 0.5, 0.5]):
        _step_strategy(mkt, p, 1000 + i, cfg, closed)
    assert mkt.open_position is None
    # Now spike the price up so strategy enters NO.
    _step_strategy(mkt, 0.65, 2000, cfg, closed)
    assert mkt.open_position is not None
    assert mkt.open_position.side == "no"
    # And revert back through EWMA → closes the trade.
    _step_strategy(mkt, 0.45, 2001, cfg, closed)
    assert mkt.open_position is None
    assert len(closed) == 1
    # NO position opened at 0.65 and closed at 0.45 → profit.
    assert closed[0].pnl > 0


def test_strategy_stop_loss_closes_position() -> None:
    cfg = ForwardConfig(
        stake=10.0,
        taker_fee=0.0,
        slippage=0.0,
        warmup_ticks=2,
        ewma_halflife_ticks=3,
        entry_threshold=0.05,
        stop_threshold=0.12,
        max_hold_ticks=100,
        cooldown_ticks=0,
    )
    state = _fresh_state(cfg)
    mkt = _add_market(state)
    closed = state.closed_trades

    for i, p in enumerate([0.5, 0.5, 0.5, 0.5, 0.5]):
        _step_strategy(mkt, p, 1000 + i, cfg, closed)
    # Spike down → strategy enters YES (expects mean revert UP).
    _step_strategy(mkt, 0.4, 2000, cfg, closed)
    assert mkt.open_position is not None
    assert mkt.open_position.side == "yes"
    # Price keeps going down past the stop → exit at loss.
    _step_strategy(mkt, 0.2, 2001, cfg, closed)
    assert mkt.open_position is None
    assert len(closed) == 1
    assert closed[0].exit_reason == "stop"
    assert closed[0].pnl < 0


def test_state_json_roundtrip() -> None:
    state = _fresh_state()
    mkt = _add_market(state, "m42")
    mkt.price_history.append((1000, 0.5))
    mkt.ewma = 0.5
    mkt.ticks = 1
    state.closed_trades.append(
        ForwardTrade(
            market_id="m42",
            question="q",
            side="yes",
            entry_ts=900,
            entry_price=0.4,
            exit_ts=1000,
            exit_price=0.5,
            exit_reason="reverted",
            stake=10.0,
            fees_paid=0.2,
            pnl=1.5,
        )
    )
    state.equity_snapshots.append((1000, 501.5))

    text = state_to_json(state)
    restored = state_from_json(text)

    assert restored.started_at == state.started_at
    assert set(restored.markets.keys()) == {"m42"}
    assert restored.markets["m42"].price_history == [(1000, 0.5)]
    assert restored.markets["m42"].ticks == 1
    assert len(restored.closed_trades) == 1
    assert restored.closed_trades[0].pnl == 1.5
    assert restored.equity_snapshots == [(1000, 501.5)]


def test_load_state_returns_fresh_when_file_missing(tmp_path) -> None:
    path = tmp_path / "does-not-exist.json"
    state = load_state(path)
    assert state.poll_count == 0
    assert state.markets == {}


def test_save_and_load_state_roundtrip(tmp_path) -> None:
    state = _fresh_state()
    _add_market(state, "abc")
    state.markets["abc"].ticks = 5
    path = tmp_path / "fwd.json"
    save_state(state, path)

    restored = load_state(path)
    assert restored.markets["abc"].ticks == 5


def test_snapshot_shape() -> None:
    state = _fresh_state()
    mkt = _add_market(state)
    mkt.price_history.append((1000, 0.5))
    mkt.ewma = 0.5
    snap = snapshot(state)
    for key in (
        "bankroll_start",
        "bankroll_current",
        "realized_pnl",
        "n_closed_trades",
        "n_tracked_markets",
        "open_positions",
        "closed_trades",
        "tracked_markets",
        "equity",
        "cfg",
    ):
        assert key in snap
    assert snap["bankroll_start"] == pytest.approx(500.0)
    assert snap["n_tracked_markets"] == 1


def test_forward_config_rejects_bad_input() -> None:
    # Dataclass accepts anything, but ensure a reasonable default.
    cfg = ForwardConfig()
    assert cfg.bankroll > 0
    assert cfg.stake > 0
    assert cfg.entry_threshold > 0
