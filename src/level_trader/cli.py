"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .broker.base import Broker
from .broker.paper import PaperBroker
from .config import Config
from .exchange.gate import GateFutures
from .exchange.universe import select_universe
from .levels.detector import detect_levels, ohlcv_to_df
from .polymarket import BacktestConfig, MeanReversionConfig, SimulatorConfig, run_backtest
from .trader import Trader

console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def _build_exchange(cfg: Config, want_live: bool) -> GateFutures:
    if cfg.exchange.name != "gate":
        raise SystemExit(f"Only 'gate' is implemented. Got {cfg.exchange.name}.")
    api_key = os.getenv("GATE_API_KEY") if want_live else None
    secret = os.getenv("GATE_API_SECRET") if want_live else None
    return GateFutures(api_key=api_key, secret=secret)


def _build_broker(cfg: Config, exchange: GateFutures, want_live: bool) -> Broker:
    if want_live:
        if cfg.broker.mode != "live":
            raise SystemExit("--live passed but config.broker.mode is not 'live'. Refusing to run.")
        if not (os.getenv("GATE_API_KEY") and os.getenv("GATE_API_SECRET")):
            raise SystemExit("--live requires GATE_API_KEY and GATE_API_SECRET env vars.")
        from .broker.live import LiveGateBroker

        return LiveGateBroker(exchange, cfg.broker, leverage=cfg.risk.leverage)
    return PaperBroker(starting_equity=cfg.risk.account_equity, cfg=cfg.broker)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    _setup_logging(cfg.logging.level)
    exchange = _build_exchange(cfg, want_live=args.live)
    broker = _build_broker(cfg, exchange, want_live=args.live)
    trader = Trader(exchange, broker, cfg)
    result = trader.run(max_iterations=args.iterations)
    console.print_json(data=result)
    return 0


def _cmd_levels(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    _setup_logging(cfg.logging.level)
    exchange = _build_exchange(cfg, want_live=False)
    syms = [args.symbol] if args.symbol else select_universe(exchange, cfg.universe)
    table = Table(title="Detected levels")
    table.add_column("symbol")
    table.add_column("tf")
    table.add_column("price", justify="right")
    table.add_column("kind")
    table.add_column("touches", justify="right")
    table.add_column("score", justify="right")
    table.add_column("htf", justify="center")
    for s in syms:
        htf_rows = exchange.fetch_ohlcv(s, cfg.timeframes.context, limit=cfg.timeframes.lookback_bars)
        ex_rows = exchange.fetch_ohlcv(s, cfg.timeframes.execution, limit=cfg.timeframes.lookback_bars)
        htf_df = ohlcv_to_df(htf_rows)
        ex_df = ohlcv_to_df(ex_rows)
        htf_levels = detect_levels(htf_df, cfg.levels)
        ex_levels = detect_levels(ex_df, cfg.levels, htf_levels=htf_levels)
        for lv in ex_levels[: args.top]:
            table.add_row(
                s, cfg.timeframes.execution, f"{lv.price:.6g}", lv.kind,
                str(lv.touches), f"{lv.score:.2f}", "Y" if lv.htf_confirmed else "",
            )
    console.print(table)
    return 0


def _cmd_universe(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    _setup_logging(cfg.logging.level)
    exchange = _build_exchange(cfg, want_live=False)
    syms = select_universe(exchange, cfg.universe)
    for s in syms:
        console.print(s)
    return 0


def _cmd_polymarket_backtest(args: argparse.Namespace) -> int:
    _setup_logging("INFO")
    cfg = BacktestConfig(
        n_markets=args.n_markets,
        seed=args.seed,
        taker_fee=args.taker_fee,
        slippage=args.slippage,
        starting_bankroll=args.bankroll,
        simulator=SimulatorConfig(steps=args.steps),
        strategy=MeanReversionConfig(
            ewma_halflife=args.ewma_halflife,
            entry_threshold=args.entry_threshold,
            stop_threshold=args.stop_threshold,
            max_hold_ticks=args.max_hold,
            stake_per_trade=args.stake,
        ),
    )
    result = run_backtest(cfg)
    summary = result.summary()

    table = Table(title=f"Polymarket backtest ({summary['n_markets']} markets, seed={cfg.seed})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)

    if args.trade_log:
        path = Path(args.trade_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for trade in result.trades:
                f.write(
                    json.dumps(
                        {
                            "market_id": trade.market_id,
                            "side": trade.side,
                            "entry_tick": trade.entry_tick,
                            "entry_price": trade.entry_price,
                            "exit_tick": trade.exit_tick,
                            "exit_price": trade.exit_price,
                            "exit_reason": trade.exit_reason,
                            "stake": trade.stake,
                            "quantity": trade.quantity,
                            "fees_paid": trade.fees_paid,
                            "pnl": trade.pnl,
                        }
                    )
                    + "\n"
                )
        console.print(f"wrote {len(result.trades)} trades to {path}")

    # Exit non-zero so CI / shell callers can flag regressions.
    return 0 if result.total_pnl > 0 else 1


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.trade_log)
    if not path.exists():
        console.print(f"No trade log at {path}")
        return 1
    trades = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not trades:
        console.print("No trades logged yet.")
        return 0
    from .broker.base import Position
    from .journal import summarize

    positions = [Position(**t) for t in trades]
    summary = summarize(positions)
    console.print_json(data=summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="level-trader", description="Level-trading crypto agent")
    parser.add_argument("--config", default="config.yaml", help="path to config YAML")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the agent (paper by default)")
    p_run.add_argument("--live", action="store_true", help="use live Gate broker (requires API keys)")
    p_run.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="stop after N iterations (useful for a bounded paper session)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_lv = sub.add_parser("levels", help="print detected levels for one or all symbols")
    p_lv.add_argument("--symbol", default=None)
    p_lv.add_argument("--top", type=int, default=8)
    p_lv.set_defaults(func=_cmd_levels)

    p_u = sub.add_parser("universe", help="print the auto-selected universe")
    p_u.set_defaults(func=_cmd_universe)

    p_r = sub.add_parser("report", help="summarize a run from the trade log")
    p_r.add_argument("--trade-log", default="runs/trades.jsonl")
    p_r.set_defaults(func=_cmd_report)

    p_pm = sub.add_parser(
        "polymarket-backtest",
        help="backtest the Polymarket mean-reversion strategy on synthetic markets",
    )
    p_pm.add_argument("--n-markets", type=int, default=500)
    p_pm.add_argument("--seed", type=int, default=7)
    p_pm.add_argument("--steps", type=int, default=200, help="ticks per market")
    p_pm.add_argument("--taker-fee", type=float, default=0.01)
    p_pm.add_argument("--slippage", type=float, default=0.005)
    p_pm.add_argument("--bankroll", type=float, default=10_000.0)
    p_pm.add_argument("--ewma-halflife", type=int, default=20)
    p_pm.add_argument("--entry-threshold", type=float, default=0.05)
    p_pm.add_argument("--stop-threshold", type=float, default=0.12)
    p_pm.add_argument("--max-hold", type=int, default=60)
    p_pm.add_argument("--stake", type=float, default=100.0)
    p_pm.add_argument(
        "--trade-log",
        default="runs/polymarket_trades.jsonl",
        help="where to write per-trade JSONL (pass empty string to skip)",
    )
    p_pm.set_defaults(func=_cmd_polymarket_backtest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
