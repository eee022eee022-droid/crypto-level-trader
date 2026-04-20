import numpy as np

from level_trader.polymarket.simulator import MarketPath
from level_trader.polymarket.strategy import MeanReversionConfig, MeanReversionStrategy


def _constant_path(p: float, steps: int, outcome: int) -> MarketPath:
    prices = np.full(steps, p, dtype=float)
    prices[-1] = float(outcome)
    return MarketPath(market_id="t", prices=prices, outcome=outcome, p_true=p)


def test_strategy_does_not_trade_flat_market():
    strat = MeanReversionStrategy(MeanReversionConfig())
    market = _constant_path(0.5, steps=200, outcome=1)
    trades = strat.trade_market(market, taker_fee=0.0, slippage=0.0)
    # Only possible "trade" is one that resolves immediately — with a
    # perfectly flat path there is no deviation from EWMA so no entries.
    assert trades == []


def test_strategy_takes_profit_when_price_reverts():
    cfg = MeanReversionConfig(
        ewma_halflife=10,
        warmup_ticks=10,
        entry_threshold=0.05,
        stop_threshold=0.5,  # disable stops for this test
        cooldown_ticks=0,
        stake_per_trade=100.0,
        no_trade_band_before_close=1,
    )
    # 30 ticks of 0.5, then a dip to 0.4 for a few ticks, then back to 0.5,
    # then resolution at 1 (YES wins). We expect to buy YES on the dip and
    # take profit on the revert back to EWMA.
    prices = np.concatenate([
        np.full(30, 0.5),
        np.full(5, 0.40),
        np.full(60, 0.5),
    ])
    prices = np.append(prices, 1.0)
    market = MarketPath(market_id="rev", prices=prices, outcome=1, p_true=0.5)

    strat = MeanReversionStrategy(cfg)
    trades = strat.trade_market(market, taker_fee=0.0, slippage=0.0)
    assert len(trades) >= 1
    first = trades[0]
    assert first.side == "yes"
    assert first.exit_reason == "reverted"
    assert first.pnl > 0


def test_strategy_stops_out_on_adverse_move():
    cfg = MeanReversionConfig(
        ewma_halflife=5,
        warmup_ticks=5,
        entry_threshold=0.05,
        stop_threshold=0.10,
        cooldown_ticks=0,
        stake_per_trade=100.0,
        no_trade_band_before_close=1,
    )
    prices = np.concatenate([
        np.full(20, 0.5),
        np.full(1, 0.40),
        np.linspace(0.40, 0.20, 30),
    ])
    prices = np.append(prices, 0.0)  # NO wins — YES position gets crushed
    market = MarketPath(market_id="stop", prices=prices, outcome=0, p_true=0.5)

    strat = MeanReversionStrategy(cfg)
    trades = strat.trade_market(market, taker_fee=0.0, slippage=0.0)
    assert len(trades) >= 1
    assert trades[0].side == "yes"
    assert trades[0].exit_reason in ("stop", "resolution", "timeout")
    assert trades[0].pnl < 0
