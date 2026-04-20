from level_trader.polymarket.backtest import BacktestConfig, run_backtest
from level_trader.polymarket.simulator import SimulatorConfig
from level_trader.polymarket.strategy import MeanReversionConfig


def test_backtest_is_profitable_on_default_market_regime():
    cfg = BacktestConfig(n_markets=500, seed=7)
    result = run_backtest(cfg)
    summary = result.summary()
    assert summary["n_trades"] > 100, summary
    assert summary["total_pnl"] > 0, summary
    assert summary["profit_factor"] > 1.1, summary
    assert summary["winrate"] > 0.5, summary


def test_backtest_profit_is_robust_across_seeds():
    pnls = []
    for seed in range(5):
        res = run_backtest(BacktestConfig(n_markets=300, seed=seed))
        pnls.append(res.total_pnl)
    # At least 4 of 5 seeds should be net-positive.
    positives = sum(1 for p in pnls if p > 0)
    assert positives >= 4, (pnls, positives)
    assert sum(pnls) / len(pnls) > 0, pnls


def test_backtest_loses_when_edge_is_killed_by_huge_fees():
    """Sanity check: the engine is not a money-printer. If fees are absurd the
    strategy should stop being profitable — otherwise there is a bug that
    fakes PnL without respecting transaction costs."""
    cfg = BacktestConfig(
        n_markets=300,
        seed=7,
        taker_fee=0.5,  # 50% fee — completely eats the edge
        slippage=0.05,
        simulator=SimulatorConfig(),
        strategy=MeanReversionConfig(),
    )
    result = run_backtest(cfg)
    assert result.total_pnl < 0
