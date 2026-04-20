import numpy as np

from level_trader.polymarket.simulator import SimulatorConfig, simulate_market, simulate_markets


def test_market_prices_stay_in_band_and_end_at_outcome():
    cfg = SimulatorConfig()
    rng = np.random.default_rng(42)
    m = simulate_market(cfg, rng)
    assert len(m.prices) == cfg.steps
    assert (m.prices >= cfg.price_floor).all()
    assert (m.prices <= cfg.price_ceiling).all() or m.prices[-1] in (0.0, 1.0)
    assert m.outcome in (0, 1)
    assert m.prices[-1] == float(m.outcome)


def test_seed_is_deterministic():
    cfg = SimulatorConfig()
    a = simulate_markets(cfg, n_markets=20, seed=123)
    b = simulate_markets(cfg, n_markets=20, seed=123)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.prices, y.prices)
        assert x.outcome == y.outcome
        assert x.p_true == y.p_true


def test_outcome_is_calibrated_to_p_true_across_many_markets():
    cfg = SimulatorConfig(p_true_alpha=1.0, p_true_beta=1.0)
    markets = simulate_markets(cfg, n_markets=2000, seed=0)
    avg_p = float(np.mean([m.p_true for m in markets]))
    avg_outcome = float(np.mean([m.outcome for m in markets]))
    # Beta(1,1) is uniform so avg p_true ≈ 0.5 and avg outcome ≈ 0.5.
    assert abs(avg_p - 0.5) < 0.03
    assert abs(avg_outcome - 0.5) < 0.04
