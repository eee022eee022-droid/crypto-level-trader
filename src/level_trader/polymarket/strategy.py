"""Mean-reversion strategy for Polymarket binary markets.

The idea: prediction-market prices wiggle around a latent true probability
with mean-reverting noise that is most violent early in the market's life.
When the current price deviates from its own EWMA by more than a threshold
(that must exceed round-trip fees + slippage), we bet on reversion back to
the EWMA. We take profit when price crosses back through the EWMA and cut
losses if the deviation grows past a stop threshold. Anything still open at
resolution settles to 0 or 1.

Notation:
* ``price`` is the YES-token price in (0, 1).
* Trading "YES" means we buy a YES token at ``price`` that redeems for
  $1 if the market resolves YES, else $0.
* Trading "NO" is symmetric: we buy a NO token at ``1 - price`` that
  redeems for $1 on NO, else $0. PnL on NO is therefore ``entry_price -
  exit_price`` in YES-price terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .simulator import MarketPath

Side = Literal["yes", "no"]


class MeanReversionConfig(BaseModel):
    """Parameters of :class:`MeanReversionStrategy`."""

    ewma_halflife: int = Field(20, ge=1, description="EWMA half-life in ticks.")
    warmup_ticks: int = Field(20, ge=1, description="Ticks required before trading.")
    entry_threshold: float = Field(
        0.05, gt=0.0, description="Price deviation from EWMA required to enter."
    )
    stop_threshold: float = Field(
        0.12, gt=0.0, description="Adverse deviation at which to cut."
    )
    exit_cross: bool = Field(
        True, description="Take profit when price crosses back through EWMA."
    )
    max_hold_ticks: int = Field(
        60, ge=1, description="Time-based exit if price hasn't reverted."
    )
    cooldown_ticks: int = Field(5, ge=0, description="Bars to wait after a closed trade.")
    stake_per_trade: float = Field(
        100.0, gt=0.0, description="USD notional committed per trade."
    )
    min_price: float = Field(0.05, gt=0.0, lt=0.5)
    max_price: float = Field(0.95, gt=0.5, lt=1.0)
    no_trade_band_before_close: int = Field(
        5, ge=0, description="Don't open new trades this close to resolution."
    )


@dataclass
class Trade:
    """A single round-trip trade recorded by the backtester."""

    market_id: str
    side: Side
    entry_tick: int
    entry_price: float
    stake: float
    quantity: float
    exit_tick: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    fees_paid: float = 0.0
    pnl: float = 0.0

    @property
    def closed(self) -> bool:
        return self.exit_tick is not None


def _ewma(prices: np.ndarray, halflife: int) -> np.ndarray:
    """EWMA with given half-life, returning a same-length array."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    out = np.empty_like(prices, dtype=float)
    m = prices[0]
    out[0] = m
    for i in range(1, len(prices)):
        m = alpha * prices[i] + (1.0 - alpha) * m
        out[i] = m
    return out


class MeanReversionStrategy:
    """Deviation-from-EWMA strategy, single-position-per-market."""

    def __init__(self, cfg: MeanReversionConfig) -> None:
        self.cfg = cfg

    def trade_market(
        self,
        market: MarketPath,
        taker_fee: float,
        slippage: float,
    ) -> list[Trade]:
        """Run the strategy on one market path and return closed trades."""
        cfg = self.cfg
        prices = market.prices
        T = len(prices)
        ewma = _ewma(prices, cfg.ewma_halflife)

        trades: list[Trade] = []
        open_trade: Trade | None = None
        cooldown_until = 0

        resolution_tick = T - 1
        last_open_allowed = resolution_tick - cfg.no_trade_band_before_close

        for t in range(T):
            price_raw = float(prices[t])
            mean = float(ewma[t])

            if open_trade is not None:
                # Resolution: settle at outcome.
                if t == resolution_tick:
                    exit_price = float(market.outcome)
                    _close(open_trade, t, exit_price, "resolution", taker_fee, slippage)
                    trades.append(open_trade)
                    open_trade = None
                    continue

                reverted = (
                    open_trade.side == "yes" and price_raw >= mean
                ) or (open_trade.side == "no" and price_raw <= mean)
                if cfg.exit_cross and reverted:
                    _close(open_trade, t, price_raw, "reverted", taker_fee, slippage)
                    trades.append(open_trade)
                    open_trade = None
                    cooldown_until = t + cfg.cooldown_ticks
                    continue

                deviation = price_raw - mean
                adverse = (
                    open_trade.side == "yes" and deviation <= -cfg.stop_threshold
                ) or (open_trade.side == "no" and deviation >= cfg.stop_threshold)
                if adverse:
                    _close(open_trade, t, price_raw, "stop", taker_fee, slippage)
                    trades.append(open_trade)
                    open_trade = None
                    cooldown_until = t + cfg.cooldown_ticks
                    continue

                if t - open_trade.entry_tick >= cfg.max_hold_ticks:
                    _close(open_trade, t, price_raw, "timeout", taker_fee, slippage)
                    trades.append(open_trade)
                    open_trade = None
                    cooldown_until = t + cfg.cooldown_ticks
                    continue
                continue

            # No open position — look for entry.
            if t < cfg.warmup_ticks or t < cooldown_until or t > last_open_allowed:
                continue
            if price_raw < cfg.min_price or price_raw > cfg.max_price:
                continue

            deviation = price_raw - mean
            if deviation >= cfg.entry_threshold:
                open_trade = _open(market.market_id, "no", t, price_raw, cfg, taker_fee, slippage)
            elif deviation <= -cfg.entry_threshold:
                open_trade = _open(market.market_id, "yes", t, price_raw, cfg, taker_fee, slippage)

        # Force-close anything still open using the terminal price.
        if open_trade is not None:
            _close(open_trade, resolution_tick, float(market.outcome), "resolution", taker_fee, slippage)
            trades.append(open_trade)
        return trades


def _fill_price(side: Side, ref: float, slippage: float, is_entry: bool) -> float:
    """Apply slippage: entries cost more, exits fill worse."""
    if is_entry:
        # Buying YES at a higher price (or NO at a lower price, which is a
        # higher 1-p) eats us.
        if side == "yes":
            return min(1.0, ref * (1.0 + slippage))
        return max(0.0, ref * (1.0 - slippage))
    # Exiting a YES position is a sell, so fills at a worse (lower) price.
    if side == "yes":
        return max(0.0, ref * (1.0 - slippage))
    return min(1.0, ref * (1.0 + slippage))


def _open(
    market_id: str,
    side: Side,
    tick: int,
    ref_price: float,
    cfg: MeanReversionConfig,
    taker_fee: float,
    slippage: float,
) -> Trade:
    fill = _fill_price(side, ref_price, slippage, is_entry=True)
    token_price = fill if side == "yes" else (1.0 - fill)
    token_price = max(min(token_price, 0.999), 0.001)
    quantity = cfg.stake_per_trade / token_price
    fee = cfg.stake_per_trade * taker_fee
    return Trade(
        market_id=market_id,
        side=side,
        entry_tick=tick,
        entry_price=fill,
        stake=cfg.stake_per_trade,
        quantity=quantity,
        fees_paid=fee,
    )


def _close(
    trade: Trade,
    tick: int,
    ref_price: float,
    reason: str,
    taker_fee: float,
    slippage: float,
) -> None:
    fill = _fill_price(trade.side, ref_price, slippage, is_entry=False)
    if trade.side == "yes":
        token_exit = fill
    else:
        token_exit = 1.0 - fill
    proceeds = trade.quantity * token_exit
    exit_fee = proceeds * taker_fee
    pnl = proceeds - trade.stake - exit_fee - trade.fees_paid
    trade.exit_tick = tick
    trade.exit_price = fill
    trade.exit_reason = reason
    trade.fees_paid += exit_fee
    trade.pnl = pnl
