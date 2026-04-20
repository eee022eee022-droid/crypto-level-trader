"""Polymarket live-data adapter.

Pulls real historical market data from Polymarket's public HTTP APIs so the
same backtest engine that runs on the synthetic simulator can run on real
resolved markets. Everything is read-only: no orders are ever placed. The
strategy still trades virtual dollars against real price paths -- "demo
money on real markets".

Endpoints used (both unauthenticated and free):
* ``https://gamma-api.polymarket.com/markets`` — market discovery.
* ``https://clob.polymarket.com/prices-history`` — minute-level price history
  for a given CLOB token id.

The Polymarket prices-history endpoint only returns data within a finite
retention window and requires an explicit ``startTs``/``endTs`` pair. Markets
whose history is no longer retained (or that never traded) are skipped.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

import numpy as np

from .simulator import MarketPath

log = logging.getLogger(__name__)

_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
_CLOB_URL = "https://clob.polymarket.com/prices-history"
_USER_AGENT = "level-trader-polymarket-adapter/0.1"
_HTTP_TIMEOUT = 20.0


@dataclass(frozen=True)
class LiveFetchConfig:
    """Parameters for :func:`fetch_resolved_markets`."""

    limit: int = 50
    """Target number of resolved markets to return (with price history)."""

    min_volume: float = 10_000.0
    """Minimum market volume (USD) — filters tiny / dead markets."""

    min_duration_hours: float = 24.0
    """Minimum market lifetime from startDate to endDate, in hours."""

    max_age_days: int = 60
    """Skip markets that closed more than this many days ago (no price data)."""

    fidelity_minutes: int = 60
    """Price-history resolution requested from CLOB (``fidelity`` query arg)."""

    max_candidates: int = 2000
    """Upper bound on gamma-api rows scanned before giving up."""

    page_size: int = 200
    """Markets per gamma-api page."""


def _http_get(url: str, params: dict | None = None) -> dict | list:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _parse_iso(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(_dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _fetch_gamma_page(offset: int, page_size: int) -> list[dict]:
    params = {
        "closed": "true",
        "limit": str(page_size),
        "order": "closedTime",
        "ascending": "false",
        "offset": str(offset),
    }
    data = _http_get(_GAMMA_URL, params)
    if isinstance(data, list):
        return data
    return []


def _fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity_minutes: int,
) -> list[dict]:
    params = {
        "market": token_id,
        "startTs": str(start_ts),
        "endTs": str(end_ts),
        "fidelity": str(fidelity_minutes),
    }
    try:
        data = _http_get(_CLOB_URL, params)
    except urllib.error.HTTPError as e:
        log.debug("clob prices-history %s -> HTTP %s", token_id[:12], e.code)
        return []
    except (urllib.error.URLError, TimeoutError) as e:
        log.debug("clob prices-history %s -> %s", token_id[:12], e)
        return []
    if isinstance(data, dict):
        return list(data.get("history") or [])
    return []


def _to_market_path(market: dict, history: list[dict]) -> MarketPath | None:
    """Convert a gamma-api market + CLOB history to a :class:`MarketPath`.

    The CLOB history is the price of one of the two outcome tokens in (0, 1);
    we treat ``clobTokenIds[0]`` as the YES token. If the market resolved YES
    (``outcomePrices[0] == "1"``) the outcome is 1, else 0.
    """
    outcome_prices_raw = market.get("outcomePrices") or "[]"
    try:
        outcome_prices = json.loads(outcome_prices_raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not outcome_prices:
        return None
    outcome = 1 if str(outcome_prices[0]) == "1" else 0

    prices = np.array([float(h["p"]) for h in history], dtype=float)
    if prices.size < 30:
        return None

    # Polymarket price feeds already live in (0, 1). Clip defensively and pin
    # the final tick to the resolved outcome so settle PnL is exact.
    prices = np.clip(prices, 1e-4, 1.0 - 1e-4)
    prices = np.append(prices, float(outcome))

    return MarketPath(
        market_id=str(market.get("id")),
        prices=prices,
        outcome=outcome,
        p_true=float(outcome),
    )


def fetch_resolved_markets(cfg: LiveFetchConfig | None = None) -> list[MarketPath]:
    """Fetch up to ``cfg.limit`` resolved Polymarket markets with real price paths.

    Markets without retained price history or too little activity are
    silently skipped. The returned list may be shorter than ``cfg.limit`` if
    the filters are too strict for the retention window.
    """
    cfg = cfg or LiveFetchConfig()
    now = int(time.time())

    out: list[MarketPath] = []
    scanned = 0
    offset = 0
    while len(out) < cfg.limit and scanned < cfg.max_candidates:
        page = _fetch_gamma_page(offset, cfg.page_size)
        if not page:
            break
        for m in page:
            scanned += 1
            if scanned >= cfg.max_candidates:
                break
            end_ts = _parse_iso(m.get("endDate"))
            start_ts = _parse_iso(m.get("startDate"))
            if end_ts is None or start_ts is None:
                continue
            if end_ts > now:
                continue
            if now - end_ts > cfg.max_age_days * 86400:
                continue
            if (end_ts - start_ts) < cfg.min_duration_hours * 3600:
                continue
            try:
                volume = float(m.get("volumeNum") or m.get("volume") or 0.0)
            except (TypeError, ValueError):
                volume = 0.0
            if volume < cfg.min_volume:
                continue
            tokens_raw = m.get("clobTokenIds") or "[]"
            try:
                tokens = json.loads(tokens_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not tokens:
                continue

            history = _fetch_price_history(
                tokens[0],
                start_ts=start_ts,
                end_ts=min(end_ts, now),
                fidelity_minutes=cfg.fidelity_minutes,
            )
            mp = _to_market_path(m, history)
            if mp is None:
                continue
            out.append(mp)
            if len(out) >= cfg.limit:
                break
        offset += cfg.page_size

    return out
