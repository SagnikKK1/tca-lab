"""Spread estimators: tick-bound (primary, bookTicker-validated) and
Abdi-Ranaldo (stress diagnostic)."""

from datetime import timedelta

import numpy as np
import polars as pl

from tcalab.spreads import abdi_ranaldo_half_spread, tick_bound_half_spread
from tests.conftest import make_klines


def _bounce_panel(symbol, t0, n_days, spread_frac, daily_vol=0.02, seed=5):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, daily_vol / np.sqrt(24), n)))
    side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    half = spread_frac / 2.0
    kl = make_klines(symbol, t0, n)
    return kl.with_columns(
        pl.Series("close", mid * (1 + side * half)),
        pl.Series("high", mid * (1 + half)),
        pl.Series("low", mid * (1 - half)),
    )


def _tick_grid_panel(symbol, t0, n_days, tick, price0, seed=9):
    rng = np.random.default_rng(seed)
    n = n_days * 24
    steps = rng.choice([-2, -1, -1, 0, 1, 1, 2], size=n)
    price = price0 + np.cumsum(steps) * tick
    return make_klines(symbol, t0, n).with_columns(pl.Series("close", price))


class TestTickBound:
    def test_matches_structural_truth(self, t0):
        panel = _tick_grid_panel("XTZUSDT", t0, 60, tick=0.001, price0=0.65)
        est = tick_bound_half_spread(panel)
        last_close = panel.sort("open_time")["close"][-1]
        expected = 0.001 / (2 * last_close) * 1e4
        got = est.sort("date")["half_spread_bps"][-1]
        assert abs(got - expected) / expected < 0.01

    def test_point_in_time(self, t0):
        panel = _tick_grid_panel("XTZUSDT", t0, 40, tick=0.001, price0=0.65)
        cutoff = t0 + timedelta(days=25)
        full = tick_bound_half_spread(panel)
        trunc = tick_bound_half_spread(panel.filter(pl.col("open_time") <= cutoff))
        joined = full.filter(pl.col("date") <= cutoff.date() - timedelta(days=1)).join(
            trunc, on=["symbol", "date"], suffix="_t"
        )
        assert (joined["half_spread_bps"] - joined["half_spread_bps_t"]).abs().max() < 1e-12

    def test_adapts_to_retiering(self, t0):
        fine = _tick_grid_panel("AAAUSDT", t0, 40, tick=0.001, price0=0.65, seed=1)
        coarse = _tick_grid_panel(
            "AAAUSDT", t0 + timedelta(days=40), 40, tick=0.01, price0=0.65, seed=2
        )
        est = tick_bound_half_spread(pl.concat([fine, coarse]))
        early = est.filter(pl.col("date") == (t0 + timedelta(days=30)).date())
        late = est.filter(pl.col("date") == (t0 + timedelta(days=79)).date())
        assert late["half_spread_bps"][0] > 5 * early["half_spread_bps"][0]


class TestAbdiRanaldo:
    def test_recovers_known_spread(self, t0):
        panel = _bounce_panel("AAAUSDT", t0, 120, spread_frac=0.004)
        est = abdi_ranaldo_half_spread(panel, floor_bps=0.0, cap_bps=1000.0)
        assert 15.0 <= est["half_spread_bps"].median() <= 25.0

    def test_resolves_single_digit_bps(self, t0):
        panel = _bounce_panel("BTCUSDT", t0, 120, spread_frac=0.0002)
        est = abdi_ranaldo_half_spread(panel, floor_bps=0.0, cap_bps=1000.0)
        assert est["half_spread_bps"].median() < 3.0

    def test_point_in_time(self, t0):
        panel = _bounce_panel("AAAUSDT", t0, 100, spread_frac=0.004)
        cutoff = t0 + timedelta(days=60)
        full = abdi_ranaldo_half_spread(panel)
        trunc = abdi_ranaldo_half_spread(panel.filter(pl.col("open_time") <= cutoff))
        joined = full.filter(pl.col("date") <= cutoff.date() - timedelta(days=1)).join(
            trunc, on=["symbol", "date"], suffix="_t"
        )
        assert (joined["half_spread_bps"] - joined["half_spread_bps_t"]).abs().max() < 1e-12
