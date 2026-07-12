"""CostModel arithmetic: exponents, multiplier, per-row inputs, funding."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from tcalab.costs import CostModel, funding_pnl, load_fee_schedule


def _cost(cm: CostModel, dw=1.0, sigma=0.02, volume=1e9, fee=None, spread=None):
    df = pl.DataFrame({"dw": [dw], "sigma": [sigma], "vol": [volume]})
    fee_expr = pl.lit(fee) if fee is not None else None
    spread_expr = pl.lit(spread) if spread is not None else None
    return df.select(
        cm.trade_cost_return(
            pl.col("dw"), pl.col("sigma"), pl.col("vol"),
            taker_fee_bps=fee_expr, half_spread_bps=spread_expr,
        ).alias("c")
    )["c"][0]


def test_linear_terms():
    cm = CostModel(taker_fee_bps=5.0, half_spread_bps=1.0, impact_coeff=0.0)
    assert abs(_cost(cm) - 6.0 / 1e4) < 1e-15


def test_size_sensitivity_scaling():
    for beta in (0.5, 0.7, 1.0):
        cm1 = CostModel(0.0, 0.0, 1.0, aum_usd=1e5, size_sensitivity=beta)
        cm2 = CostModel(0.0, 0.0, 1.0, aum_usd=2e5, size_sensitivity=beta)
        assert abs(_cost(cm2) / _cost(cm1) - 2**beta) < 1e-9


def test_volume_sensitivity_deep_names_cheaper():
    cm = CostModel(0.0, 0.0, 1.0, volume_sensitivity=0.3)
    # equal participation: scale volume and AUM together
    thin = _cost(CostModel(0.0, 0.0, 1.0, aum_usd=1e5, volume_sensitivity=0.3),
                 volume=1e8)
    deep = _cost(CostModel(0.0, 0.0, 1.0, aum_usd=1e7, volume_sensitivity=0.3),
                 volume=1e10)
    assert deep < thin
    assert abs(deep / thin - (1e10 / 1e8) ** -0.3) < 1e-9
    _ = cm


def test_cost_multiplier_scales_all_terms():
    base = _cost(CostModel(5.0, 1.0, 1.0))
    scaled = _cost(CostModel(5.0, 1.0, 1.0, cost_multiplier=1.5))
    assert abs(scaled / base - 1.5) < 1e-12


def test_per_row_inputs_override_fallbacks():
    cm = CostModel(taker_fee_bps=99.0, half_spread_bps=99.0, impact_coeff=0.0)
    assert abs(_cost(cm, fee=5.0, spread=1.0) - 6.0 / 1e4) < 1e-15


def test_packaged_fee_schedule_loads():
    sched = load_fee_schedule()
    assert sched.height >= 1
    assert sched["taker_bps"][0] == 5.0


def _positions():
    return pl.DataFrame(
        {
            "date": [date(2023, 1, 2)] * 2,
            "symbol": ["LONGY", "SHORTY"],
            "weight": [0.5, -0.5],
            "exec_date": [date(2023, 1, 3)] * 2,
        }
    )


def test_funding_long_pays_short_receives():
    funding = pl.DataFrame(
        {
            "symbol": ["LONGY", "SHORTY"],
            "calc_time": [datetime(2023, 1, 3, 8, tzinfo=UTC)] * 2,
            "funding_interval_hours": [8, 8],
            "last_funding_rate": [0.0001, 0.0001],
        }
    )
    out = funding_pnl(_positions(), funding)
    assert out.filter(pl.col("symbol") == "LONGY")["funding_pnl"][0] == -0.5 * 0.0001
    assert out.filter(pl.col("symbol") == "SHORTY")["funding_pnl"][0] == +0.5 * 0.0001


def test_funding_midnight_convention():
    positions = _positions().filter(pl.col("symbol") == "LONGY").with_columns(
        pl.lit(1.0).alias("weight")
    )
    end_midnight = pl.DataFrame(
        {
            "symbol": ["LONGY"],
            "calc_time": [datetime(2023, 1, 4, 0, tzinfo=UTC)],
            "funding_interval_hours": [8],
            "last_funding_rate": [0.001],
        }
    )
    assert funding_pnl(positions, end_midnight)["funding_pnl"][0] == pytest.approx(-0.001)
    entry_midnight = end_midnight.with_columns(
        pl.lit(datetime(2023, 1, 3, 0, tzinfo=UTC)).alias("calc_time")
    )
    assert funding_pnl(positions, entry_midnight)["funding_pnl"][0] == 0.0
