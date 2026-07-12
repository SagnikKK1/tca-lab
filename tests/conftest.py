"""Synthetic hourly-klines fixtures matching the tcalab panel schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest


def make_klines(
    symbol: str,
    start: datetime,
    n_bars: int,
    volume: float = 100.0,
    count: int = 50,
    price: float = 10.0,
) -> pl.DataFrame:
    times = [start + timedelta(hours=i) for i in range(n_bars)]
    return pl.DataFrame(
        {
            "symbol": [symbol] * n_bars,
            "open_time": times,
            "open": [price] * n_bars,
            "high": [price * 1.01] * n_bars,
            "low": [price * 0.99] * n_bars,
            "close": [price] * n_bars,
            "volume": [volume] * n_bars,
            "quote_volume": [volume * price] * n_bars,
            "count": [count] * n_bars,
        }
    ).with_columns(pl.col("open_time").dt.replace_time_zone("UTC"))


@pytest.fixture
def t0() -> datetime:
    return datetime(2023, 1, 1, tzinfo=UTC)
