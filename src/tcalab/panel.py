"""Panel access for studies — a schema contract, not a data layer.

tcalab does not own market-data infrastructure. Studies consume hourly OHLCV
parquet files with this schema (the contract any provider must meet):

    symbol: str, open_time: datetime[UTC], open/high/low/close: f64,
    volume: f64, quote_volume: f64, count: i64

The default glob points at a sibling quant-alpha-lab checkout's curated
store; override with TCALAB_PANEL or the `glob` argument.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

DEFAULT_GLOB = str(
    Path(__file__).resolve().parents[3].parent
    / "quant-alpha-lab" / "data" / "curated" / "klines" / "*.parquet"
)


def load_klines(glob: str | None = None) -> pl.DataFrame:
    glob = glob or os.environ.get("TCALAB_PANEL", DEFAULT_GLOB)
    return pl.scan_parquet(glob).collect()


def daily_frame(klines: pl.DataFrame) -> pl.DataFrame:
    """Per (symbol, date): realized daily vol and dollar volume — the two
    PIT quantities the cost models consume."""
    return (
        klines.sort("symbol", "open_time")
        .with_columns(
            pl.col("open_time").dt.date().alias("date"),
            (pl.col("close") / pl.col("close").shift(1)).log().over("symbol").alias("lr"),
        )
        .group_by("symbol", "date", maintain_order=True)
        .agg(
            (pl.col("lr").std() * (24**0.5)).alias("sigma"),
            pl.col("quote_volume").sum().alias("dollar_volume"),
        )
    )
