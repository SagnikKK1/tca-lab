"""tcalab — pre-trade transaction-cost models with reality-checked parameters."""

from tcalab.costs import CostModel, funding_pnl, load_fee_schedule
from tcalab.spreads import abdi_ranaldo_half_spread, tick_bound_half_spread

__version__ = "0.1.0"
__all__ = [
    "CostModel",
    "funding_pnl",
    "load_fee_schedule",
    "tick_bound_half_spread",
    "abdi_ranaldo_half_spread",
]
