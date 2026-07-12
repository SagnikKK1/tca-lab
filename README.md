# tca-lab

Pre-trade transaction-cost models for crypto perpetuals where **every
parameter is sourced, calibrated, or reconciled against real market data** —
built without client fill data, with the identification limits of public
data stated rather than hidden.

```
cost = fees + spread/2 + α · σ · (Q/V)^β · (V/V_ref)^(−κ)
```

## Every term, reality-checked

| term | provenance | reality check | evidence |
|---|---|---|---|
| fees | dated registry, official schedule | deterministic | `src/tcalab/fees.csv` |
| spread/2 | structural: `tick/(2·price)`, tick inferred PIT from data | **1.6% median error, rank corr 1.000** vs real quoted spreads (tick-level bookTicker history) | `reports/spread_validation.csv` |
| β (size exponent) | fitted on tape | **0.45, 95% CI [0.31, 0.59]** — brackets the √-law's 0.5 | `studies/calibrate_impact.py` |
| α (impact level) | tape regression, 4,891 five-minute flow windows | **bounded [0.5 lit, 1.64 tape]**; band brackets real book-crossing costs at child sizes | `reports/impact_calibration_obs.csv` |
| model validity edge | 500-level order-book reconstruction | breakdown located at ~$1–5M single-crossing in mid-caps | `reports/book_walk_study.csv` |

## Findings that came from the data, not the literature

1. **Spreads on this venue are structural, not statistical.** Every sampled
   real quote's median spread equals exactly one tick (BTC 0.019 bps …
   CRV 12.0 bps, all `tick/(2·price)` to ~3 decimals). The statistical
   estimator we built first (Abdi-Ranaldo) failed its reconciliation —
   89% median error, vol-noise floor overcharging majors ~100× — and is
   retained as a stress diagnostic, not the charged cost.
2. **The √-law's exponent emerges from the tape.** A free-exponent fit gives
   β = 0.45 [0.31, 0.59] (CV-stable across days: 0.40–0.54); linearity
   (Kyle, β=1) is clearly rejected out-of-sample, but exponents in
   [0.4, 0.6] (Almgren's 0.6, √-law's 0.5) are indistinguishable.
3. **α from public flow is an upper bound, and says so.** Contemporaneous
   flow includes feedback trading, so the tape α (1.64 [1.56, 1.72]) sits
   above causal client-fill estimates (0.5–1.0). Capacity work therefore
   runs over the band [0.5, 1.64], not a point.
4. **Two "real costs" exist, and models must declare which they price.**
   Tape-calibrated concave models price a *worked metaorder*; walking a
   real 500-level book prices *instant crossing*. The model bake-off
   (`reports/cost_model_comparison.csv`) shows the ranking flips between
   the two references — a cost model without a stated object of prediction
   is not a model.

## Governance

Impact hyperparameters (α, β, κ) are calibrated from market data or swept as
scenario axes with their CIs. They are **never tuned jointly with a
strategy** — that would launder data-mining through the cost model. The
`cost_multiplier` field is the "costs ±50%" sensitivity axis.

## Layout

```
src/tcalab/        CostModel, spread models, funding accrual, fee registry,
                   panel schema contract, checksummed downloader
studies/           validate_spread_model, calibrate_impact, book_walk_study,
                   compare_cost_models — each regenerates its reports/ CSV
reports/           the evidence the README cites
tests/             estimator recovery, PIT truncation, exponent scaling,
                   funding-by-leg conventions
```

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"
pytest                                # unit tests, no network
TCALAB_PANEL="path/to/klines/*.parquet" python studies/compare_cost_models.py
```

Studies consume hourly OHLCV parquet (schema in `src/tcalab/panel.py`);
data files are not part of this repo.

## Used by

[quant-alpha-lab](https://github.com/SagnikKK1/quant-alpha-lab) — a
cross-sectional ML alpha platform whose backtests charge these models
(vendored there until this package is public/installable; this repo is
canonical for all cost-model code and studies).
