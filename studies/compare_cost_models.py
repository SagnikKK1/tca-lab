"""Cost-model bake-off: published impact families vs OUR two reality checks.

Models (all as impact(part) with level fitted, exponent per provenance):
  spread_only   impact = 0                          (naive floor)
  kyle_linear   impact = c * sigma * part           (Kyle 1985)
  almgren       impact = c * sigma * part^0.6       (Almgren et al. 2005)
  sqrt_law      impact = c * sigma * part^0.5       (Barra/Toth/CFM)
  power_fit     impact = c * sigma * part^beta      (beta free)
  log_concave   impact = c * sigma * ln(1+part/p0)  (Bouchaud-flavored)

Evaluation 1 — tape: leave-one-symbol-day-out CV on the 4,891 five-minute
windows (reports/impact_calibration_obs.csv): fit levels (and beta for
power_fit) on 16 groups, score OOS R^2 on the held-out group. No model is
allowed to win by memorizing a day.

Evaluation 2 — book walk: median absolute percentage error of
tick/2 + impact(size/V_d) against the real crossing curves
(reports/book_walk_study.csv), sizes <= $1M (inside visible liquidity).

Usage: python scripts/compare_cost_models.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tcalab.panel import daily_frame, load_klines
from tcalab.spreads import tick_bound_half_spread

P0 = 1e-4  # log model scale; ~median tape participation


def features(part: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "kyle_linear": sigma * part,
        "almgren": sigma * part**0.6,
        "sqrt_law": sigma * part**0.5,
        "log_concave": sigma * np.log1p(part / P0),
    }


def fit_level(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / (x @ x)) if (x @ x) > 0 else 0.0


def main() -> int:
    obs = pl.read_csv("reports/impact_calibration_obs.csv")
    obs = obs.with_columns((pl.col("x") / pl.col("part").sqrt()).alias("sigma"))
    groups = obs.select("symbol", "date").unique().sort("symbol", "date")

    # ---- Evaluation 1: tape, leave-one-symbol-day-out --------------------
    names = ["spread_only", "kyle_linear", "almgren", "sqrt_law", "power_fit",
             "log_concave"]
    sse = dict.fromkeys(names, 0.0)
    sst = 0.0
    betas = []
    for g in groups.iter_rows(named=True):
        mask = (pl.col("symbol") == g["symbol"]) & (pl.col("date") == g["date"])
        train, test = obs.filter(~mask), obs.filter(mask)
        ptr, str_ = train["part"].to_numpy(), train["sigma"].to_numpy()
        yte = test["y"].to_numpy()
        pte, ste = test["part"].to_numpy(), test["sigma"].to_numpy()
        ytr = train["y"].to_numpy()
        sst += float(((yte - ytr.mean()) ** 2).sum())

        ftr, fte = features(ptr, str_), features(pte, ste)
        sse["spread_only"] += float((yte**2).sum())
        for name in ("kyle_linear", "almgren", "sqrt_law", "log_concave"):
            c = fit_level(ftr[name], ytr)
            sse[name] += float(((yte - c * fte[name]) ** 2).sum())
        # power_fit: beta via train log-log, then level
        lx = np.log(ptr)
        ly = np.log(np.abs(ytr) / str_ + 1e-12)
        beta = float(np.polyfit(lx, ly, 1)[0])
        betas.append(beta)
        xtr, xte = str_ * ptr**beta, ste * pte**beta
        c = fit_level(xtr, ytr)
        sse["power_fit"] += float(((yte - c * xte) ** 2).sum())

    tape = {n: 1 - sse[n] / sst for n in names}

    # ---- Evaluation 2: book-walk curves ----------------------------------
    walk = pl.read_csv("reports/book_walk_study.csv").with_columns(
        pl.col("date").str.to_date()
    ).filter(pl.col("size_usd") <= 1e6)
    klines = load_klines()
    daily = daily_frame(klines).select("symbol", "date", "sigma", "dollar_volume")
    spreads = tick_bound_half_spread(klines)
    walk = walk.join(daily, on=["symbol", "date"]).join(
        spreads, on=["symbol", "date"]
    ).with_columns((pl.col("size_usd") / pl.col("dollar_volume")).alias("part"))

    p = walk["part"].to_numpy()
    s = walk["sigma"].to_numpy()
    real = walk["walk_cost_bps"].to_numpy()
    half_spread = walk["half_spread_bps"].to_numpy()

    # levels fitted on the FULL tape (frozen before seeing the books)
    pf, sf, yf = obs["part"].to_numpy(), obs["sigma"].to_numpy(), obs["y"].to_numpy()
    ff = features(pf, sf)
    beta_full = float(np.median(betas))
    levels = {n: fit_level(ff[n], yf) for n in ff}
    levels["power_fit"] = fit_level(sf * pf**beta_full, yf)

    walk_mape = {}
    fwalk = features(p, s)
    fwalk["power_fit"] = s * p**beta_full
    for n in names:
        imp = np.zeros_like(p) if n == "spread_only" else levels[n] * fwalk[n]
        pred = half_spread + imp * 1e4
        walk_mape[n] = float(np.median(np.abs(pred - real) / real))

    table = pl.DataFrame(
        {
            "model": names,
            "exponent": ["-", "1.0", "0.6", "0.5", f"{beta_full:.3f} (fit)",
                         "log"],
            "tape_oos_r2": [round(tape[n], 4) for n in names],
            "bookwalk_median_ape": [f"{walk_mape[n]:.0%}" for n in names],
        }
    ).sort("tape_oos_r2", descending=True)
    print(table)
    print(f"\npower_fit beta across CV folds: median {beta_full:.3f}, "
          f"range [{min(betas):.3f}, {max(betas):.3f}]")
    table.write_csv("reports/cost_model_comparison.csv")
    print("-> reports/cost_model_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
