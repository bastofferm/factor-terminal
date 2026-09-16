"""Produce ex-ante risk forecasts and score them against what actually happened.

The lag discipline is the whole point and is enforced structurally rather than by
convention: for a forecast dated t, the loadings come from a window ending at or
before t, and the factor covariance is estimated on returns up to t. The realised
volatility it is compared against is measured over t+1 .. t+h.

    python -m backend.pipeline.run_risk --instrument US:AAPL --spec <spec_id>
                                        [--horizon 21] [--cov-method blend]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any, Sequence

import numpy as np
import pandas as pd

from backend.core import covariance as cv
from backend.core import risk as rk
from backend.pipeline.dbsync import (
    bulk_insert, connect, etl_run, mark_item_done, prune_items, run_failed,
)
from backend.pipeline.run_estimation import load_factor_panel, load_instrument

JOB = "run_risk"

# Observations used to estimate the factor covariance at each as-of date. Long
# enough that a 40-factor matrix is estimable, short enough to be a current view.
COV_WINDOW = 504

# Cross-sectional shrinkage target for specific risk, and the floor below which an
# instrument's idiosyncratic volatility is not believed.
SPECIFIC_SHRINK = 0.25
SPECIFIC_FLOOR_ANN = 0.02

# Conditioning limits above which a window's betas are not trusted for forecasting.
# Across this universe the median condition number is 54 and the 90th percentile
# 100; only 2002-03, when few factors yet existed, runs into the hundreds. These
# thresholds exclude roughly 3% of windows, almost all of them there.
MAX_CONDITION_NUMBER = 200.0
MAX_VIF = 100.0


def latest_spec(cur: Any) -> str | None:
    cur.execute("SELECT spec_id FROM dim_model_spec ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None


def spec_is_orthogonalized(cur: Any, spec_id: str) -> bool:
    """Which factor panel this spec's betas were estimated on.

    Load-bearing. Portfolio risk is beta' Sigma beta, so Sigma has to be the
    covariance of the same series the betas refer to. Reading the orthogonalised
    panel for a spec estimated on raw factors silently pairs betas with the wrong
    covariance and every predicted volatility is wrong.
    """
    cur.execute("SELECT orthogonalized FROM dim_model_spec WHERE spec_id = %s",
                (spec_id,))
    row = cur.fetchone()
    return True if row is None else bool(row[0])


def load_loadings(cur: Any, spec_id: str, instrument_id: str) -> pd.DataFrame:
    """Loadings in wide form, indexed by window_end."""
    cur.execute(
        "SELECT window_end, factor_id, beta FROM fact_loading "
        "WHERE spec_id = %s AND instrument_id = %s",
        (spec_id, instrument_id),
    )
    df = pd.DataFrame(cur.fetchall(), columns=["window_end", "factor_id", "beta"])
    if df.empty:
        return df
    return df.pivot(index="window_end", columns="factor_id", values="beta").sort_index()


def load_window_meta(cur: Any, spec_id: str, instrument_id: str) -> pd.DataFrame:
    """Residual volatility plus the conditioning diagnostics for each window."""
    cur.execute(
        "SELECT window_end, resid_vol_ann, condition_number, max_vif "
        "FROM fact_regression_meta WHERE spec_id = %s AND instrument_id = %s "
        "ORDER BY window_end",
        (spec_id, instrument_id),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["resid_vol", "cond", "vif"])
    df = pd.DataFrame(rows, columns=["window_end", "resid_vol", "cond", "vif"])
    return df.set_index("window_end").astype(float)


def reliability(cond: float, vif: float,
                orthogonalized: bool = True) -> tuple[bool, str | None]:
    """Is this window's loading vector fit to forecast with?

    With 40 factors on a 252-day window, a period where only a handful of factors
    yet exist leaves the design near-singular. The betas then explode in offsetting
    pairs and beta' Sigma beta with them - AAPL produced a 360% predicted volatility
    that way. Model uncertainty is penalised rather than ignored, so such windows
    are flagged and kept out of the scoring.

    The VIF limit applies only on the orthogonalised panel, and the asymmetry is not
    a convenience. On the raw panel high VIFs are what the factor set *is*: eq_us
    regressed on the other thirty-nine raw factors has an R-squared near 0.9998,
    because eq_global is one of them. Measured on US:AAPL, 236 of 249 raw-panel
    windows exceed a VIF of 100 with a mean max VIF of 866 - and their forecasts are
    not bad ones (mean predicted 0.34 against mean realised 0.28, maximum 0.68, a
    long way from the 3.6 that motivated the gate). Applying the orthogonalised
    threshold there would reject the panel by construction rather than on evidence,
    and "the model can be estimated on raw factors" would be true in name only.

    The condition number is the invariant that actually governs how far beta' Sigma
    beta can blow up, so it gates both panels unchanged. VIF keeps the role its own
    docstring gives it - saying *which* factor is responsible - and is recorded on
    every forecast either way. Same treatment as ARCH-LM in the stationarity
    battery: recorded, not gated, where what it detects is a property of the design
    rather than a defect in it.
    """
    problems = []
    if np.isfinite(cond) and cond > MAX_CONDITION_NUMBER:
        problems.append(f"condition number {cond:.0f} above {MAX_CONDITION_NUMBER:.0f}")
    if orthogonalized and np.isfinite(vif) and vif > MAX_VIF:
        problems.append(f"max VIF {vif:.0f} above {MAX_VIF:.0f}")
    return (not problems), ("; ".join(problems) or None)


def peer_specific_variance(cur: Any, spec_id: str) -> float:
    """Median residual variance across every instrument on this spec.

    Shrinking toward it stops a single short or unusually quiet history producing a
    falsely precise idiosyncratic risk.
    """
    cur.execute(
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY resid_vol_ann) "
        "FROM fact_regression_meta WHERE spec_id = %s AND resid_vol_ann IS NOT NULL",
        (spec_id,),
    )
    row = cur.fetchone()
    return float(row[0]) ** 2 if row and row[0] else np.nan


# ---------------------------------------------------------------------------

def forecast(
    y: pd.Series,
    factors: pd.DataFrame,
    loadings: pd.DataFrame,
    window_meta: pd.DataFrame,
    horizon: int,
    cov_method: str = "blend",
    peer_var: float = np.nan,
    orthogonalized: bool = True,
) -> pd.DataFrame:
    """One ex-ante forecast per loading window, scored against the forward outcome.

    `orthogonalized` describes the panel `factors` and `loadings` both come from; it
    reaches only the reliability gate, whose VIF limit means something different on
    each panel. See `reliability`.
    """
    if loadings.empty:
        return pd.DataFrame()

    factor_names = list(loadings.columns)
    F = factors.reindex(y.index)[factor_names]
    realized = pd.Series(rk.forward_realized(y.to_numpy(), horizon), index=y.index)

    rows: list[dict] = []
    for as_of in loadings.index:
        if as_of not in F.index:
            continue
        pos = F.index.get_loc(as_of)

        # Covariance from returns up to and including as_of. The loading window also
        # ends at as_of, so nothing after as_of enters the forecast.
        lo = max(0, pos - COV_WINDOW + 1)
        window = F.iloc[lo : pos + 1]
        usable = [f for f in factor_names if window[f].notna().mean() >= 0.9]
        if len(usable) < 2:
            continue

        R = window[usable].dropna().to_numpy()
        if R.shape[0] < max(60, len(usable) + 5):
            continue

        cov_res = cv.estimate(R, usable, method=cov_method, halflife=60.0)
        beta = loadings.loc[as_of, usable].to_numpy(dtype=float)

        if as_of not in window_meta.index:
            continue
        meta = window_meta.loc[as_of]
        raw_spec = float(meta["resid_vol"])
        if not np.isfinite(raw_spec):
            continue
        ok, reason = reliability(float(meta["cond"]), float(meta["vif"]),
                                 orthogonalized)
        spec_var = raw_spec**2
        if np.isfinite(peer_var):
            spec_var = (1 - SPECIFIC_SHRINK) * spec_var + SPECIFIC_SHRINK * peer_var
        floored = spec_var < SPECIFIC_FLOOR_ANN**2
        spec_var = max(spec_var, SPECIFIC_FLOOR_ANN**2)

        total, systematic, specific = cv.predicted_volatility(beta, cov_res.cov, spec_var)
        if not np.isfinite(total) or total <= 0:
            continue

        rows.append({
            "as_of_date": as_of,
            "sigma_pred_ann": total,
            "sigma_factor_ann": systematic,
            "sigma_specific_ann": specific,
            "factor_risk_share": (systematic**2 / total**2) if total > 0 else np.nan,
            "sigma_realized_ann": float(realized.get(as_of, np.nan)),
            "var95_pred": float(rk.value_at_risk(np.array([total]), 0.95)[0]),
            "var99_pred": float(rk.value_at_risk(np.array([total]), 0.99)[0]),
            "es97_5_pred": float(rk.expected_shortfall(np.array([total]), 0.975)[0]),
            "floored": floored,
            "condition_number": float(meta["cond"]),
            "max_vif": float(meta["vif"]),
            "is_reliable": ok,
            "unreliable_reason": reason,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["bias_ratio"] = df["sigma_realized_ann"] / df["sigma_pred_ann"]
    return df


def reliable_only(fc: pd.DataFrame) -> pd.DataFrame:
    return fc[fc["is_reliable"]] if "is_reliable" in fc.columns else fc


def daily_predicted_series(y: pd.Series, fc: pd.DataFrame) -> pd.Series:
    """Carry each forecast forward until the next one.

    Between roll-forward dates the current forecast is what was actually in force,
    so this is the series a VaR backtest must use. Forward-filling is legitimate
    here precisely because the value does not change until the model is re-run;
    the shift(1) ensures day t uses the forecast that existed before t.
    """
    s = pd.Series(np.nan, index=y.index, dtype=float)
    s.loc[fc["as_of_date"]] = fc["sigma_pred_ann"].to_numpy()
    return s.ffill().shift(1)


def backtest(y: pd.Series, fc: pd.DataFrame, horizon: int) -> dict:
    """Score the forecasts: bias, Mincer-Zarnowitz, and VaR coverage at 95% and 99%."""
    usable = reliable_only(fc)
    scored = usable.dropna(subset=["sigma_realized_ann"])
    daily_pred = daily_predicted_series(y, usable)
    aligned = pd.DataFrame({"r": y, "sigma": daily_pred}).dropna()

    bias = rk.bias_statistics(
        scored["sigma_pred_ann"].to_numpy(),
        scored["sigma_realized_ann"].to_numpy(),
        returns=None,
    )
    z = rk.standardized_returns(aligned["r"].to_numpy(), aligned["sigma"].to_numpy())
    z = z[np.isfinite(z)]

    mz = rk.mincer_zarnowitz(scored["sigma_pred_ann"].to_numpy(),
                             scored["sigma_realized_ann"].to_numpy())
    c95 = rk.coverage_test(aligned["r"].to_numpy(), aligned["sigma"].to_numpy(), 0.95)
    c99 = rk.coverage_test(aligned["r"].to_numpy(), aligned["sigma"].to_numpy(), 0.99)

    from scipy import stats as _st
    return {
        "n_forecasts": int(len(scored)),
        "n_excluded": int(len(fc) - len(usable)),
        "mean_bias": bias.mean_bias,
        "median_bias": bias.median_bias,
        "z_std": float(np.std(z, ddof=1)) if z.size > 2 else np.nan,
        "z_kurtosis": float(_st.kurtosis(z)) if z.size > 2 else np.nan,
        "mz_alpha": mz.alpha, "mz_beta": mz.beta,
        "mz_alpha_p": mz.alpha_p, "mz_beta_p": mz.beta_p,
        "mz_joint_p": mz.joint_p, "mz_r2": mz.r2,
        "exceptions_95": c95.exceptions, "expected_95": c95.expected,
        "kupiec_stat_95": c95.kupiec_stat, "kupiec_p_95": c95.kupiec_p,
        "christoffersen_stat_95": c95.christoffersen_stat,
        "christoffersen_p_95": c95.christoffersen_p,
        "exceptions_99": c99.exceptions, "expected_99": c99.expected,
        "kupiec_stat_99": c99.kupiec_stat, "kupiec_p_99": c99.kupiec_p,
        "christoffersen_stat_99": c99.christoffersen_stat,
        "christoffersen_p_99": c99.christoffersen_p,
        "sample_start": aligned.index.min() if len(aligned) else None,
        "sample_end": aligned.index.max() if len(aligned) else None,
    }


# ---------------------------------------------------------------------------

def _f(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def write(cur: Any, spec_id: str, instrument_id: str, horizon: int, cov_method: str,
          fc: pd.DataFrame, bt: dict) -> int:
    cur.execute(
        "DELETE FROM fact_risk_forecast WHERE spec_id=%s AND instrument_id=%s "
        "AND horizon_days=%s AND cov_method=%s",
        (spec_id, instrument_id, horizon, cov_method))
    if fc.empty:
        return 0

    bulk_insert(cur, """
        INSERT INTO fact_risk_forecast
            (spec_id, instrument_id, as_of_date, horizon_days, cov_method,
             sigma_pred_ann, sigma_factor_ann, sigma_specific_ann, factor_risk_share,
             sigma_realized_ann, bias_ratio, var95_pred, var99_pred, es97_5_pred,
             betas_window_end, condition_number, max_vif, is_reliable, unreliable_reason)
        VALUES %s
    """, [(spec_id, instrument_id, r.as_of_date, horizon, cov_method,
           _f(r.sigma_pred_ann), _f(r.sigma_factor_ann), _f(r.sigma_specific_ann),
           _f(r.factor_risk_share), _f(r.sigma_realized_ann), _f(r.bias_ratio),
           _f(r.var95_pred), _f(r.var99_pred), _f(r.es97_5_pred), r.as_of_date,
           _f(r.condition_number), _f(r.max_vif), bool(r.is_reliable), r.unreliable_reason)
          for r in fc.itertuples()])

    if bt.get("sample_start") is not None:
        cur.execute("""
            INSERT INTO fact_risk_backtest
                (spec_id, instrument_id, sample_start, sample_end, horizon_days,
                 cov_method, n_forecasts, n_excluded, mean_bias, median_bias, z_std, z_kurtosis,
                 mz_alpha, mz_beta, mz_alpha_p, mz_beta_p, mz_joint_p, mz_r2,
                 exceptions_95, exceptions_99, expected_95, expected_99,
                 kupiec_stat_95, kupiec_p_95, kupiec_stat_99, kupiec_p_99,
                 christoffersen_stat_95, christoffersen_p_95,
                 christoffersen_stat_99, christoffersen_p_99)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (spec_id, instrument_id, sample_start, sample_end,
                         horizon_days, cov_method) DO UPDATE SET
                n_forecasts = EXCLUDED.n_forecasts, n_excluded = EXCLUDED.n_excluded,
                mean_bias = EXCLUDED.mean_bias,
                median_bias = EXCLUDED.median_bias, z_std = EXCLUDED.z_std,
                z_kurtosis = EXCLUDED.z_kurtosis, mz_alpha = EXCLUDED.mz_alpha,
                mz_beta = EXCLUDED.mz_beta, mz_alpha_p = EXCLUDED.mz_alpha_p,
                mz_beta_p = EXCLUDED.mz_beta_p, mz_joint_p = EXCLUDED.mz_joint_p,
                mz_r2 = EXCLUDED.mz_r2, computed_at = now()
        """, (spec_id, instrument_id, bt["sample_start"], bt["sample_end"], horizon,
              cov_method, bt["n_forecasts"], bt["n_excluded"], _f(bt["mean_bias"]), _f(bt["median_bias"]),
              _f(bt["z_std"]), _f(bt["z_kurtosis"]), _f(bt["mz_alpha"]), _f(bt["mz_beta"]),
              _f(bt["mz_alpha_p"]), _f(bt["mz_beta_p"]), _f(bt["mz_joint_p"]), _f(bt["mz_r2"]),
              bt["exceptions_95"], bt["exceptions_99"], _f(bt["expected_95"]),
              _f(bt["expected_99"]), _f(bt["kupiec_stat_95"]), _f(bt["kupiec_p_95"]),
              _f(bt["kupiec_stat_99"]), _f(bt["kupiec_p_99"]),
              _f(bt["christoffersen_stat_95"]), _f(bt["christoffersen_p_95"]),
              _f(bt["christoffersen_stat_99"]), _f(bt["christoffersen_p_99"])))
    return len(fc)


def run(instruments: Sequence[str], spec_id: str, horizon: int,
        cov_method: str, quiet: bool = False) -> int:
    with connect() as conn, conn.cursor() as cur:
        orthogonalized = spec_is_orthogonalized(cur, spec_id)
        factors = load_factor_panel(cur, orthogonalized)
        peer_var = peer_specific_variance(cur, spec_id)

    keys = [f"{spec_id[:8]}:{i}:{horizon}" for i in instruments]
    prune_items(JOB, keys)

    with etl_run(JOB, scope={"spec_id": spec_id, "horizon": horizon,
                             "cov_method": cov_method}) as run_id:
        for inst in instruments:
            item = f"{spec_id[:8]}:{inst}:{horizon}"
            try:
                with connect() as conn, conn.cursor() as cur:
                    y = load_instrument(cur, inst)
                    loadings = load_loadings(cur, spec_id, inst)
                    wmeta = load_window_meta(cur, spec_id, inst)

                if loadings.empty:
                    mark_item_done(run_id, JOB, item, "skipped", rows_out=0,
                                   error="no loadings; run run_estimation first")
                    print(f"  {inst:12s} no loadings for this spec", file=sys.stderr)
                    continue

                fc = forecast(y, factors, loadings, wmeta, horizon, cov_method,
                              peer_var, orthogonalized)
                bt = backtest(y, fc, horizon) if not fc.empty else {}

                with connect() as conn, conn.cursor() as cur:
                    n = write(cur, spec_id, inst, horizon, cov_method, fc, bt)

                mark_item_done(run_id, JOB, item, "succeeded", rows_out=n)
                if not quiet and bt:
                    print(f"  {inst:12s} {bt['n_forecasts']:4d} forecasts  "
                          f"bias={bt['mean_bias']:.3f}  z_std={bt['z_std']:.3f}  "
                          f"MZ_beta={bt['mz_beta']:.2f} (p={bt['mz_joint_p']:.3f})  "
                          f"VaR95 {bt['exceptions_95']}/{bt['expected_95']:.0f} "
                          f"(p={bt['kupiec_p_95']:.3f})"
                          + (f"  [{bt['n_excluded']} ill-conditioned]"
                             if bt["n_excluded"] else ""))
            except Exception as exc:
                mark_item_done(run_id, JOB, item, "failed", error=str(exc))
                print(f"  {inst:12s} FAILED: {exc}", file=sys.stderr)

    return run_failed(run_id)


def main() -> int:
    ap = argparse.ArgumentParser(description="Forecast risk and score it out of sample")
    ap.add_argument("--instrument", action="append")
    ap.add_argument("--spec", help="spec_id; default is the most recently created")
    ap.add_argument("--horizon", type=int, default=21,
                    help="forward window for realised volatility, in trading days")
    ap.add_argument("--cov-method", choices=("sample", "ewma", "ledoit_wolf", "blend"),
                    default="blend")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        spec_id = args.spec or latest_spec(cur)
        if not spec_id:
            print("no model spec exists; run run_estimation first", file=sys.stderr)
            return 1
        if args.instrument:
            instruments = args.instrument
        else:
            cur.execute("SELECT DISTINCT instrument_id FROM fact_loading "
                        "WHERE spec_id = %s ORDER BY 1", (spec_id,))
            instruments = [r[0] for r in cur.fetchall()]

    if not instruments:
        print(f"no instruments have loadings for spec {spec_id}", file=sys.stderr)
        return 1

    with connect() as conn, conn.cursor() as cur:
        panel = "orthogonalised" if spec_is_orthogonalized(cur, spec_id) else "raw"
    print(f"spec {spec_id}  horizon {args.horizon}d  cov {args.cov_method}  "
          f"{panel} factors  {len(instruments)} instruments")
    return 1 if run(instruments, spec_id, args.horizon, args.cov_method) else 0


if __name__ == "__main__":
    raise SystemExit(main())
