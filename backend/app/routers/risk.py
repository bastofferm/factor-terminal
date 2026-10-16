"""Predicted versus realised risk for a single security."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app import db
from backend.core import distribution as dist
from backend.core import risk as rk
from backend.pipeline import run_risk as rr

router = APIRouter()


class RiskRequest(BaseModel):
    instrument_id: str
    spec_id: str
    horizon_days: int = Field(default=21, ge=5, le=252)
    # A plain method, or the label of a risk assembly the model agent chose. The
    # agent's forecasts are stored, never recomputed here: forcing one would have
    # to know the whole RiskSpec, and the agent run that chose it already scored it.
    cov_method: str = Field(default="blend",
                            pattern="^(sample|ewma|ledoit_wolf|blend|agent:[0-9a-f]{10})$")
    force: bool = False


def _run_blocking(instruments: list[str], spec_id: str, horizon: int, method: str) -> None:
    rr.run(instruments, spec_id, horizon, method, quiet=True)


def _returns(instrument_id: str) -> pd.Series:
    """The security's excess returns, as the forecasts were made and scored on."""
    from backend.pipeline import run_estimation as est
    from backend.pipeline.dbsync import connect
    with connect() as conn, conn.cursor() as cur:
        y = est.load_instrument(cur, instrument_id)
    y.index = pd.to_datetime(y.index)
    return y


@router.post("/security")
async def security_risk(req: RiskRequest) -> dict:
    """Ex-ante forecast against the realised outcome, plus the full backtest.

    Every forecast uses loadings and a covariance matrix estimated on data ending at
    or before its own date; the realised volatility it is scored against covers the
    following `horizon_days`. Forecasts built on ill-conditioned windows are returned
    but flagged, and excluded from the scoring.
    """
    n = await db.fetchval(
        "SELECT count(*) FROM fact_risk_forecast WHERE spec_id=$1 AND instrument_id=$2 "
        "AND horizon_days=$3 AND cov_method=$4",
        req.spec_id, req.instrument_id, req.horizon_days, req.cov_method)

    if (not n or req.force) and req.cov_method.startswith("agent:"):
        raise HTTPException(
            404, "no stored forecasts for that agent model; run the model agent again")
    if not n or req.force:
        has_loadings = await db.fetchval(
            "SELECT count(*) FROM fact_loading WHERE spec_id=$1 AND instrument_id=$2",
            req.spec_id, req.instrument_id)
        if not has_loadings:
            raise HTTPException(
                422, "no loadings for this spec and instrument; estimate them first "
                     "via POST /api/loadings/estimate")
        await asyncio.to_thread(_run_blocking, [req.instrument_id], req.spec_id,
                                req.horizon_days, req.cov_method)

    forecasts = await db.fetch(
        """
        SELECT as_of_date, sigma_pred_ann, sigma_factor_ann, sigma_specific_ann,
               factor_risk_share, sigma_realized_ann, bias_ratio,
               var95_pred, var99_pred, es97_5_pred,
               condition_number, max_vif, is_reliable, unreliable_reason,
               sigma_horizon_ann
        FROM fact_risk_forecast
        WHERE spec_id=$1 AND instrument_id=$2 AND horizon_days=$3 AND cov_method=$4
        ORDER BY as_of_date
        """,
        req.spec_id, req.instrument_id, req.horizon_days, req.cov_method,
    )
    if not forecasts:
        raise HTTPException(422, "no forecasts could be produced for this combination")

    # What the one-day forecast is a forecast of: the volatility of the day after
    # its date, measured over the five days centred on that day. The h-day forecast
    # is paired with sigma_realized_ann, the h days after its date, as stored.
    y = await asyncio.to_thread(_returns, req.instrument_id)
    ctr = pd.Series(rk.centred_realized(y.to_numpy(), 5), index=y.index)
    nxt = y.index.searchsorted(pd.DatetimeIndex([f["as_of_date"] for f in forecasts]),
                               side="right")
    for f, k in zip(forecasts, nxt):
        ok = k < len(y) and np.isfinite(ctr.iloc[k])
        f["next_date"] = y.index[k].date() if k < len(y) else None
        f["realized_centred_ann"] = float(ctr.iloc[k]) if ok else None

    # The full-history backtest, and - for a model the agent chose - the one on the
    # held-out span its search never saw. Only the second is evidence about the
    # chosen model; the first includes the dates it was selected on.
    rows = await db.fetch(
        """
        SELECT * FROM fact_risk_backtest
        WHERE spec_id=$1 AND instrument_id=$2 AND horizon_days=$3 AND cov_method=$4
        ORDER BY computed_at DESC
        """,
        req.spec_id, req.instrument_id, req.horizon_days, req.cov_method,
    )
    backtest = next((r for r in rows if r.get("scope") in (None, "full")), None)
    held_out = next((r for r in rows if r.get("scope") == "test"), None)

    n_unreliable = sum(1 for f in forecasts if not f["is_reliable"])
    return {
        "instrument_id": req.instrument_id,
        "spec_id": req.spec_id,
        "horizon_days": req.horizon_days,
        "cov_method": req.cov_method,
        "forecasts": forecasts,
        "backtest": backtest,
        "backtest_test": held_out,
        "n_forecasts": len(forecasts),
        "n_unreliable": n_unreliable,
        "interpretation": _interpret(backtest, n_unreliable, held_out),
    }


def _benchmark_line(bt: dict, prefix: str = "") -> str | None:
    """Model QLIKE against the security's own EWMA volatility, in words."""
    q, q0 = bt.get("qlike"), bt.get("qlike_own_ewma")
    t, p = bt.get("dm_t_own_ewma"), bt.get("dm_p_own_ewma")
    if q is None or q0 is None or t is None or p is None:
        return None
    if p < 0.05:
        verdict = ("beats" if t < 0 else "loses to") + " it significantly"
    else:
        verdict = "is not distinguishable from it"
    return (f"{prefix}Against the naive forecast - the security's own EWMA volatility - "
            f"the model {verdict}: QLIKE {q:.3f} against {q0:.3f} "
            f"(Diebold-Mariano p = {p:.3f}).")


def _horizon_line(bt: dict) -> str | None:
    """The h-day forecast against the h-day return, where it differs from the daily one."""
    hl, ho = bt.get("horizon_loss"), bt.get("horizon_loss_own_ewma")
    p, t, nw = bt.get("dm_p_horizon_own_ewma"), bt.get("dm_t_horizon_own_ewma"), bt.get("nw_lags")
    if hl is None or ho is None or not nw:
        return None
    verdict = ("beats" if (t or 0) < 0 else "loses to") + " own EWMA significantly" \
        if p is not None and p < 0.05 else "is not distinguishable from own EWMA"
    return (f"The h-day forecast adds the cross-day covariance of factors that close at "
            f"different times ({nw} lag{'s' if nw > 1 else ''}). Scored against h-day "
            f"returns it {verdict}: loss {hl:.3f} against {ho:.3f}.")


def _interpret(bt: dict | None, n_unreliable: int,
               held_out: dict | None = None) -> list[str]:
    """Plain statements of what the statistics say.

    Written for an analyst who knows the tests but should not have to re-derive each
    threshold: the numbers are all returned alongside, so nothing here replaces them.
    """
    if not bt:
        return ["No backtest is available yet for this combination."]

    notes: list[str] = []
    line = _benchmark_line(bt)
    if line:
        notes.append(line)
    if held_out:
        line = _benchmark_line(held_out, "On the held-out span the agent never saw: ")
        if line:
            notes.append(line)
    line = _horizon_line(held_out or bt)
    if line:
        notes.append(line)

    z = bt.get("z_std")
    if z is not None:
        if abs(z - 1.0) < 0.10:
            notes.append(
                f"Calibrated: standardised returns have a standard deviation of "
                f"{z:.2f} against a target of 1.00.")
        elif z > 1.0:
            notes.append(
                f"Risk is understated: standardised returns have a standard deviation "
                f"of {z:.2f}, so realised moves were {z:.2f}x the forecast.")
        else:
            notes.append(
                f"Risk is overstated: standardised returns have a standard deviation "
                f"of {z:.2f}.")

    beta, joint_p = bt.get("mz_beta"), bt.get("mz_joint_p")
    if beta is not None and joint_p is not None:
        if joint_p > 0.05:
            notes.append(
                f"Mincer-Zarnowitz does not reject the forecast (slope {beta:.2f}, "
                f"joint p = {joint_p:.3f}).")
        else:
            notes.append(
                f"Mincer-Zarnowitz rejects unbiasedness (slope {beta:.2f}, joint "
                f"p = {joint_p:.3f}). Note a slope below 1 is partly mechanical when "
                f"volatility moves faster than the measurement horizon.")

    for level in (95, 99):
        exc, exp = bt.get(f"exceptions_{level}"), bt.get(f"expected_{level}")
        kp, cp = bt.get(f"kupiec_p_{level}"), bt.get(f"christoffersen_p_{level}")
        if exc is None or exp is None:
            continue
        line = f"VaR {level}%: {exc} exceptions against {exp:.0f} expected"
        if kp is not None:
            line += f" (Kupiec p = {kp:.3f}"
            if cp is not None:
                line += f", Christoffersen p = {cp:.3f}"
            line += ")"
        if cp is not None and kp is not None and kp > 0.05 and cp < 0.05:
            line += " — the count is right but the breaches cluster, which is the "
            line += "more dangerous failure."
        notes.append(line)

    if n_unreliable:
        notes.append(
            f"{n_unreliable} forecast(s) came from ill-conditioned estimation windows "
            f"and were excluded from the scoring.")
    return notes


@router.get("/decomposition/{spec_id}/{instrument_id}")
async def decomposition(spec_id: str, instrument_id: str,
                        cov_method: str = "blend") -> dict:
    """Factor versus specific risk through time, and the block split at the latest date.

    The question is not what the instrument earned but why it is
    risky, which is answered by the marginal contributions rather than the loadings.
    """
    series = await db.fetch(
        """
        SELECT as_of_date, sigma_pred_ann, sigma_factor_ann, sigma_specific_ann,
               factor_risk_share
        FROM fact_risk_forecast
        WHERE spec_id=$1 AND instrument_id=$2 AND cov_method=$3 AND is_reliable
        ORDER BY as_of_date
        """,
        spec_id, instrument_id, cov_method,
    )
    if not series:
        raise HTTPException(404, "no reliable forecasts stored for that combination")

    contributions = await db.fetch(
        """
        WITH latest AS (
            SELECT max(window_end) AS w FROM fact_loading
            WHERE spec_id=$1 AND instrument_id=$2
        )
        SELECT r.block_id, sum(abs(l.beta)) AS gross_exposure,
               count(*) AS n_factors,
               sum(CASE WHEN abs(l.t_stat) > 2 THEN 1 ELSE 0 END) AS n_significant
        FROM fact_loading l
        JOIN ref_factor r USING (factor_id)
        JOIN latest ON latest.w = l.window_end
        WHERE l.spec_id=$1 AND l.instrument_id=$2
        GROUP BY r.block_id ORDER BY 2 DESC
        """,
        spec_id, instrument_id,
    )
    return {
        "spec_id": spec_id,
        "instrument_id": instrument_id,
        "series": series,
        "block_exposure": contributions,
    }


@router.get("/standardized/{spec_id}/{instrument_id}")
async def standardized(spec_id: str, instrument_id: str, horizon_days: int = 21,
                       cov_method: str = "blend", bins: int = 50) -> dict:
    """Distribution of returns divided by the forecast that was in force.

    If the model is calibrated this is standard-normal-ish with unit variance; fat
    tails that survive standardisation mean the tail risk is real rather than a
    volatility-timing failure.
    """
    rows = await db.fetch(
        """
        SELECT as_of_date, sigma_pred_ann FROM fact_risk_forecast
        WHERE spec_id=$1 AND instrument_id=$2 AND horizon_days=$3
          AND cov_method=$4 AND is_reliable
        ORDER BY as_of_date
        """,
        spec_id, instrument_id, horizon_days, cov_method,
    )
    if not rows:
        raise HTTPException(404, "no forecasts stored for that combination")

    returns = await db.fetch(
        "SELECT date, ret_log FROM fact_input_return "
        "WHERE instrument_id=$1 AND ret_log IS NOT NULL ORDER BY date",
        instrument_id,
    )
    y = pd.Series({r["date"]: float(r["ret_log"]) for r in returns}).sort_index()
    pred = pd.Series({r["as_of_date"]: float(r["sigma_pred_ann"]) for r in rows})
    aligned = pd.DataFrame({"r": y, "sigma": pred.reindex(y.index).ffill().shift(1)}).dropna()
    if aligned.empty:
        raise HTTPException(422, "forecasts and returns do not overlap")

    z = rk.standardized_returns(aligned["r"].to_numpy(), aligned["sigma"].to_numpy())
    z = z[np.isfinite(z)]

    from scipy import stats as sstats

    # Same treatment as the factor histogram: standardised returns are if anything
    # fatter-tailed than raw ones, since dividing by the forecast removes the
    # volatility clustering but not the jumps.
    d = dist.estimate(z, bins=bins, tail_quantile=0.002, grid_points=400)
    q = dist.qq_points(z, max_points=400)

    # A standard normal is the reference here, not a fitted one: the whole question
    # is whether dividing by the forecast produces unit variance.
    reference = sstats.norm.pdf(np.asarray(d.grid)).round(5).tolist() if d.grid else []

    return {
        "instrument_id": instrument_id,
        "n": int(z.size),
        "std": float(np.std(z, ddof=1)),
        "skew": float(sstats.skew(z)),
        "excess_kurtosis": float(sstats.kurtosis(z)),
        "bin_centres": [round(v, 4) for v in d.bin_centres],
        "density": [round(v, 5) for v in d.density],
        "bin_width": d.bin_width,
        "n_bins": d.n_bins,
        "lo": d.lo,
        "hi": d.hi,
        "n_outside": d.n_outside,
        "grid": [round(v, 4) for v in d.grid],
        "kde": [round(v, 5) for v in d.kde],
        "normal_pdf": reference,
        "t_pdf": [round(v, 5) for v in d.t_pdf],
        "t_df": d.t_df,
        "qq_sample": [round(v, 4) for v in q["sample"]],
        "qq_normal": sstats.norm.ppf(q["probs"]).round(4).tolist() if q["probs"] else [],
    }
