"""What a pipeline run actually did, in words and in parameters.

`etl_run` records a job name, a mode, a status and a JSONB `scope`. That is enough
to audit a run and not enough to read one: `{"spec_id": "a8d1454d...", "horizon":
21, "cov_method": "blend"}` does not say that the risk job scored 249 monthly
forecasts of AAPL against the following month's realised volatility, on betas
estimated over 252 days on the raw factor panel.

This module supplies the missing half. Each job gets a description of what it does,
and its scope is turned into a short list of the parameters that actually change the
result — rendered as chips rather than as a JSON blob, because the point is to see
at a glance how two runs of the same job differed.

The parameters that matter most are not in `scope` at all. `run_estimation` and
`run_risk` record a `spec_id`, and every choice behind it — window, step, estimator,
weighting, ridge penalty, Dimson lags, and which factor panel — lives in
`dim_model_spec`. So the spec is resolved and unpacked here: two risk runs that
differ only in `orthogonalized` look identical in `scope`, and that difference is
the one most worth seeing.

Pure functions over rows. The router does the SQL.
"""

from __future__ import annotations

import json
from typing import Any

# What each job is for, in the analyst's terms rather than the scheduler's.
_WHAT_IT_DOES: dict[str, str] = {
    "sync_warehouse": "Mirrors the xbrl_sec warehouse into this database over "
                      "postgres_fdw: instrument prices, published reference "
                      "factors, and any security requested for analysis. Reads "
                      "nothing from the internet.",
    "ingest_yahoo": "Fetches daily prices from Yahoo Finance for the factor "
                    "universe and converts them to log total returns. Adjusted "
                    "closes only — a price-return series would bias every equity "
                    "beta down by the dividend yield.",
    "ingest_fred": "Fetches level series from the FRED API: policy rates, the "
                   "Treasury curve, funding spreads and financial-conditions "
                   "indices. Levels are stored raw and made stationary later, at "
                   "construction time.",
    "build_factors": "Constructs the forty factors of section 2.2 from the "
                     "instrument and level panels, in block-hierarchy order, and "
                     "residualises each one against the factors above it. Writes "
                     "both series: the raw factor and the orthogonalised one.",
    "run_diagnostics": "Runs the stationarity battery over every factor and "
                       "instrument: ADF and KPSS read jointly, Phillips-Perron, "
                       "Zivot-Andrews, the Lo-MacKinlay variance ratio, Ljung-Box, "
                       "ARCH-LM and the zero-return share. Records a verdict per "
                       "series and window.",
    "run_estimation": "Estimates rolling factor loadings for each requested "
                      "security: one regression per window end, with standard "
                      "errors, conditioning diagnostics and a stability comparison "
                      "against the previous window.",
    "run_risk": "Turns those loadings into ex-ante volatility forecasts and scores "
                "them against what actually happened. Every forecast uses a "
                "covariance estimated on data ending at or before its own date; "
                "the realised volatility it is judged on covers the following "
                "horizon.",
    "scheduler": "The nightly refresh: ingestion, construction and diagnostics in "
                 "dependency order, each step recorded as its own run.",
}

# Jobs whose `mode` is a real choice rather than a default nobody set.
_MODE_MATTERS = {"sync_warehouse", "ingest_yahoo", "ingest_fred", "build_factors"}

_MODE_HINT = {
    "full": "rebuilt from the start of history",
    "incremental": "only what was missing since the last run",
    "rolling": "trailing 504-day residualisation, refitted every 21 observations",
    "expanding": "residualised on all history to date — causal, but never forgets",
    "full_sample": "residualised once on the whole sample — exactly orthogonal, "
                   "and puts future information into historical factor values",
}

_ESTIMATOR_HINT = {
    "ols": "ordinary least squares",
    "huber": "Huber M-estimator, so one crisis day cannot set the beta",
    "ridge": "ridge-penalised, to stabilise correlated factors",
}

_COV_HINT = {
    "blend": "EWMA blended with a Ledoit-Wolf shrunk sample matrix",
    "ewma": "exponentially weighted, recent observations count for more",
    "ledoit_wolf": "Ledoit-Wolf shrinkage toward constant correlation",
    "sample": "plain sample covariance — the least stable of the four",
}

_STEP_NAME = {1: "daily", 5: "weekly", 21: "monthly", 63: "quarterly",
              126: "semi-annual", 252: "annual"}


def _chip(label: str, value: Any, hint: str | None = None,
          emphasis: bool = False) -> dict:
    """One parameter. `emphasis` marks the ones that change the answer most."""
    return {"label": label, "value": str(value), "hint": hint,
            "emphasis": emphasis}


def _as_dict(value: Any) -> dict:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}
    return dict(value or {})


def describe(job: str) -> str:
    return _WHAT_IT_DOES.get(job, "")


def chips(job: str, mode: str | None, scope: Any,
          spec: dict | None = None) -> list[dict]:
    """The parameters of one run, most consequential first.

    `spec` is the `dim_model_spec` row named by scope['spec_id'], already fetched,
    or None for the jobs that do not have one.
    """
    scope = _as_dict(scope)
    out: list[dict] = []

    if mode and (job in _MODE_MATTERS or mode not in ("incremental",)):
        out.append(_chip("mode", mode, _MODE_HINT.get(mode),
                         emphasis=job == "build_factors"))

    if spec:
        out.extend(_spec_chips(spec))

    # Scope keys the spec does not already cover.
    if "horizon" in scope:
        h = int(scope["horizon"])
        out.append(_chip("horizon", f"{h}d",
                         f"realised volatility is measured over the {h} trading days "
                         f"after each forecast date", emphasis=True))
    if "cov_method" in scope:
        m = str(scope["cov_method"])
        out.append(_chip("covariance", m, _COV_HINT.get(m)))
    if "window" in scope:
        out.append(_chip("window", f"{scope['window']}d",
                         "diagnostics are computed on this trailing window"))
    if "kinds" in scope and scope["kinds"]:
        kinds = ", ".join(str(k) for k in scope["kinds"])
        out.append(_chip("series", kinds, "which kinds of series were tested"))
    if "n_instruments" in scope:
        out.append(_chip("securities", scope["n_instruments"],
                         "instruments estimated in this run"))
    if "orth_mode" in scope and not any(c["label"] == "mode" for c in out):
        out.append(_chip("orthogonalisation", scope["orth_mode"],
                         _MODE_HINT.get(str(scope["orth_mode"])), emphasis=True))

    return out


def _spec_chips(spec: dict) -> list[dict]:
    """Unpack a model spec into the choices that produced its numbers.

    Ordered by how much each one moves the result, not by column order. The factor
    panel comes first because it is the choice that changes what a beta *means*,
    and two runs differing only in it are indistinguishable in `scope`.
    """
    out: list[dict] = []
    orth = spec.get("orthogonalized")
    if orth is not None:
        out.append(_chip(
            "panel", "orthogonalised" if orth else "raw",
            "block-hierarchy residuals; each beta is incremental over the blocks "
            "above it" if orth else
            "factors before residualisation; betas read directly, but the blocks "
            "overlap by construction",
            emphasis=True))

    window = spec.get("window_days")
    if window:
        out.append(_chip("window", f"{window}d",
                         f"each regression reads {window} trading days",
                         emphasis=True))

    step = spec.get("step_days")
    if step:
        name = _STEP_NAME.get(int(step))
        out.append(_chip("step", f"{step}d",
                         f"the window rolls forward {step} days at a time"
                         + (f" ({name})" if name else ""),
                         emphasis=True))

    est = spec.get("estimator")
    if est:
        out.append(_chip("estimator", est, _ESTIMATOR_HINT.get(str(est))))

    if spec.get("ridge_lambda"):
        out.append(_chip("ridge lambda", spec["ridge_lambda"],
                         "shrinks correlated loadings toward zero"))

    weighting = spec.get("weighting")
    if weighting and weighting != "equal":
        hl = spec.get("ewma_halflife")
        out.append(_chip("weighting", weighting,
                         f"exponential decay, half-life {hl} days" if hl else None))

    if spec.get("dimson_lags"):
        out.append(_chip("Dimson lags", spec["dimson_lags"],
                         "sums lagged betas, for instruments that price after the "
                         "factor close"))

    if spec.get("hac_lags") is not None:
        out.append(_chip("HAC lags", spec["hac_lags"],
                         "Newey-West standard errors"))

    if spec.get("winsor_lo") is not None:
        lo, hi = spec.get("winsor_lo"), spec.get("winsor_hi")
        out.append(_chip("winsorised", f"{lo:.0%}-{hi:.0%}",
                         "returns clipped to these quantiles before fitting"))

    n = spec.get("n_factors")
    if n:
        out.append(_chip("factors", n, "regressors in the design"))

    return out


def outcome(run: dict, items: dict) -> str:
    """One sentence on what came out, from the run's own counters.

    Deliberately derived rather than stored: a sentence written at run time would
    describe what the job set out to do, and this describes what it did.
    """
    job = run.get("job", "")
    status = run.get("status")
    if status == "running":
        return "Still running."
    if status == "failed":
        return f"Failed: {run.get('error') or 'no error recorded'}."

    ok = items.get("succeeded", 0)
    failed = items.get("failed", 0)
    skipped = items.get("skipped", 0)
    rows = run.get("rows_out") or 0

    # (singular, plural). Spelled out rather than suffixed with an "s": the two
    # jobs that process series would otherwise report "238 seriess".
    singular, plural = {
        "build_factors": ("factor", "factors"),
        "run_diagnostics": ("series", "series"),
        "run_estimation": ("security", "securities"),
        "run_risk": ("security", "securities"),
        "sync_warehouse": ("table", "tables"),
        "ingest_yahoo": ("instrument", "instruments"),
        "ingest_fred": ("series", "series"),
    }.get(job, ("item", "items"))
    noun = singular if ok == 1 else plural

    verb = "scored" if job == "run_risk" else "processed"
    parts = [f"{ok:,} {noun} {verb}, writing {rows:,} rows"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")

    secs = run.get("duration_seconds")
    if secs is not None:
        parts.append(f"in {_duration(secs)}")
    return ". ".join([", ".join(parts)]) + "."


def _duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1:
        return "under a second"
    if seconds < 1.5:
        return "1 second"
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} minutes"
    return f"{minutes / 60:.1f} hours"
