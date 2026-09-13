"""Roll the regression over time and persist the loadings.

Implements the user-facing controls of PDF section 6: estimation window length,
roll-forward step (1d / 1w / 1m / 3m), equal or exponentially-decayed weighting,
OLS / Huber / ridge, HAC lag choice, Dimson lead-lag, and winsorisation.

Every parameter combination hashes to a spec_id, so a repeated request is a table
read rather than a refit.

    python -m backend.pipeline.run_estimation --instrument SPY --window 252 --step 21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from backend.core import regression as reg
from backend.core import transforms as tr
from backend.pipeline.dbsync import bulk_insert, connect, etl_run, mark_item_done, run_failed

JOB = "run_estimation"

STEP_ALIASES = {"1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}

# A factor missing more than this share of a window is dropped from that window
# rather than truncating the sample for every other factor. Style ETFs start in
# 2012-2014, so a fixed factor set would throw away a decade of history.
MIN_FACTOR_COVERAGE = 0.90


@dataclass
class Spec:
    """The estimation parameters. Hashes to a stable spec_id."""

    factor_set: list[str]
    window_days: int = 252
    step_days: int = 21
    estimator: str = "ols"
    weighting: str = "equal"
    ewma_halflife: int | None = None
    hac_lags: int | None = None
    ridge_lambda: float = 0.0
    dimson_lags: int = 0
    orthogonalized: bool = True
    winsor_lo: float | None = None
    winsor_hi: float | None = None
    min_obs: int = 126
    base_ccy: str = "USD"

    def canonical(self) -> dict:
        d = asdict(self)
        d["factor_set"] = sorted(self.factor_set)
        return d

    @property
    def spec_id(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def name(self) -> str:
        bits = [f"{self.window_days}d/{self.step_days}d", self.estimator]
        if self.weighting == "ewma":
            bits.append(f"hl{self.ewma_halflife}")
        if self.ridge_lambda:
            bits.append(f"ridge{self.ridge_lambda:g}")
        if self.dimson_lags:
            bits.append(f"dimson{self.dimson_lags}")
        return " ".join(bits)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_factor_panel(cur: Any, orthogonalized: bool = True) -> pd.DataFrame:
    col = "ret_orth" if orthogonalized else "ret_excess"
    cur.execute(
        f"SELECT date, factor_id, {col} FROM fact_factor_return "
        f"WHERE {col} IS NOT NULL AND factor_id IN "
        f"(SELECT factor_id FROM ref_factor WHERE is_active)"
    )
    df = pd.DataFrame(cur.fetchall(), columns=["date", "factor_id", "value"])
    if df.empty:
        raise SystemExit("no factor returns; run build_factors first")
    return df.pivot(index="date", columns="factor_id", values="value").sort_index()


def load_instrument(cur: Any, instrument_id: str) -> pd.Series:
    cur.execute(
        "SELECT r.date, r.ret_log, l.value FROM fact_input_return r "
        "LEFT JOIN fact_input_level l ON l.date = r.date AND l.series_id = %s "
        "WHERE r.instrument_id = %s AND r.ret_log IS NOT NULL ORDER BY r.date",
        ("FRED:DFF", instrument_id),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"no returns for {instrument_id!r}")
    df = pd.DataFrame(rows, columns=["date", "ret", "cash"]).set_index("date")
    df["cash"] = df["cash"].ffill()
    # The instrument's excess return is what the factors explain; the factors are
    # themselves excess returns, so both sides must be on the same footing.
    excess = tr.to_excess_return(df["ret"].to_numpy(), df["cash"].to_numpy())
    return pd.Series(excess, index=df.index).dropna()


def factor_constituents(cur: Any) -> dict[str, set[str]]:
    """Map each instrument to the factors built from it.

    Regressing an instrument on a factor constructed from that same instrument is
    circular: GLD against cm_gold returns an R-squared near 1 and a loading that
    says nothing. The estimator still runs — an analyst may legitimately want to see
    a factor ETF's exposure — but the overlap has to be visible, because a spurious
    0.99 fit is far more dangerous than a low one.
    """
    cur.execute("SELECT factor_id, construction -> 'inputs' FROM ref_factor WHERE is_active")
    out: dict[str, set[str]] = {}

    def walk(node: Any) -> Iterable[str]:
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        elif isinstance(node, dict):
            for key, v in node.items():
                if key in ("instrument", "long", "short", "universe", "underlying",
                           "vix", "pairs", "minuend", "subtrahend"):
                    yield from walk(v)

    for factor_id, inputs in cur.fetchall():
        for name in walk(inputs or {}):
            out.setdefault(name, set()).add(factor_id)
    return out


def blocked_factors(cur: Any, window: int) -> set[str]:
    """Factors whose trailing-window diagnostic says fail.

    PDF section 2 makes stationary inputs a precondition, so a failed series is
    excluded from estimation rather than merely flagged.
    """
    cur.execute(
        """
        SELECT DISTINCT ON (series_key) series_key, verdict
        FROM fact_series_diagnostics
        WHERE series_type = 'factor' AND window_days IN (%s, 0)
        ORDER BY series_key, window_days DESC, as_of_date DESC
        """,
        (window,),
    )
    return {k for k, v in cur.fetchall() if v == "fail"}


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------

def window_ends(index: pd.Index, window: int, step: int, min_obs: int) -> list:
    """Roll-forward dates, newest last. The first window needs `window` observations
    before it can be formed."""
    n = len(index)
    if n < max(window, min_obs):
        return []
    positions = list(range(n - 1, window - 2, -step))
    return [index[p] for p in reversed(positions)]


def estimate(
    y: pd.Series,
    factors: pd.DataFrame,
    spec: Spec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll the regression. Returns (loadings long-format, per-window diagnostics)."""
    joined = factors.reindex(y.index)
    ends = window_ends(y.index, spec.window_days, spec.step_days, spec.min_obs)

    loading_rows: list[dict] = []
    meta_rows: list[dict] = []
    prev_betas: np.ndarray | None = None
    prev_names: list[str] | None = None

    for end in ends:
        pos = y.index.get_loc(end)
        lo = pos - spec.window_days + 1
        if lo < 0:
            continue
        win_y = y.iloc[lo : pos + 1]
        win_X = joined.iloc[lo : pos + 1]

        # Keep only factors actually observed over this window. A style factor that
        # did not exist in 2005 must not truncate the sample for the whole panel.
        coverage = win_X.notna().mean()
        usable = [f for f in spec.factor_set
                  if f in coverage.index and coverage[f] >= MIN_FACTOR_COVERAGE]
        if len(usable) < 1:
            continue
        win_X = win_X[usable]

        rows_ok = win_y.notna() & win_X.notna().all(axis=1)
        if rows_ok.sum() < max(spec.min_obs, len(usable) + 2):
            continue

        yv = win_y[rows_ok].to_numpy()
        Xv = win_X[rows_ok].to_numpy()

        if spec.winsor_lo is not None and spec.winsor_hi is not None:
            yv = tr.winsorize(yv, spec.winsor_lo, spec.winsor_hi)

        design = reg.dimson_design(Xv, spec.dimson_lags) if spec.dimson_lags else Xv
        weights = (reg.ewma_weights(design.shape[0], spec.ewma_halflife or 60)
                   if spec.weighting == "ewma" else None)

        r = reg.fit(
            yv, design,
            factor_names=usable,
            estimator=spec.estimator,
            sample_weights=weights,
            hac_lags=spec.hac_lags,
            ridge_lambda=spec.ridge_lambda,
        )
        if r.n_obs == 0 or r.betas.size == 0:
            continue

        betas, ses = r.betas, r.se
        if spec.dimson_lags:
            betas, ses = reg.collapse_dimson(betas, ses, k=len(usable), lags=spec.dimson_lags)
        vifs = r.vif[: len(usable)] if r.vif.size >= len(usable) else np.full(len(usable), np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            tstats = np.where(ses > 0, betas / np.where(ses > 0, ses, 1.0), np.nan)

        window_start = win_y.index[0]
        for j, fid in enumerate(usable):
            loading_rows.append({
                "window_end": end, "factor_id": fid,
                "beta": float(betas[j]),
                "se": float(ses[j]) if np.isfinite(ses[j]) else None,
                "t_stat": float(tstats[j]) if np.isfinite(tstats[j]) else None,
                "p_value": float(r.p_values[j]) if j < r.p_values.size
                           and np.isfinite(r.p_values[j]) else None,
                "vif": float(vifs[j]) if np.isfinite(vifs[j]) else None,
            })

        # Compared on the factors common to this window and the last. The factor
        # set moves as coverage allows — style ETFs only start in 2012 — so
        # demanding an identical set left 169 gaps in 178 windows, all of them at
        # the moment the exposure set changed.
        l1, corr, overlap = reg.beta_stability(prev_betas, betas, prev_names, usable)
        prev_betas, prev_names = betas, list(usable)

        meta_rows.append({
            "window_end": end, "window_start": window_start, "n_obs": r.n_obs,
            "alpha": _f(r.alpha), "se_alpha": _f(r.se_alpha), "t_alpha": _f(r.t_alpha),
            "r2": _f(r.r2), "adj_r2": _f(r.adj_r2), "f_stat": _f(r.f_stat), "f_p": _f(r.f_p),
            "rmse": _f(r.rmse), "resid_vol_ann": _f(r.resid_vol_ann),
            "durbin_watson": _f(r.durbin_watson),
            "condition_number": _f(r.condition_number), "max_vif": _f(r.max_vif),
            "lb_resid_p": _f(r.lb_resid_p), "arch_lm_resid_p": _f(r.arch_lm_resid_p),
            "beta_shift_l1": _f(l1), "beta_corr_prev": _f(corr),
            "beta_overlap": overlap or None,
        })

    return pd.DataFrame(loading_rows), pd.DataFrame(meta_rows)


def _f(v: Any) -> float | None:
    """NaN and infinity are not storable as double precision in a meaningful way;
    a NULL says 'not computed' honestly."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def upsert_spec(cur: Any, spec: Spec) -> str:
    cur.execute(
        """
        INSERT INTO dim_model_spec
            (spec_id, name, factor_set, estimator, window_days, step_days, weighting,
             ewma_halflife, hac_lags, ridge_lambda, dimson_lags, orthogonalized,
             winsor_lo, winsor_hi, min_obs, base_ccy)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (spec_id) DO NOTHING
        """,
        (spec.spec_id, spec.name(), json.dumps(sorted(spec.factor_set)), spec.estimator,
         spec.window_days, spec.step_days, spec.weighting, spec.ewma_halflife,
         spec.hac_lags, spec.ridge_lambda, spec.dimson_lags, spec.orthogonalized,
         spec.winsor_lo, spec.winsor_hi, spec.min_obs, spec.base_ccy),
    )
    return spec.spec_id


def write_results(cur: Any, spec_id: str, instrument_id: str,
                  loadings: pd.DataFrame, meta: pd.DataFrame) -> int:
    cur.execute("DELETE FROM fact_loading WHERE spec_id=%s AND instrument_id=%s",
                (spec_id, instrument_id))
    cur.execute("DELETE FROM fact_regression_meta WHERE spec_id=%s AND instrument_id=%s",
                (spec_id, instrument_id))
    if loadings.empty:
        return 0

    bulk_insert(cur, """
        INSERT INTO fact_loading
            (spec_id, instrument_id, window_end, factor_id, beta, se, t_stat, p_value, vif)
        VALUES %s
    """, [(spec_id, instrument_id, r.window_end, r.factor_id, r.beta, r.se,
           r.t_stat, r.p_value, r.vif) for r in loadings.itertuples()])

    bulk_insert(cur, """
        INSERT INTO fact_regression_meta
            (spec_id, instrument_id, window_end, window_start, n_obs, alpha, se_alpha,
             t_alpha, r2, adj_r2, f_stat, f_p, rmse, resid_vol_ann, durbin_watson,
             condition_number, max_vif, lb_resid_p, arch_lm_resid_p,
             beta_shift_l1, beta_corr_prev, beta_overlap)
        VALUES %s
    """, [(spec_id, instrument_id, r.window_end, r.window_start, r.n_obs, r.alpha,
           r.se_alpha, r.t_alpha, r.r2, r.adj_r2, r.f_stat, r.f_p, r.rmse,
           r.resid_vol_ann, r.durbin_watson, r.condition_number, r.max_vif,
           r.lb_resid_p, r.arch_lm_resid_p, r.beta_shift_l1, r.beta_corr_prev,
           None if pd.isna(r.beta_overlap) else int(r.beta_overlap))
          for r in meta.itertuples()])

    cur.execute("""
        UPDATE fact_regression_meta m SET quality_score = q.score
        FROM v_regression_quality q
        WHERE q.spec_id = m.spec_id AND q.instrument_id = m.instrument_id
          AND q.window_end = m.window_end
          AND m.spec_id = %s AND m.instrument_id = %s
    """, (spec_id, instrument_id))
    return len(loadings)


def run_for(instruments: Sequence[str], spec: Spec, quiet: bool = False) -> int:
    with connect() as conn, conn.cursor() as cur:
        factors = load_factor_panel(cur, spec.orthogonalized)
        blocked = blocked_factors(cur, spec.window_days)
        constituents = factor_constituents(cur)
        upsert_spec(cur, spec)

    keep = [f for f in spec.factor_set if f not in blocked]
    if blocked & set(spec.factor_set):
        print(f"  excluded by diagnostics: {sorted(blocked & set(spec.factor_set))}")
    spec.factor_set = keep

    with etl_run(JOB, scope={"spec_id": spec.spec_id, "n_instruments": len(instruments)}) as run_id:
        for inst in instruments:
            try:
                with connect() as conn, conn.cursor() as cur:
                    y = load_instrument(cur, inst)
                loadings, meta = estimate(y, factors, spec)
                with connect() as conn, conn.cursor() as cur:
                    n = write_results(cur, spec.spec_id, inst, loadings, meta)
                mark_item_done(run_id, JOB, f"{spec.spec_id[:8]}:{inst}", "succeeded",
                               rows_out=n,
                               min_date=meta.window_end.min() if not meta.empty else None,
                               max_date=meta.window_end.max() if not meta.empty else None)
                if not quiet:
                    if meta.empty:
                        print(f"  {inst:12s} no estimable window")
                    else:
                        last = meta.iloc[-1]
                        print(f"  {inst:12s} {len(meta):4d} windows  "
                              f"last R2={last.adj_r2:.3f}  resid_vol={last.resid_vol_ann:.1%}  "
                              f"n={int(last.n_obs)}")
                circular = constituents.get(inst, set()) & set(spec.factor_set)
                if circular and not quiet:
                    print(f"  {'':12s} note: {inst} is a constituent of "
                          f"{sorted(circular)}; its fit is partly circular")
            except Exception as exc:
                mark_item_done(run_id, JOB, f"{spec.spec_id[:8]}:{inst}", "failed", error=str(exc))
                print(f"  {inst:12s} FAILED: {exc}", file=sys.stderr)

    return run_failed(run_id)


def default_factor_set(cur: Any) -> list[str]:
    cur.execute("SELECT factor_id FROM ref_factor WHERE is_active ORDER BY hierarchy_level, factor_id")
    return [r[0] for r in cur.fetchall()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate rolling factor loadings")
    ap.add_argument("--instrument", action="append", help="repeatable; default is the factor universe")
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--step", default="1m", help="1d, 1w, 1m, 3m or a number of days")
    ap.add_argument("--estimator", choices=("ols", "huber", "ridge"), default="ols")
    ap.add_argument("--weighting", choices=("equal", "ewma"), default="equal")
    ap.add_argument("--halflife", type=int, default=60)
    ap.add_argument("--hac-lags", type=int, default=None, help="default: Newey-West automatic rule")
    ap.add_argument("--ridge-lambda", type=float, default=0.0)
    ap.add_argument("--dimson-lags", type=int, default=0)
    ap.add_argument("--winsor", action="store_true", help="clip returns to 1%%-99%%")
    ap.add_argument("--min-obs", type=int, default=126)
    ap.add_argument("--factors", help="comma-separated subset; default is every active factor")
    ap.add_argument("--raw-factors", action="store_true",
                    help="use pre-orthogonalisation factor returns")
    args = ap.parse_args()

    step = STEP_ALIASES.get(args.step, None)
    if step is None:
        try:
            step = int(args.step)
        except ValueError:
            print(f"--step must be a number or one of {sorted(STEP_ALIASES)}", file=sys.stderr)
            return 2

    with connect() as conn, conn.cursor() as cur:
        factor_set = args.factors.split(",") if args.factors else default_factor_set(cur)
        if args.instrument:
            instruments = args.instrument
        else:
            cur.execute("SELECT instrument_id FROM ref_instrument "
                        "WHERE role IN ('analysis','both') AND is_live ORDER BY 1")
            instruments = [r[0] for r in cur.fetchall()]

    spec = Spec(
        factor_set=factor_set,
        window_days=args.window,
        step_days=step,
        estimator="ridge" if args.ridge_lambda > 0 and args.estimator == "ols" else args.estimator,
        weighting=args.weighting,
        ewma_halflife=args.halflife if args.weighting == "ewma" else None,
        hac_lags=args.hac_lags,
        ridge_lambda=args.ridge_lambda,
        dimson_lags=args.dimson_lags,
        orthogonalized=not args.raw_factors,
        winsor_lo=0.01 if args.winsor else None,
        winsor_hi=0.99 if args.winsor else None,
        min_obs=args.min_obs,
    )
    print(f"spec {spec.spec_id}  [{spec.name()}]  {len(spec.factor_set)} factors, "
          f"{len(instruments)} instruments")
    return 1 if run_for(instruments, spec) else 0


if __name__ == "__main__":
    raise SystemExit(main())
