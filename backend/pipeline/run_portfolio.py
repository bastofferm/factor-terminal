"""Portfolio-level validation: does the model's risk aggregate?

Every other test in the project scores one security's total volatility. This
one forecasts the risk of portfolios built from a universe of US large caps -
the market, each sector, within-sector pairs, cross-sector spreads, random
baskets - from the stocks' betas, the factor covariance and their specific risks,
and scores those forecasts against what the portfolios did. It is the only test
of the off-diagonal structure, which is what a factor model adds over a list of
volatilities.

Five configurations share everything but the part under test:

    no sector             the equity factor template
    own sector            the template plus each stock's own GICS sector factor
    own sector, lead-lag  the same betas; the h-day forecast adds the cross-day
                          factor covariance (covariance.lead_lag_term)
    no sector, fast       the first two with a 21-day factor covariance
    own sector, fast      half-life in place of 42 days

Every forecast is causal: betas from the latest window ending at or before the
day, the factor covariance and specific risks through that day, scored on the
next. Results go to fact_portfolio_backtest.

    python -m backend.pipeline.run_portfolio [--start 2017-01-01] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

import numpy as np
import pandas as pd

from backend.agent import space as sp
from backend.core import covariance as cv
from backend.core import portfolio as pf
from backend.core import risk as rk
from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr
from backend.pipeline.dbsync import connect, etl_run, mark_item_done, mark_items, run_failed
from backend.pipeline.factor_defs import SECTOR_FACTORS

JOB = "portfolio_validation"

# Five large caps per GICS sector with history from 2008, four in Communication
# Services, where the catalogue has no CMCSA and files GOOGL elsewhere.
UNIVERSE: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "ORCL", "CSCO"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Health Care": ["JNJ", "PFE", "MRK", "UNH", "ABT"],
    "Industrials": ["HON", "CAT", "UNP", "MMM", "LMT"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "LOW"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Materials": ["SHW", "APD", "ECL", "NEM", "FCX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["PLD", "AMT", "SPG", "EQR", "PSA"],
    "Communication Services": ["VZ", "T", "DIS", "NFLX"],
}

# The risk assembly the model agent converged on for every security it was run
# on - a daily refresh, out-of-sample specific risk, the beta-noise correction -
# with a 42-day factor covariance half-life as the base case and the agent's 21
# days as the fast one. Measured on this universe (October 2026): the fast
# covariance follows each portfolio's volatility day by day, the slower one
# calibrates its 21-day risk and its 99% tail better.
COV_HALFLIFE = 42.0
FAST_COV_HALFLIFE = 21.0
SPECIFIC_HALFLIFE = 63.0
LEAD_LAGS = 2
LEAD_LAG_WINDOW = 504

# (name, own sector factor, lead-lag lags, factor covariance half-life)
CONFIGS = (("no sector", False, 0, COV_HALFLIFE),
           ("own sector", True, 0, COV_HALFLIFE),
           ("own sector, lead-lag", True, LEAD_LAGS, COV_HALFLIFE),
           ("no sector, fast", False, 0, FAST_COV_HALFLIFE),
           ("own sector, fast", True, 0, FAST_COV_HALFLIFE))


def universe() -> dict[str, str]:
    return {f"US:{t}": sec for sec, ts in UNIVERSE.items() for t in ts}


def ensure_mirrored(names: list[str]) -> None:
    from backend.pipeline import sync
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT instrument_id FROM fact_input_return "
                    "WHERE instrument_id = ANY(%s)", (names,))
        have = {r[0] for r in cur.fetchall()}
    for n in names:
        if n not in have:
            with connect() as conn, conn.cursor() as cur:
                sync.sync_security(cur, n)
            print(f"  mirrored {n}")


def _fit(y: pd.Series, F: pd.DataFrame, factors: list[str]):
    spec = est.Spec(factor_set=factors, window_days=252, step_days=sp.STEP_DAYS)
    loadings, meta, covs = est.estimate_with_cov(y, F, spec)
    L = loadings.pivot(index="window_end", columns="factor_id", values="beta").sort_index()
    L.index = pd.to_datetime(L.index)
    covs = {pd.Timestamp(k): v for k, v in covs.items()}
    return L, covs


def _held(index: pd.DatetimeIndex, ends: pd.DatetimeIndex) -> np.ndarray:
    """Position in `ends` of the latest window ending at or before each day; -1 if none."""
    return ends.searchsorted(index, side="right") - 1


def fit_universe(ys: dict[str, pd.Series], F: pd.DataFrame, template: list[str],
                 sector_of: dict[str, str], sector: bool) -> dict[str, tuple]:
    """Every stock's rolling fit, on the template with or without its sector factor."""
    fits = {}
    for n in sorted(ys):
        fs = list(template)
        own = SECTOR_FACTORS.get(sector_of[n])
        if sector and own in F.columns:
            fs.append(own)
        fits[n] = _fit(ys[n], F, fs)
    return fits


def assemble(ys: dict[str, pd.Series], F: pd.DataFrame, fits: dict[str, tuple],
             days: pd.DatetimeIndex, W: np.ndarray,
             cov_halflife: float = COV_HALFLIFE,
             specific_halflife: float = SPECIFIC_HALFLIFE,
             nw: int = 0, horizon: int = 21) -> dict:
    """Daily risk forecasts for the portfolios in the columns of W.

    For each day t: B_t from each holding's latest window ending at or before t,
    Sigma_F through t (causal EWMA over pairs of observed factors), specific
    variances through t; the forecast is for t+1:

        sigma^2_p = x' Sigma_F x - sum_i w_i^2 tr(Sigma_F V_i) + sum_i w_i^2 s_i^2,
        x = B' w

    `W` is holdings x portfolios, rows in sorted(ys) order. Returns the forecast
    path (annualised), the h-day path when `nw` lead-lag lags are asked for, and
    every component, so a caller can decompose the last day.
    """
    names = sorted(ys)
    U = sorted({f for L, _ in fits.values() for f in L.columns})
    K = len(U)
    T = len(days)
    pos_in_F = F.index.get_indexer(days)
    path = pf.ewma_cov_path(F[U].to_numpy(dtype=float), cov_halflife)[pos_in_F]  # T x K x K
    path = np.nan_to_num(path)
    FU = F[U].reindex(days).to_numpy(dtype=float)

    # Per holding and day: the held betas aligned to U, the held beta covariance
    # aligned to U, and the specific variance.
    B = np.zeros((T, len(names), K))
    noise = np.zeros((T, len(names)))
    s2 = np.full((T, len(names)), np.nan)
    pred_next = np.full((T, len(names)), np.nan)
    for i, n in enumerate(names):
        L, covs = fits[n]
        held = _held(days, L.index)
        Lw = L.reindex(columns=U).to_numpy(dtype=float)              # windows x K
        Lw0 = np.nan_to_num(Lw)
        Vw = np.zeros((len(L), K, K))
        for k, we in enumerate(L.index):
            if we in covs:
                vn, V = covs[we]
                ix = [U.index(f) for f in vn if f in U]
                jj = [j for j, f in enumerate(vn) if f in U]
                Vw[k][np.ix_(ix, ix)] = V[np.ix_(jj, jj)]
        ok = held >= 0
        B[ok, i] = Lw0[held[ok]]
        # tr(Sigma V), the beta-noise term, day by day.
        noise[ok, i] = np.einsum("tkl,tlk->t", path[ok], Vw[held[ok]])
        sv = rr.oos_specific_variance(ys[n], F.reindex(ys[n].index), L, covs,
                                      specific_halflife) / rk.TRADING_DAYS
        s2[:, i] = sv.reindex(days).clip(lower=rr.SPECIFIC_FLOOR_ANN**2 / rk.TRADING_DAYS)
        s2[~ok, i] = np.nan
        # The out-of-sample residual of day t+1 uses the betas held on day t.
        pred_next[:-1, i] = np.einsum("tk,tk->t", B[:-1, i], np.nan_to_num(FU[1:]))

    W2 = W**2
    X = np.einsum("tnk,np->tkp", B, W)                                   # T x K x P
    sys_ = np.einsum("tkp,tkl,tlp->tp", X, path, X) - noise @ W2
    spec = np.nan_to_num(s2) @ W2
    sig = np.sqrt((np.maximum(sys_, 0.0) + spec) * rk.TRADING_DAYS)
    usable = np.all(np.isfinite(s2), axis=1)
    sig[~usable] = np.nan

    sig_h = np.full_like(sig, np.nan)
    if nw:
        FA = F[U].to_numpy(dtype=float)
        for t in np.flatnonzero(usable):
            lo = max(0, pos_in_F[t] - LEAD_LAG_WINDOW + 1)
            term = cv.lead_lag_term(FA[lo:pos_in_F[t] + 1], nw, horizon, "ewma",
                                    cov_halflife, annualize=False)
            Ch = cv.horizon_covariance(path[t], np.nan_to_num(term))
            sys_h = np.einsum("kp,kl,lp->p", X[t], Ch, X[t]) - noise[t] @ W2
            sig_h[t] = np.sqrt((np.maximum(sys_h, 0.0) + spec[t]) * rk.TRADING_DAYS)

    Y = np.column_stack([ys[n].reindex(days).to_numpy(dtype=float) for n in names])
    return {"names": names, "factors": U, "days": days, "W": W, "B": B, "path": path,
            "noise": noise, "s2": s2, "usable": usable, "sig": sig, "sig_h": sig_h,
            "Y": Y, "pred_next": pred_next}


def run_config(name: str, nw: int, ys: dict[str, pd.Series], F: pd.DataFrame,
               fits: dict[str, tuple], sector_of: dict[str, str], days: pd.DatetimeIndex,
               portfolios: list[pf.Portfolio], horizon: int,
               cov_halflife: float = COV_HALFLIFE,
               specific_halflife: float = SPECIFIC_HALFLIFE) -> tuple[list[dict], dict]:
    """Forecast and score every portfolio under one configuration."""
    names = sorted(ys)
    W = np.array([[p.weights.get(n, 0.0) for p in portfolios] for n in names])   # n x P
    a = assemble(ys, F, fits, days, W, cov_halflife, specific_halflife, nw, horizon)
    usable, Y = a["usable"], a["Y"]
    resid = pd.DataFrame(np.nan, index=days, columns=names)
    resid.iloc[1:] = Y[1:] - a["pred_next"][:-1]
    resid[~np.concatenate([[False], usable[:-1]])] = np.nan

    R = np.nan_to_num(Y)
    rows = []
    for j, p in enumerate(portfolios):
        r = pd.Series(R @ W[:, j], index=days)
        m = pf.score(r, pd.Series(a["sig"][:, j], index=days), horizon,
                     pd.Series(a["sig_h"][:, j], index=days) if nw else None)
        rows.append({"config": name, "portfolio": p.name, "family": p.family,
                     "n_names": p.n_names, **m})
    structure = pf.residual_structure(resid, sector_of)
    return rows, structure


def run(start: str = "2017-01-01", horizon: int = 21, write: bool = True,
        quiet: bool = False) -> dict:
    sector_of = universe()
    names = sorted(sector_of)
    ensure_mirrored(names)
    with connect() as conn, conn.cursor() as cur:
        F = est.load_factor_panel(cur, True)
        F.index = pd.to_datetime(F.index)
        cur.execute("SELECT factor_id, block_id, role FROM ref_factor WHERE is_active "
                    "ORDER BY hierarchy_level, factor_id")
        reg = cur.fetchall()
        ys = {}
        for n in names:
            y = est.load_instrument(cur, n)
            y.index = pd.to_datetime(y.index)
            ys[n] = y
    blocks = {f: b for f, b, _ in reg}
    core = [f for f, _, role in reg if role == "core"]
    days = F.index[(F.index >= pd.Timestamp(start))]
    days = days[days <= min(y.index.max() for y in ys.values())]
    portfolios = pf.make_portfolios(sector_of)

    run_id = str(uuid.uuid4())
    out = {"run_id": run_id, "rows": [], "structure": {}, "summary": {}}
    template = sp.resolve_factors(sp.EstimationChoice(factor_set="template"), core, blocks,
                                  "equity")
    fits: dict[bool, dict] = {}
    for name, sector, nw, cov_hl in CONFIGS:
        if not quiet:
            print(f"  {name} ...", flush=True)
        if sector not in fits:
            fits[sector] = fit_universe(ys, F, template, sector_of, sector)
        rows, structure = run_config(name, nw, ys, F, fits[sector], sector_of, days,
                                     portfolios, horizon, cov_halflife=cov_hl)
        out["rows"] += rows
        out["structure"][name] = structure
        out["summary"][name] = pf.summarise(rows)

    if write:
        _store(run_id, horizon, out)
    if not quiet:
        _print(out)
    return out


def _clean(v: Any):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _store(run_id: str, horizon: int, out: dict) -> None:
    with connect() as conn, conn.cursor() as cur:
        for r in out["rows"]:
            metrics = {k: v for k, v in r.items()
                       if k not in ("config", "portfolio", "family", "n_names")}
            cur.execute(
                "INSERT INTO fact_portfolio_backtest (run_id, config, portfolio, family, "
                "n_names, horizon_days, metrics) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (run_id, r["config"], r["portfolio"], r["family"], r["n_names"], horizon,
                 json.dumps(_clean(metrics))))
        for config, s in out["structure"].items():
            cur.execute(
                "INSERT INTO fact_portfolio_backtest (run_id, config, portfolio, family, "
                "n_names, horizon_days, metrics) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (run_id, config, "residual structure", "residuals",
                 int(s.get("n_names") or 0), horizon,
                 json.dumps(_clean({**s, "summary": out["summary"][config]}))))


def _print(out: dict) -> None:
    for config, summ in out["summary"].items():
        st = out["structure"].get(config, {})
        print(f"\n  {config}: residual corr within sector "
              f"{st.get('mean_corr_within_sector', float('nan')):.3f}, across "
              f"{st.get('mean_corr_across_sectors', float('nan')):.3f}; residual PCs "
              f"{st.get('pc_share')}")
        losses = [r for r in out["rows"] if r["config"] == config]
        gap = np.nanmean([r["loss"] - r["loss_own_ewma"] for r in losses])
        print(f"    daily loss minus own EWMA's, mean over portfolios: {gap:+.4f}")
        print(f"    {'family':8s} {'n':>3s} {'in band':>8s} {'bias':>6s} {'z std':>6s} "
              f"{'<EWMA':>6s} {'K99 ok':>6s} {'K99t ok':>7s}")
        for fam, s in summ.items():
            print(f"    {fam:8s} {s['n']:3d} {s['bias_in_band']:8.2f} "
                  f"{s['median_bias_stat']:6.2f} {s['median_z_std']:6.2f} "
                  f"{s['beats_own_ewma']:6.2f} {s['kupiec_99_pass']:6.2f} "
                  f"{s['kupiec_99_t_pass']:7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the model's risk at portfolio level")
    ap.add_argument("--start", default="2017-01-01",
                    help="first scored day; every factor exists from mid-2016 on")
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    args = ap.parse_args()
    if args.dry_run:
        run(args.start, args.horizon, write=False)
        return 0
    with etl_run(JOB, scope={"start": args.start, "horizon": args.horizon}) as etl_id:
        mark_items(etl_id, JOB, ["universe"])
        try:
            out = run(args.start, args.horizon)
            mark_item_done(etl_id, JOB, "universe", "succeeded", rows_out=len(out["rows"]))
        except Exception as exc:
            mark_item_done(etl_id, JOB, "universe", "failed", error=str(exc))
            print(f"FAILED: {exc}", file=sys.stderr)
    return 1 if run_failed(etl_id) else 0


if __name__ == "__main__":
    raise SystemExit(main())
