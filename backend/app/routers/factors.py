"""Single-factor analysis: returns, risk over time, distribution, diagnostics."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from scipy import stats

from backend.app import db
from backend.core import distribution as dist
from backend.core import stationarity as st
from backend.core import summary as summ
from backend.pipeline import formula as fm

router = APIRouter()

TRADING_DAYS = 252


# Which of the two stored series a request is about.
#
#   excess  the factor itself: a log excess return over cash, stationary, and what
#           an analyst means by "what did broad commodity do".
#   orth    the same series after block-hierarchy orthogonalisation, which is what
#           the regression consumes.
#
# These endpoints serve the Factor Explorer, which describes the model, so they
# default to "orth" — the series the regression, the covariance and the risk
# forecast all actually use. The raw view lives on its own page (/raw), which
# passes basis=excess, so the two are never mixed on one screen.
#
# The difference is large enough that mixing them would be a real error, not a
# nuance: eq_us returns 12.9% a year at 17.3% volatility, its residual after
# removing eq_global -0.2% at 4.2%.
_BASIS_COLUMN = {"excess": "ret_excess", "orth": "ret_orth"}


def _column(basis: str) -> str:
    col = _BASIS_COLUMN.get(basis)
    if col is None:
        raise HTTPException(
            400, f"basis must be one of {sorted(_BASIS_COLUMN)}, not {basis!r}")
    return col


async def _series(factor_id: str, start: date | None, end: date | None,
                  basis: str = "orth") -> pd.Series:
    col = _column(basis)
    rows = await db.fetch(
        f"""
        SELECT date, {col} AS value FROM fact_factor_return
        WHERE factor_id = $1 AND {col} IS NOT NULL
          AND ($2::date IS NULL OR date >= $2)
          AND ($3::date IS NULL OR date <= $3)
        ORDER BY date
        """,
        factor_id, start, end,
    )
    if not rows:
        raise HTTPException(404, f"no {basis} returns for factor {factor_id!r}")
    s = pd.Series({r["date"]: float(r["value"]) for r in rows})
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


@router.get("/{factor_id}/series")
async def series(factor_id: str, start: date | None = None, end: date | None = None,
                 cumulative: bool = True, basis: str = "orth") -> dict:
    """Daily returns with both cumulative paths.

    `cumulative` is the running sum of log returns; `compounded` is the same thing
    turned back into a simple return, exp(sum) - 1, which is what an investor
    actually ends up with. They are the same quantity expressed two ways and they
    diverge by more than people expect: a log path that ends at +0.5 is a +65%
    return, and one at -1.0 is -63%, not -100%.

    Both are returned together so the two panels can never be computed from
    different samples or a different basis.
    """
    s = await _series(factor_id, start, end, basis)
    out = {
        "factor_id": factor_id,
        "basis": basis,
        "dates": [d.date().isoformat() for d in s.index],
        "returns": s.round(8).tolist(),
    }
    if cumulative:
        log_path = s.cumsum()
        out["cumulative"] = log_path.round(8).tolist()
        out["compounded"] = np.expm1(log_path).round(8).tolist()
    return out


@router.get("/{factor_id}/inputs")
async def inputs(factor_id: str, start: date | None = None,
                 end: date | None = None) -> dict:
    """The stored series a factor is built from, as stored.

    Nothing is derived here: an instrument returns its adjusted close and a level
    series returns its level. That is the point of the endpoint. A cumulated
    return path is a construction, and when the construction is what is in doubt
    it cannot also be the evidence — a bad first observation moved the whole
    compounded path down by 78% and left a plausible-looking shape behind it.

    Units are reported per series rather than assumed, because a factor can be
    built from a price in dollars and a rate in percent at the same time, and
    putting those on one axis would be a chart that lies about scale.
    """
    row = await db.fetchrow(
        "SELECT construction FROM ref_factor WHERE factor_id = $1", factor_id)
    if row is None:
        raise HTTPException(404, f"no factor {factor_id!r}")

    construction = row["construction"]
    if isinstance(construction, str):
        construction = json.loads(construction)
    raw_inputs = (construction or {}).get("inputs") or {}
    names: list[str] = []
    _walk_inputs(raw_inputs, names)
    names = list(dict.fromkeys(names))
    if not names:
        return {"factor_id": factor_id, "series": [], "note": "no stored inputs"}

    # A construction names four different kinds of thing and they live in four
    # tables. Resolving by lookup rather than by the key it sat under means a
    # factor that mixes them - a gilt ETF priced in GBP, say - comes back whole.
    instruments = {r["id"]: r for r in await db.fetch(
        "SELECT instrument_id AS id, currency, asset_class FROM ref_instrument "
        "WHERE instrument_id = ANY($1)", names)}
    level_meta = {r["id"]: r for r in await db.fetch(
        "SELECT series_id AS id, name, unit FROM ref_level_series "
        "WHERE series_id = ANY($1)", names)}
    curves = {r["curve_id"] for r in await db.fetch(
        "SELECT DISTINCT curve_id FROM ref_level_series "
        "WHERE curve_id = ANY($1)", names)}
    currencies = {r["ccy"] for r in await db.fetch(
        "SELECT DISTINCT ccy FROM fact_input_fx WHERE ccy = ANY($1)", names)}

    # A curve input carries the tenors it wants; without them the whole curve is
    # the honest answer rather than an arbitrary subset.
    tenors = raw_inputs.get("tenors") if isinstance(raw_inputs, dict) else None

    async def level_rows(series_id: str) -> list:
        return await db.fetch(
            """
            SELECT date, value FROM fact_input_level
            WHERE series_id = $1 AND value IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """, series_id, start, end)

    def packed(sid: str, kind: str, label: str, unit: str | None, rows) -> dict:
        return {"id": sid, "kind": kind, "label": label, "unit": unit,
                "dates": [r["date"].isoformat() for r in rows],
                "values": [round(float(r["value"]), 6) for r in rows]}

    series: list[dict] = []
    for name in names:
        if name in instruments:
            meta = instruments[name]
            # An instrument's id is its ticker, which is the label a reader wants;
            # the asset class is what tells them why it is in this factor.
            label = f"{name} · {meta['asset_class']}" if meta["asset_class"] else name
            rows = await db.fetch(
                """
                SELECT date, adj_close AS value FROM fact_input_return
                WHERE instrument_id = $1 AND adj_close IS NOT NULL
                  AND ($2::date IS NULL OR date >= $2)
                  AND ($3::date IS NULL OR date <= $3)
                ORDER BY date
                """, name, start, end)
            series.append(packed(name, "instrument", label,
                                 meta["currency"] or "price", rows))

        elif name in level_meta:
            meta = level_meta[name]
            series.append(packed(name, "level", meta["name"] or name,
                                 meta["unit"] or "level", await level_rows(name)))

        elif name in curves:
            legs = await db.fetch(
                "SELECT series_id, name, unit, tenor_years FROM ref_level_series "
                "WHERE curve_id = $1 AND ($2::numeric[] IS NULL "
                "                         OR tenor_years = ANY($2)) "
                "ORDER BY tenor_years", name, tenors)
            for leg in legs:
                series.append(packed(
                    leg["series_id"], "level",
                    leg["name"] or f"{name} {float(leg['tenor_years']):g}y",
                    leg["unit"] or "percent", await level_rows(leg["series_id"])))

        elif name in currencies:
            rows = await db.fetch(
                """
                SELECT date, usd_per_unit AS value FROM fact_input_fx
                WHERE ccy = $1 AND usd_per_unit IS NOT NULL
                  AND ($2::date IS NULL OR date >= $2)
                  AND ($3::date IS NULL OR date <= $3)
                ORDER BY date
                """, name, start, end)
            series.append(packed(name, "fx", f"{name} per USD", "USD", rows))

        else:
            # Named in the construction but not stored under that id anywhere.
            # Reported rather than dropped: a silently missing leg is how a factor
            # ends up built from less than it claims.
            series.append({"id": name, "kind": "missing", "label": name,
                           "unit": None, "dates": [], "values": []})

    return {"factor_id": factor_id, "series": series}


@router.get("/{factor_id}/stats")
async def summary_stats(factor_id: str, start: date | None = None,
                        end: date | None = None, basis: str = "orth") -> dict:
    """Annualised moments, drawdown and tail measures.

    Shares backend.core.summary with the Raw Explorer, so the two pages cannot
    disagree about how a Sharpe or a drawdown is computed while describing
    different series.
    """
    s = await _series(factor_id, start, end, basis)
    out = summ.describe(s.to_numpy())
    out.update({
        "factor_id": factor_id,
        "basis": basis,
        "first_date": s.index.min().date().isoformat(),
        "last_date": s.index.max().date().isoformat(),
    })
    return out


@router.get("/{factor_id}/rolling-risk")
async def rolling_risk(factor_id: str, windows: str = "21,63,252",
                       start: date | None = None, end: date | None = None,
                       basis: str = "orth") -> dict:
    """Trailing annualised volatility at several window lengths.

    Short windows show the regime, long windows show the level. Plotting them
    together is how a drift in one becomes visible against the other.
    """
    s = await _series(factor_id, start, end, basis)
    try:
        sizes = [int(w) for w in windows.split(",") if w.strip()]
    except ValueError:
        raise HTTPException(400, "windows must be comma-separated integers")

    out: dict = {"factor_id": factor_id,
                 "dates": [d.date().isoformat() for d in s.index],
                 "series": {}}
    for w in sizes:
        if w < 2:
            continue
        roll = s.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(TRADING_DAYS)
        out["series"][str(w)] = [None if pd.isna(v) else round(float(v), 6) for v in roll]
    return out


@router.get("/{factor_id}/histogram")
async def histogram(factor_id: str, bins: int | None = None,
                    tail_quantile: float = 0.001, grid: int = 400,
                    start: date | None = None, end: date | None = None,
                    basis: str = "orth") -> dict:
    """Empirical distribution with a kernel density estimate and fitted overlays.

    `bins` defaults to the Freedman-Diaconis rule and the drawing range to the 0.1st
    to 99.9th percentile, both because an equal-width rule over the full range of a
    fat-tailed return series spends every bin on empty tail. Observations outside
    the range are counted and reported, never silently dropped.

    The Student-t fit is drawn alongside the normal because the gap between them in
    the tail is the clearest way to show why a normal VaR under-counts breaches.
    """
    s = await _series(factor_id, start, end, basis)
    x = s.to_numpy()
    d = dist.estimate(x, bins=bins, tail_quantile=tail_quantile, grid_points=grid)

    return {
        "factor_id": factor_id,
        "bin_centres": [round(v, 8) for v in d.bin_centres],
        "density": [round(v, 6) for v in d.density],
        "bin_width": d.bin_width,
        "n_bins": d.n_bins,
        "bin_rule": d.bin_rule,
        "grid": [round(v, 8) for v in d.grid],
        "kde": [round(v, 6) for v in d.kde],
        "normal_pdf": [round(v, 6) for v in d.normal_pdf],
        "t_pdf": [round(v, 6) for v in d.t_pdf],
        "t_df": d.t_df,
        "mean": d.mean,
        "sd": d.sd,
        "lo": d.lo,
        "hi": d.hi,
        "n": d.n,
        "n_outside": d.n_outside,
        "kde_bandwidth": d.kde_bandwidth,
        "jarque_bera_p": float(stats.jarque_bera(x).pvalue),
    }


@router.get("/{factor_id}/qq")
async def qq_plot(factor_id: str, start: date | None = None,
                  end: date | None = None, points: int = 500,
                  basis: str = "orth") -> dict:
    """Sample quantiles against Normal and Student-t theoretical quantiles."""
    s = await _series(factor_id, start, end, basis)
    q = dist.qq_points(s.to_numpy(), max_points=points)
    return {
        "factor_id": factor_id,
        "sample_quantiles": [round(v, 8) for v in q["sample"]],
        "normal_quantiles": [round(v, 8) for v in q["normal"]],
        "t_quantiles": [round(v, 8) for v in q["t"]],
        "t_df": q["t_df"],
    }


@router.get("/{factor_id}/acf")
async def acf(factor_id: str, lags: int = 30, start: date | None = None,
              end: date | None = None, basis: str = "orth") -> dict:
    """Autocorrelation of returns and of squared returns.

    The two answer different questions. Autocorrelation in returns is a stale-pricing
    warning. Autocorrelation in squared returns is volatility clustering, which is
    normal and is a reason to use HAC errors, not a defect.
    """
    s = await _series(factor_id, start, end, basis)
    x = s.to_numpy() - s.mean()
    x2 = s.to_numpy() ** 2
    x2 = x2 - x2.mean()
    n = x.size

    def _acf(v: np.ndarray) -> list[float]:
        denom = float(np.dot(v, v))
        if denom <= 0:
            return [0.0] * (lags + 1)
        return [float(np.dot(v[k:], v[:n - k]) / denom) if k else 1.0
                for k in range(lags + 1)]

    return {
        "factor_id": factor_id,
        "lags": list(range(lags + 1)),
        "acf": _acf(x),
        "acf_squared": _acf(x2),
        # Bartlett's approximate 95% band under the null of no autocorrelation.
        "confidence_band": float(1.96 / np.sqrt(n)),
        "n_obs": int(n),
    }


@router.get("/{factor_id}/diagnostics")
async def diagnostics(factor_id: str, window: int | None = None,
                      recompute: bool = False) -> dict:
    """The stationarity battery, from the store or recomputed on demand."""
    if recompute:
        s = await _series(factor_id, None, None)
        is_sparse = bool(await db.fetchval(
            "SELECT (construction -> 'inputs' ->> 'sparse')::boolean "
            "FROM ref_factor WHERE factor_id = $1", factor_id))
        if window:
            s = s.iloc[-window:]
        d = st.analyse(s.to_numpy(), dates=[i.date() for i in s.index], sparse=is_sparse)
        out = d.as_dict()
        out["factor_id"] = factor_id
        out["window_days"] = window or 0
        out["source"] = "recomputed"
        return out

    rows = await db.fetch(
        """
        SELECT DISTINCT ON (window_days) *
        FROM fact_series_diagnostics
        WHERE series_key = $1 AND series_type = 'factor'
          AND ($2::int IS NULL OR window_days = $2)
        ORDER BY window_days, as_of_date DESC
        """,
        factor_id, window,
    )
    if not rows:
        raise HTTPException(404, f"no diagnostics stored for {factor_id!r}; "
                                 f"run backend.pipeline.run_diagnostics")
    return {"factor_id": factor_id, "source": "stored", "windows": rows}


@router.get("/{factor_id}/reference-comparison")
async def reference_comparison(factor_id: str) -> dict:
    """Correlation with every published Fama-French and AQR series.

    This is what justifies calling an in-house construction a "value" or "momentum"
    factor. The published series lag by one to two months and can never drive the
    daily model, so they exist purely for this check.
    """
    rows = await db.fetch(
        """
        WITH mine AS (
            SELECT date, ret_orth AS r FROM fact_factor_return
            WHERE factor_id = $1 AND ret_orth IS NOT NULL
        )
        SELECT ref.dataset, ref.factor, count(*) AS n_overlap,
               corr(mine.r, ref.ret_pct / 100.0) AS correlation
        FROM fact_reference_factor ref
        JOIN mine USING (date)
        WHERE ref.ret_pct IS NOT NULL
        GROUP BY ref.dataset, ref.factor
        HAVING count(*) >= 250
        ORDER BY abs(corr(mine.r, ref.ret_pct / 100.0)) DESC NULLS LAST
        LIMIT 15
        """,
        factor_id,
    )
    return {"factor_id": factor_id, "comparisons": rows}


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

# Where each source table's data actually comes from, for the provenance line.
_SOURCE_LABEL = {
    "fact_cross_asset": "Yahoo Finance, via the xbrl_sec warehouse",
    "yahoo": "Yahoo Finance",
}

_METHOD_PROSE = {
    "single": "A single instrument's excess return over cash.",
    "basket": "An equally weighted long basket against an equally weighted short "
              "basket, so the common direction cancels and what is left is the "
              "spread between them.",
    "curve": "A yield-curve shape factor. Yield changes at the chosen tenors are "
             "turned into returns with a synthetic bond of matching duration, so "
             "the factor is a return rather than a change in a level.",
    "spread": "The difference between two instruments' excess returns.",
    "spread_level": "A standardised daily change in a quoted spread, not a price "
                    "return — the underlying series is a level, so it is "
                    "differenced before it can enter the model.",
    "level_transform": "A level series made stationary by the transform recorded "
                       "against it, then standardised.",
    "synthetic_bond": "A yield series converted into a bond return using a "
                      "duration assumption, because the raw series is a level.",
    "fx_carry": "Policy-rate differentials against the base currency, weighted "
                "equally across the available legs.",
    "tsmom": "A rule-based time-series momentum strategy: each instrument is held "
             "long or short according to the sign of its own trailing return.",
    "xs_momentum": "A rule-based cross-sectional momentum strategy: the strongest "
                   "instruments are held long against the weakest.",
    "variance_premium": "Implied variance minus subsequently realised variance, "
                        "the premium a variance seller earns on average.",
    "realized_vol_change": "The change in realised volatility, differenced because "
                           "the level itself is not stationary.",
}


def _walk_inputs(node, out: list[str]) -> None:
    """Collect every identifier mentioned anywhere in a construction's inputs."""
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for v in node:
            _walk_inputs(v, out)
    elif isinstance(node, dict):
        for key, v in node.items():
            # 'shape' and 'tenors' describe the recipe, not its ingredients.
            # 'inverted' likewise: it lists the currency legs of a carry basket
            # that are quoted the other way round, so its values are labels like
            # "JPY" and not instruments. Collecting them made fx_carry claim two
            # inputs it does not have.
            if key in ("shape", "transform", "window", "lookback", "sparse",
                       "inverted", "sign", "scale_to_vol"):
                continue
            _walk_inputs(v, out)


def _formula(construction: dict, targets: list[str] | None) -> dict | None:
    """The exact construction and orthogonalisation equations for this factor.

    Rendered from the stored construction JSON rather than from a hand-written
    table, so a factor whose rule changes cannot keep an out-of-date formula on
    screen. A method the renderer does not know is reported as a gap rather than
    faked — a plausible-looking wrong formula is worse than none.
    """
    try:
        return fm.for_factor({"construction": construction,
                              "orthogonalize_against": list(targets or [])})
    except ValueError as exc:
        return {"unavailable": str(exc)}


@router.get("/{factor_id}/profile")
async def profile(factor_id: str) -> dict:
    """Everything needed to explain one factor: what it is, and where it comes from.

    The sidebar shows forty identifiers. This resolves one of them into the
    instruments and level series it is actually built from, with their providers
    and coverage, so a reader does not have to open factor_defs.py to find out what
    `liq_risk_off` contains.
    """
    row = await db.fetchrow(
        """
        SELECT f.factor_id, f.name, f.block_id, f.hierarchy_level, f.version,
               f.construction, f.orthogonalize_against, b.name AS block_name,
               b.description AS block_description
        FROM ref_factor f
        LEFT JOIN ref_factor_block b USING (block_id)
        WHERE f.factor_id = $1
        """,
        factor_id,
    )
    if not row:
        raise HTTPException(404, f"no such factor: {factor_id!r}")

    construction = row["construction"]
    if isinstance(construction, str):
        construction = json.loads(construction)
    construction = construction or {}
    method = construction.get("method")
    inputs = construction.get("inputs") or {}

    names: list[str] = []
    _walk_inputs(inputs, names)
    unique = list(dict.fromkeys(names))

    instruments = await db.fetch(
        """
        SELECT instrument_id, source_ticker, source_table, asset_class, currency,
               is_total_return, is_live, first_obs, last_obs, n_obs, notes
        FROM ref_instrument WHERE instrument_id = ANY($1::text[])
        """,
        unique,
    ) if unique else []

    levels = await db.fetch(
        """
        SELECT r.series_id, r.name, r.category, r.unit, r.transform,
               r.tenor_years, r.curve_id, l.first_obs, l.last_obs, l.n_obs
        FROM ref_level_series r
        LEFT JOIN (SELECT series_id, min(date) AS first_obs, max(date) AS last_obs,
                          count(*) AS n_obs
                   FROM fact_input_level GROUP BY 1) l USING (series_id)
        WHERE r.series_id = ANY($1::text[]) OR r.curve_id = ANY($1::text[])
        ORDER BY r.tenor_years NULLS LAST, r.series_id
        """,
        unique,
    ) if unique else []

    # A curve factor names its curve and the tenors it uses; without this the whole
    # curve is listed and rt_us_slope appears to read eleven maturities when it
    # reads two.
    tenors = inputs.get("tenors") if isinstance(inputs, dict) else None
    if tenors:
        wanted = {round(float(t), 4) for t in tenors}
        levels = [l for l in levels
                  if l["tenor_years"] is None
                  or round(float(l["tenor_years"]), 4) in wanted]

    for i in instruments:
        i["source"] = _SOURCE_LABEL.get(i["source_table"], i["source_table"])
    for l in levels:
        # Level series ids are namespaced by provider: FRED:DGS10, ECB:BUND_10Y.
        l["source"] = str(l["series_id"]).split(":", 1)[0]

    coverage = await db.fetchrow(
        """
        SELECT min(date) AS first_date, max(date) AS last_date, count(*) AS n_obs,
               stddev_samp(ret_excess) * sqrt(252) AS vol_ann
        FROM fact_factor_return
        WHERE factor_id = $1 AND ret_excess IS NOT NULL
        """,
        factor_id,
    )

    orth = await db.fetch(
        "SELECT factor_id, name, block_id FROM ref_factor "
        "WHERE factor_id = ANY($1::text[]) ORDER BY hierarchy_level, factor_id",
        row["orthogonalize_against"] or [],
    ) if row["orthogonalize_against"] else []

    return {
        "factor_id": row["factor_id"],
        "name": row["name"],
        "block_id": row["block_id"],
        "block_name": row["block_name"],
        "block_description": row["block_description"],
        "hierarchy_level": row["hierarchy_level"],
        "version": row["version"],
        "method": method,
        "method_prose": _METHOD_PROSE.get(method, ""),
        "formula": _formula(construction, row["orthogonalize_against"]),
        "inputs": inputs,
        "note": construction.get("note"),
        "instruments": instruments,
        "level_series": levels,
        "orthogonalised_against": orth,
        "coverage": dict(coverage) if coverage else None,
    }


# ---------------------------------------------------------------------------
# raw against orthogonalised
# ---------------------------------------------------------------------------

@router.get("/{factor_id}/comparison")
async def comparison(factor_id: str, start: date | None = None,
                     end: date | None = None) -> dict:
    """The same factor on both panels, on exactly the same days.

    The model can be estimated on either panel, so the honest question is what the
    hierarchy actually took out of this factor and whether that changes what the
    factor means. Everything here is measured on the intersection of the two series:
    the orthogonalised one starts 252 observations later (the burn-in before the
    first refit), and comparing a full raw history against a shorter residual would
    attribute the difference in dates to the orthogonalisation.

    `removed` is the difference, f - f~, which is the fitted part: the exposure to
    the blocks above this one that the model books against those factors instead.
    """
    rows = await db.fetch(
        """
        SELECT date, ret_excess, ret_orth FROM fact_factor_return
        WHERE factor_id = $1
          AND ret_excess IS NOT NULL AND ret_orth IS NOT NULL
          AND ($2::date IS NULL OR date >= $2)
          AND ($3::date IS NULL OR date <= $3)
        ORDER BY date
        """,
        factor_id, start, end,
    )
    if not rows:
        raise HTTPException(404, f"no overlapping returns for factor {factor_id!r}")

    meta = await db.fetchrow(
        "SELECT name, block_id, hierarchy_level, orthogonalize_against "
        "FROM ref_factor WHERE factor_id = $1", factor_id)
    targets = list(meta["orthogonalize_against"] or []) if meta else []

    days = [r["date"] for r in rows]
    dates = [d.isoformat() for d in days]
    raw = np.array([float(r["ret_excess"]) for r in rows])
    orth = np.array([float(r["ret_orth"]) for r in rows])
    removed = raw - orth

    # A level-0 factor is stored twice and is the same series both times. Saying so
    # is more useful than printing a correlation of 1.000 and a variance share of 0.
    identical = bool(np.allclose(raw, orth, rtol=0, atol=1e-15))

    var_raw = float(np.var(raw, ddof=1))
    var_orth = float(np.var(orth, ddof=1))

    target_rows = await _target_series(targets, days[0], days[-1]) if targets else {}
    per_target = []
    X = []
    for tid, (tname, series) in target_rows.items():
        aligned = np.array([series.get(d, np.nan) for d in dates])
        ok = np.isfinite(aligned)
        per_target.append({
            "factor_id": tid,
            "name": tname,
            "n_common": int(ok.sum()),
            "corr_raw": _corr(raw[ok], aligned[ok]),
            "corr_orth": _corr(orth[ok], aligned[ok]),
        })
        X.append(aligned)

    return {
        "factor_id": factor_id,
        "name": meta["name"] if meta else factor_id,
        "block_id": meta["block_id"] if meta else None,
        "hierarchy_level": meta["hierarchy_level"] if meta else None,
        "identical": identical,
        "dates": dates,
        "cumulative": {
            "raw": np.cumsum(raw).tolist(),
            "orth": np.cumsum(orth).tolist(),
            "removed": np.cumsum(removed).tolist(),
        },
        "stats": {
            "raw": summ.describe(raw),
            "orth": summ.describe(orth),
            "removed": summ.describe(removed),
        },
        "alignment": {
            "n_obs": len(dates),
            "first_date": dates[0],
            "last_date": dates[-1],
            "correlation": _corr(raw, orth),
            # What share of the raw factor's variance the hierarchy removed. Not the
            # same as the R-squared of the fit: the loadings move over time, so the
            # residual variance is not var(raw) * (1 - R^2) of any single regression.
            "variance_removed": (1.0 - var_orth / var_raw) if var_raw > 0 else None,
            "tracking_vol_ann": float(np.std(removed, ddof=1) * np.sqrt(TRADING_DAYS)),
        },
        "targets": per_target,
        "implied_betas": _implied_betas(raw, np.array(X), list(target_rows)) if X else [],
    }


async def _target_series(targets: list[str], first: date,
                         last: date) -> dict[str, tuple[str, dict]]:
    """The orthogonalised series of each residualisation target, by date.

    Orthogonalised, not raw: that is what the regression in build_factors uses, so
    a correlation against the raw target would describe a fit nobody ran.
    """
    rows = await db.fetch(
        """
        SELECT f.factor_id, r.name, f.date, f.ret_orth
        FROM fact_factor_return f JOIN ref_factor r USING (factor_id)
        WHERE f.factor_id = ANY($1::text[]) AND f.ret_orth IS NOT NULL
          AND f.date >= $2::date AND f.date <= $3::date
        """,
        targets, first, last,
    )
    out: dict[str, tuple[str, dict]] = {}
    for r in rows:
        name, series = out.setdefault(r["factor_id"], (r["name"], {}))
        series[r["date"].isoformat()] = float(r["ret_orth"])
    # Preserve the hierarchy order the factor declares rather than the query's.
    return {t: out[t] for t in targets if t in out}


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _implied_betas(y: np.ndarray, X: np.ndarray, names: list[str]) -> list[dict]:
    """Full-sample loadings of the raw factor on its targets.

    A summary of what the rolling residualisation removes on average — not the
    coefficients it actually used, which are refitted every 21 days and differ
    window by window. Labelled `average_beta` for exactly that reason.
    """
    X = X.T if X.shape[0] == len(names) else X
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < X.shape[1] + 2:
        return []
    design = np.column_stack([np.ones(int(ok.sum())), X[ok]])
    coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    fitted = design @ coef
    resid = y[ok] - fitted
    ss_tot = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else None
    return [{"factor_id": n, "average_beta": float(b), "r2": r2, "n_obs": int(ok.sum())}
            for n, b in zip(names, coef[1:])]
