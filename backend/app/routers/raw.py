"""The Raw Explorer: the data before the model touches it.

The Factor Explorer describes the model — orthogonalised factors, because that is
what the regression, the covariance and the risk forecast consume. This serves the
other question: what does the underlying series actually do, before any of that.

Two kinds of series, deliberately on one page:

  factor      a constructed factor's own log excess return over cash, before
              block-hierarchy orthogonalisation
  instrument  the log return of a single input as ingested — EWJ, GLD, DGS10's
              synthetic bond leg — with no construction applied at all

Everything is returned in one response. The page is a secondary view and the
panels are useless separately, so eight round trips would buy nothing.
"""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.app import db, provenance as prov
from backend.core import distribution as dist
from backend.core import stationarity as st
from backend.core import summary as summ
from backend.pipeline import formula as fm

router = APIRouter()

TRADING_DAYS = 252
ROLLING_WINDOWS = (21, 63, 252)


@router.get("/catalog")
async def catalog() -> dict:
    """Everything that can be examined here, grouped by kind."""
    factors = await db.fetch(
        """
        SELECT f.factor_id AS id, f.name, f.block_id AS group_id,
               b.name AS group_name,
               c.first_date, c.last_date, c.n_obs,
               c.sd_ann, array_length(f.orthogonalize_against, 1) AS n_orth
        FROM ref_factor f
        LEFT JOIN ref_factor_block b USING (block_id)
        LEFT JOIN (
            SELECT factor_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs,
                   stddev_samp(ret_excess) * sqrt(252) AS sd_ann
            FROM fact_factor_return WHERE ret_excess IS NOT NULL GROUP BY 1
        ) c ON c.factor_id = f.factor_id
        WHERE f.is_active
        ORDER BY b.sort_order, f.factor_id
        """
    )
    instruments = await db.fetch(
        """
        SELECT i.instrument_id AS id, i.source_ticker AS name,
               COALESCE(i.asset_class, 'Other') AS group_id,
               COALESCE(i.asset_class, 'Other') AS group_name,
               i.is_live, i.is_total_return, i.currency, i.role,
               c.first_date, c.last_date, c.n_obs, c.sd_ann,
               (SELECT max(date) FROM fact_input_return) - c.last_date AS days_behind,
               -- Two very different reasons a series can be behind, which the
               -- is_live flag alone conflates. It is a pure staleness test, so it
               -- cannot tell "the provider stopped publishing" from "nothing here
               -- fetches this". CL=F trades every day; it is simply outside the
               -- nightly ingest, which covers the factor universe only, and so it
               -- sits at whatever date the warehouse was last refreshed.
               CASE
                   WHEN i.is_live THEN 'live'
                   WHEN i.role = 'analysis' THEN 'not_refreshed'
                   ELSE 'discontinued'
               END AS status
        FROM ref_instrument i
        LEFT JOIN (
            SELECT instrument_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs,
                   stddev_samp(ret_log) * sqrt(252) AS sd_ann
            FROM fact_input_return WHERE ret_log IS NOT NULL GROUP BY 1
        ) c ON c.instrument_id = i.instrument_id
        WHERE c.n_obs IS NOT NULL
        ORDER BY COALESCE(i.asset_class, 'Other'), i.instrument_id
        """
    )
    # Same correction as the detail view: the list is of raw series, so it must not
    # carry names that describe the residual. The hover title comes from here.
    factors = [{**f, "model_name": f["name"], "name": raw_name(f["name"])}
               for f in factors]
    return {"factors": factors, "instruments": instruments}


# A parenthetical beginning "ex-" names what the *orthogonalisation* removes, not
# what the series is: `eq_us` is registered as "US Equity (ex-global)" because that
# is what it becomes after eq_global is residualised out. On this page the series is
# plain US equity, so carrying the registry name over would put a claim on screen
# that is the opposite of what is plotted.
#
# Only that prefix is stripped. "(10s-2s)", "(butterfly)", "(SOFR-EFFR)" and
# "(small minus large)" describe the construction and are as true raw as
# orthogonalised.
_EX_QUALIFIER = re.compile(r"\s*\((?:ex|Ex)-[^)]*\)")


def raw_name(name: str | None) -> str:
    return _EX_QUALIFIER.sub("", name or "").strip()


async def _instrument_sources(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    rows = await db.fetch(
        """
        SELECT i.instrument_id AS id, i.source_ticker, i.source_table,
               i.asset_class, i.currency, i.is_total_return,
               c.first_date, c.last_date, c.n_obs
        FROM ref_instrument i
        LEFT JOIN (
            SELECT instrument_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs
            FROM fact_input_return WHERE ret_log IS NOT NULL GROUP BY 1
        ) c USING (instrument_id)
        WHERE i.instrument_id = ANY($1::text[])
        """,
        ids,
    )
    return [{**r, "kind": "instrument", "label": r["source_ticker"],
             "source": prov.instrument_source(r["source_table"])} for r in rows]


async def _level_sources(ids: list[str],
                         tenors: list | None = None) -> list[dict]:
    """Level series named directly, plus the members of any curve named.

    `tenors` narrows a curve to the legs the factor actually reads. Without it
    rt_us_slope, a two-legged 10s-2s steepener, lists all eleven Treasury
    maturities and looks like it consumes the whole curve.
    """
    if not ids:
        return []
    rows = await db.fetch(
        """
        SELECT r.series_id AS id, r.name, r.category, r.unit, r.transform,
               r.tenor_years, l.first_date, l.last_date, l.n_obs
        FROM ref_level_series r
        LEFT JOIN (
            SELECT series_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs
            FROM fact_input_level WHERE value IS NOT NULL GROUP BY 1
        ) l USING (series_id)
        WHERE r.series_id = ANY($1::text[]) OR r.curve_id = ANY($1::text[])
        ORDER BY r.tenor_years NULLS LAST, r.series_id
        """,
        ids,
    )
    if tenors:
        wanted = {round(float(t), 4) for t in tenors}
        rows = [r for r in rows
                if r["tenor_years"] is None
                or round(float(r["tenor_years"]), 4) in wanted]

    return [{**r, "kind": "level", "label": str(r["id"]).split(":", 1)[-1],
             "source": prov.level_series_source(r["id"])} for r in rows]


async def _load(kind: str, series_id: str, start: date | None,
                end: date | None) -> tuple[pd.Series, dict]:
    """The series plus whatever metadata that kind carries."""
    if kind == "factor":
        meta = await db.fetchrow(
            """
            SELECT f.factor_id AS id, f.name, f.block_id, b.name AS block_name,
                   f.construction, f.orthogonalize_against, f.hierarchy_level
            FROM ref_factor f LEFT JOIN ref_factor_block b USING (block_id)
            WHERE f.factor_id = $1
            """,
            series_id,
        )
        rows = await db.fetch(
            """
            SELECT date, ret_excess AS value FROM fact_factor_return
            WHERE factor_id = $1 AND ret_excess IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            series_id, start, end,
        )
    elif kind == "instrument":
        meta = await db.fetchrow(
            """
            SELECT instrument_id AS id, source_ticker AS name, asset_class,
                   currency, is_total_return, is_live, role, source_table, notes
            FROM ref_instrument WHERE instrument_id = $1
            """,
            series_id,
        )
        rows = await db.fetch(
            """
            SELECT date, ret_log AS value FROM fact_input_return
            WHERE instrument_id = $1 AND ret_log IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            series_id, start, end,
        )
    else:
        raise HTTPException(400, f"kind must be 'factor' or 'instrument', not {kind!r}")

    if not meta:
        raise HTTPException(404, f"no such {kind}: {series_id!r}")
    if not rows:
        raise HTTPException(404, f"no returns stored for {kind} {series_id!r}")

    meta = dict(meta)

    if kind == "factor":
        # The registry name describes the orthogonalised factor; this page plots the
        # raw one. `name` is corrected and the registry's own wording kept beside it,
        # so the two are not silently conflated.
        meta["model_name"] = meta.get("name")
        meta["name"] = raw_name(meta.get("name"))

        construction = meta.get("construction")
        if isinstance(construction, str):
            construction = json.loads(construction or "{}")
        construction = construction or {}
        meta["construction"] = construction
        inputs = construction.get("inputs") or {}
        named = prov.walk_inputs(inputs)
        meta["sources"] = (
            await _instrument_sources(named)
            + await _level_sources(named, inputs.get("tenors")))

        # A currency named in the inputs is converted through fact_input_fx rather
        # than being an instrument or a level series, so it resolves to neither of
        # the queries above and would otherwise vanish from the source list.
        ccy = inputs.get("fx")
        if ccy:
            meta["sources"].append({
                "id": ccy, "kind": "fx", "label": ccy,
                "source": "xbrl_sec warehouse",
                "name": f"USD per {ccy}",
            })

        meta["code"] = fm.source_code(construction.get("method"), inputs)
    else:
        meta["label"] = meta.get("name")
        meta["source"] = prov.instrument_source(meta.get("source_table"))
        # One entry so the source-data panel has the same shape for both kinds.
        meta["sources"] = [{
            "id": meta["id"], "kind": "instrument", "label": meta.get("name"),
            "source": meta["source"], "name": meta.get("notes"),
        }]

    s = pd.Series({r["date"]: float(r["value"]) for r in rows})
    s.index = pd.to_datetime(s.index)
    return s.sort_index(), meta


@router.get("/{kind}/{series_id}")
async def analyse(kind: str, series_id: str, start: date | None = None,
                  end: date | None = None, bins: int | None = None) -> dict:
    """Every panel of the raw explorer, in one response.

    The stationarity battery is run live rather than read from
    fact_series_diagnostics, because the stored verdicts are for the orthogonalised
    factors. A page that shows one series and the verdict for another would be
    worse than showing no verdict at all.
    """
    s, meta = await _load(kind, series_id, start, end)
    x = s.to_numpy()
    dates = [d.date().isoformat() for d in s.index]

    log_path = s.cumsum()
    rolling = {}
    for w in ROLLING_WINDOWS:
        r = s.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(TRADING_DAYS)
        rolling[str(w)] = [None if pd.isna(v) else round(float(v), 6) for v in r]

    d = dist.estimate(x, bins=bins)
    q = dist.qq_points(x)

    # Autocorrelation of returns and of squared returns: the first is a stale-pricing
    # warning, the second is volatility clustering and is expected.
    centred = x - x.mean()
    sq = x**2
    sq = sq - sq.mean()
    n = x.size

    def _acf(v: np.ndarray, lags: int = 30) -> list[float]:
        denom = float(np.dot(v, v))
        if denom <= 0:
            return [0.0] * (lags + 1)
        return [1.0 if k == 0 else float(np.dot(v[k:], v[:n - k]) / denom)
                for k in range(lags + 1)]

    diagnostics = st.analyse(x, dates=[d_.date() for d_ in s.index]).as_dict()

    return {
        "kind": kind,
        "id": series_id,
        "meta": meta,
        "dates": dates,
        "returns": s.round(8).tolist(),
        "cumulative": log_path.round(8).tolist(),
        "compounded": np.expm1(log_path).round(8).tolist(),
        "stats": summ.describe(x),
        "rolling": rolling,
        "rolling_windows": list(ROLLING_WINDOWS),
        "distribution": {
            "bin_centres": [round(v, 8) for v in d.bin_centres],
            "density": [round(v, 6) for v in d.density],
            "bin_width": d.bin_width, "n_bins": d.n_bins,
            "grid": [round(v, 8) for v in d.grid],
            "kde": [round(v, 6) for v in d.kde],
            "normal_pdf": [round(v, 6) for v in d.normal_pdf],
            "t_pdf": [round(v, 6) for v in d.t_pdf],
            "t_df": d.t_df, "lo": d.lo, "hi": d.hi,
            "n_outside": d.n_outside, "sd": d.sd,
        },
        "qq": {
            "sample": [round(v, 8) for v in q["sample"]],
            "normal": [round(v, 8) for v in q["normal"]],
            "t": [round(v, 8) for v in q["t"]],
            "t_df": q["t_df"],
        },
        "acf": {
            "lags": list(range(31)),
            "returns": _acf(centred),
            "squared": _acf(sq),
            "confidence_band": float(1.96 / np.sqrt(n)),
        },
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# the source series, in its own units
# ---------------------------------------------------------------------------

@router.get("/source/{ref_kind}/{ref_id:path}")
async def source_series(ref_kind: str, ref_id: str, start: date | None = None,
                        end: date | None = None) -> dict:
    """The stored series before anything was done to it.

    Every other panel on this page plots a return, which is already a
    transformation: a log difference of the thing that was actually downloaded.
    This serves what sits in the warehouse table - the adjusted close, the quoted
    yield in percent, the exchange rate - so a suspicious factor can be traced back
    to a number that came from a provider rather than from this code.

    `ref_id:path` because level series ids carry a colon, and instrument ids carry
    dots and equals signs: FRED:DGS10, DX-Y.NYB, USDJPY=X.
    """
    if ref_kind == "instrument":
        meta = await db.fetchrow(
            """
            SELECT instrument_id AS id, source_ticker, source_table, currency,
                   asset_class, is_total_return
            FROM ref_instrument WHERE instrument_id = $1
            """,
            ref_id,
        )
        if not meta:
            raise HTTPException(404, f"no such instrument: {ref_id!r}")
        rows = await db.fetch(
            """
            SELECT date, adj_close AS value, close AS unadjusted
            FROM fact_input_return
            WHERE instrument_id = $1 AND adj_close IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            ref_id, start, end,
        )
        meta = dict(meta)
        info = {
            "label": meta["source_ticker"],
            "source": prov.instrument_source(meta["source_table"]),
            "unit": meta["currency"] or "price",
            "quantity": "adjusted close",
            "table": "fact_input_return.adj_close",
            "is_total_return": meta["is_total_return"],
        }

    elif ref_kind == "level":
        meta = await db.fetchrow(
            """
            SELECT series_id AS id, name, unit, category, transform, tenor_years
            FROM ref_level_series WHERE series_id = $1
            """,
            ref_id,
        )
        if not meta:
            raise HTTPException(404, f"no such level series: {ref_id!r}")
        rows = await db.fetch(
            """
            SELECT date, value, NULL::double precision AS unadjusted
            FROM fact_input_level
            WHERE series_id = $1 AND value IS NOT NULL
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            ref_id, start, end,
        )
        meta = dict(meta)
        info = {
            "label": str(meta["id"]).split(":", 1)[-1],
            "source": prov.level_series_source(meta["id"]),
            "unit": meta["unit"] or "level",
            "quantity": meta["name"] or "quoted level",
            "table": "fact_input_level.value",
            "transform": meta["transform"],
        }

    elif ref_kind == "fx":
        rows = await db.fetch(
            """
            SELECT date, usd_per_unit AS value,
                   NULL::double precision AS unadjusted
            FROM fact_input_fx
            WHERE ccy = $1
              AND ($2::date IS NULL OR date >= $2)
              AND ($3::date IS NULL OR date <= $3)
            ORDER BY date
            """,
            ref_id, start, end,
        )
        info = {
            "label": ref_id,
            "source": "xbrl_sec warehouse",
            "unit": f"USD per {ref_id}",
            "quantity": "exchange rate",
            "table": "fact_input_fx.usd_per_unit",
        }

    else:
        raise HTTPException(
            400, f"ref_kind must be instrument, level or fx, not {ref_kind!r}")

    if not rows:
        raise HTTPException(404, f"no stored values for {ref_kind} {ref_id!r}")

    dates = [r["date"].isoformat() for r in rows]
    values = [float(r["value"]) for r in rows]

    # Adjusted against unadjusted is the dividend question made visible: for a
    # total-return ETF the two diverge by the accumulated distributions, and for a
    # price-return index they sit exactly on top of each other. That is the check
    # behind excluding the ^-prefixed index levels from construction.
    unadjusted = [None if r["unadjusted"] is None else float(r["unadjusted"])
                  for r in rows]
    has_unadjusted = any(v is not None for v in unadjusted)

    return {
        "ref_kind": ref_kind,
        "id": ref_id,
        **info,
        "dates": dates,
        "values": values,
        "unadjusted": unadjusted if has_unadjusted else None,
        "n_obs": len(values),
        "first_date": dates[0],
        "last_date": dates[-1],
        "first_value": values[0],
        "last_value": values[-1],
    }
