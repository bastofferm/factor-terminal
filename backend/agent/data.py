"""Everything the agent reads, loaded once.

The search evaluates dozens of candidates on the same security, so the returns,
both factor panels and the reference facts are read from the database a single
time and held here. Tests build the same object from simulated series, which is
what lets the whole graph run against a process whose answer is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.core import stationarity as st


@dataclass
class AgentData:
    instrument_id: str
    y: pd.Series                       # daily excess log returns, DatetimeIndex
    factors: dict[str, pd.DataFrame]   # {"orth": ..., "raw": ...}, DatetimeIndex
    market: pd.Series | None           # eq_global excess return, for the CAPM benchmark
    blocks: dict[str, str]             # factor_id -> block_id
    active: list[str]                  # active factors in hierarchy order
    asset_class: str | None = None     # as the reference tables record it
    jurisdiction: str | None = None    # 'US', 'JP' for single securities
    security_type: str | None = None
    sector: str | None = None
    sector_factor: str | None = None   # the factor of its GICS sector, if built
    constituent_of: list[str] = field(default_factory=list)
    blocked: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.y = self.y.copy()
        self.y.index = pd.to_datetime(self.y.index)
        self.y = self.y.sort_index()
        for k, f in list(self.factors.items()):
            f = f.copy()
            f.index = pd.to_datetime(f.index)
            self.factors[k] = f.sort_index()
        if self.market is not None:
            self.market = self.market.copy()
            self.market.index = pd.to_datetime(self.market.index)


def load(cur: Any, instrument_id: str) -> AgentData:
    """Read one security and the factor panels it will be regressed on."""
    from backend.pipeline import run_estimation as est
    from backend.pipeline import run_risk as rr

    y = est.load_instrument(cur, instrument_id)
    factors = {"orth": est.load_factor_panel(cur, True),
               "raw": est.load_factor_panel(cur, False)}
    cur.execute("SELECT factor_id, block_id, role FROM ref_factor WHERE is_active "
                "ORDER BY hierarchy_level, factor_id")
    rows = cur.fetchall()
    blocks = {f: b for f, b, _ in rows}
    # 'all' and the templates draw on the core factors; a sector factor is added
    # only for the stock's own sector, and only when a candidate asks for it.
    active = [f for f, _, role in rows if role == "core"]

    cur.execute("SELECT asset_class FROM ref_instrument WHERE instrument_id = %s",
                (instrument_id,))
    r = cur.fetchone()
    asset_class = r[0] if r else None
    cur.execute("SELECT security_type, jurisdiction, sector, currency FROM ref_security "
                "WHERE instrument_id = %s OR ticker = %s LIMIT 1",
                (instrument_id, instrument_id))
    s = cur.fetchone()
    security_type, jurisdiction, sector = (s[0], s[1], s[2]) if s else (None, None, None)
    if ":" in instrument_id:
        jurisdiction = instrument_id.split(":", 1)[0].upper()

    constituents = est.factor_constituents(cur).get(instrument_id, set())
    blocked = est.blocked_factors(cur, 252)

    from backend.pipeline.factor_defs import SECTOR_FACTORS
    sector_factor = SECTOR_FACTORS.get(sector or "")
    if sector_factor not in factors["orth"].columns or sector_factor in blocked:
        sector_factor = None

    return AgentData(
        instrument_id=instrument_id, y=y, factors=factors,
        market=rr.load_market_factor(cur), blocks=blocks, active=active,
        asset_class=asset_class, jurisdiction=jurisdiction,
        security_type=security_type, sector=sector, sector_factor=sector_factor,
        constituent_of=sorted(constituents), blocked=set(blocked),
    )


# ---------------------------------------------------------------------------
# the profile: facts about the security that shape the search
# ---------------------------------------------------------------------------

# Markets that close before New York. Their prices react to US-hours factor moves
# a day late, which is what the Dimson correction is for.
_EARLY_CLOSE = {"JP"}
_EARLY_SUFFIXES = (".L", ".T", ".DE", ".PA", ".HK", ".AX", ".SS")


def classify(data: AgentData, before: pd.Timestamp | None = None) -> str:
    """Which factor template fits: equity, equity_jp, fixed_income, commodity, fx or multi.

    The reference tables first; where they are silent or say "Other", the
    security's own correlations with one lead factor per asset class, measured
    only on data before `before` so the choice cannot see the test span.
    """
    if data.jurisdiction == "JP":
        return "equity_jp"
    if data.jurisdiction == "US":
        return "equity"
    ac = (data.asset_class or "").lower()
    iid = data.instrument_id
    if iid.endswith("=X") or "fx" in ac or "currenc" in ac:
        return "fx"
    if any(w in ac for w in ("bond", "fixed", "credit", "govt", "treasur")):
        return "fixed_income"
    if "commod" in ac:
        return "commodity"
    if "equity" in ac:
        return "equity"

    leads = {"equity": "eq_global", "fixed_income": "rt_us_level",
             "commodity": "cm_broad", "fx": "fx_usd"}
    raw = data.factors.get("raw")
    if raw is None:
        return "multi"
    y = data.y if before is None else data.y[data.y.index < before]
    best, best_r = "multi", 0.0
    for cls, f in leads.items():
        if f not in raw.columns:
            continue
        both = pd.concat([y, raw[f]], axis=1, join="inner").dropna()
        if len(both) < 252:
            continue
        r = abs(float(both.iloc[:, 0].corr(both.iloc[:, 1])))
        if r > best_r:
            best, best_r = cls, r
    return best if best_r >= 0.3 else "multi"


def profile(data: AgentData, before: pd.Timestamp | None = None) -> dict:
    """What the agent knows about the security before it fits anything."""
    y = data.y if before is None else data.y[data.y.index < before]
    diag = st.analyse(y.to_numpy(), run_zivot_andrews=False)
    asynchronous = (data.jurisdiction in _EARLY_CLOSE
                    or data.instrument_id.endswith(_EARLY_SUFFIXES))
    return {
        "instrument_id": data.instrument_id,
        "asset_class": data.asset_class,
        "security_type": data.security_type,
        "jurisdiction": data.jurisdiction,
        "sector": data.sector,
        "sector_factor": data.sector_factor,
        "template": classify(data, before),
        "asynchronous_close": bool(asynchronous),
        "n_obs": int(y.notna().sum()),
        "first_date": y.index.min().date().isoformat() if len(y) else None,
        "last_date": data.y.index.max().date().isoformat() if len(data.y) else None,
        "vol_ann": float(np.nanstd(y.to_numpy(), ddof=1) * np.sqrt(252)) if len(y) > 2 else None,
        "zero_return_share": diag.zero_return_share,
        "excess_kurtosis": diag.excess_kurtosis,
        "ac1": diag.ac1,
        "stationarity": diag.verdict,
        "stationarity_reason": diag.verdict_reason,
        "constituent_of": list(data.constituent_of),
        "blocked_factors": sorted(data.blocked),
        "n_active_factors": len(data.active),
    }
