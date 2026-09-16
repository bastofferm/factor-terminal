"""Validate the constructed factors against published Fama-French and AQR series.

These are integration tests: they read the built factor panel from Postgres and
check it against independently-produced academic factors. This is the evidence that
the in-house construction is doing what it claims — a unit test on the arithmetic
cannot tell you that `sty_value` is actually a value factor.

They skip when the database or the factor panel is unavailable, so the pure unit
suite still runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def panels():
    try:
        from backend.pipeline.dbsync import connect
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"database layer unavailable: {exc}")

    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT date, factor_id, ret_orth FROM fact_factor_return "
                "WHERE ret_orth IS NOT NULL"
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT date, dataset, factor, ret_pct / 100.0 FROM fact_reference_factor"
            )
            ref_rows = cur.fetchall()
    except Exception as exc:
        pytest.skip(f"cannot read the factor database: {exc}")

    if not rows:
        pytest.skip("no factors built yet; run backend.pipeline.build_factors")

    F = (pd.DataFrame(rows, columns=["date", "f", "r"])
         .pivot(index="date", columns="f", values="r").sort_index())

    ref = pd.DataFrame(ref_rows, columns=["date", "dataset", "factor", "r"])
    ref["factor"] = ref["factor"].str.strip()
    ff = (ref[ref.dataset == "F-F_Research_Data_5_Factors_2x3_daily"]
          .pivot(index="date", columns="factor", values="r").sort_index())
    mom = (ref[ref.dataset == "F-F_Momentum_Factor_daily"]
           .pivot(index="date", columns="factor", values="r").sort_index())
    if not mom.empty:
        ff = ff.join(mom, how="outer")
    aqr = (ref[ref.dataset.str.startswith("AQR:")]
           .pivot(index="date", columns="dataset", values="r").sort_index())
    return F, ff, aqr


def _pair(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    return pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()


# ---------------------------------------------------------------------------
# the two factors that should reproduce their academic counterpart closely
# ---------------------------------------------------------------------------

def test_global_equity_tracks_the_market_factor(panels):
    """eq_global is built from ACWI, which is roughly two-thirds US, so it should be
    almost the same series as Fama-French Mkt-RF."""
    F, ff, _ = panels
    j = _pair(F["eq_global"], ff["Mkt-RF"])

    assert len(j) > 2000
    corr = j.corr().iloc[0, 1]
    beta = j.cov().iloc[0, 1] / j["b"].var()
    assert corr > 0.90, f"global equity vs Mkt-RF correlation only {corr:.3f}"
    assert 0.8 < beta < 1.2, f"beta to Mkt-RF is {beta:.2f}, expected near 1"


def test_size_factor_tracks_smb(panels):
    """eq_size is IWM minus IWB, a direct small-minus-large spread, so it should be
    a close proxy for SMB."""
    F, ff, _ = panels
    j = _pair(F["eq_size"], ff["SMB"])

    corr = j.corr().iloc[0, 1]
    beta = j.cov().iloc[0, 1] / j["b"].var()
    assert corr > 0.85, f"size vs SMB correlation only {corr:.3f}"
    assert 0.7 < beta < 1.3


# ---------------------------------------------------------------------------
# style factors: directionally right, not identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mine,ff_name,floor", [
    ("sty_value", "HML", 0.25),
    ("sty_momentum", "Mom", 0.40),
])
def test_style_factors_correlate_with_their_academic_counterpart(panels, mine, ff_name, floor):
    """These are long-only ETFs residualised against market and sector, standing in
    for true long-short portfolios. A correlation well below 1 is expected; a
    correlation near zero would mean the construction is wrong."""
    F, ff, _ = panels
    j = _pair(F[mine], ff[ff_name])

    assert len(j) > 1000
    corr = j.corr().iloc[0, 1]
    assert corr > floor, f"{mine} vs {ff_name} correlation only {corr:.3f}"


def test_value_and_momentum_are_negatively_correlated(panels):
    """The most robust cross-check available: value and momentum have been
    negatively correlated in every published study. If the in-house pair does not
    reproduce that, at least one of them is not measuring what it claims."""
    F, _, _ = panels
    j = _pair(F["sty_value"], F["sty_momentum"])

    assert len(j) > 1000
    assert j.corr().iloc[0, 1] < 0, "value and momentum should be negatively correlated"


def test_quality_and_lowvol_are_weak_proxies(panels):
    """Documents a known limitation rather than asserting success.

    MSCI Quality (ROE, leverage, earnings stability) is only loosely related to
    Fama-French RMW (operating profitability), and a long-only minimum-volatility
    ETF is not AQR's leveraged betting-against-beta portfolio. Both correlate
    positively but weakly. Recorded here so the weakness is visible rather than
    discovered later by someone trusting the loadings.
    """
    F, ff, aqr = panels

    q = _pair(F["sty_quality"], ff["RMW"]).corr().iloc[0, 1]
    assert 0.0 < q < 0.40, f"quality vs RMW correlation {q:.3f} outside the expected weak-positive range"

    if "AQR:BAB_Developed_Daily" in aqr.columns:
        lv = _pair(F["sty_lowvol"], aqr["AQR:BAB_Developed_Daily"]).corr().iloc[0, 1]
        assert 0.0 < lv < 0.45, f"low-vol vs BAB correlation {lv:.3f} outside the expected range"


# ---------------------------------------------------------------------------
# alternative risk premia
# ---------------------------------------------------------------------------

def test_trend_factor_correlates_with_managed_futures_etfs(panels):
    """arp_trend is a 13-asset time-series momentum rule. It should show a clear
    positive relationship to real managed-futures funds, which run the same style
    over a wider universe."""
    from backend.pipeline.dbsync import connect

    F, _, _ = panels
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT date, instrument_id, ret_log FROM fact_input_return "
            "WHERE instrument_id IN ('DBMF','KMLM') AND ret_log IS NOT NULL"
        )
        E = (pd.DataFrame(cur.fetchall(), columns=["date", "i", "r"])
             .pivot(index="date", columns="i", values="r").sort_index())

    if E.empty:
        pytest.skip("managed-futures ETFs not loaded")

    correlations = {etf: _pair(F["arp_trend"], E[etf]).corr().iloc[0, 1]
                    for etf in E.columns}
    assert all(c > 0.15 for c in correlations.values()), f"weak trend correlations: {correlations}"


# ---------------------------------------------------------------------------
# panel-level sanity
# ---------------------------------------------------------------------------

def test_every_factor_is_on_a_return_scale(panels):
    """Every factor in this model is a return. A z-score factor that
    escaped rescaling would show an annualised volatility near 1600%."""
    F, _, _ = panels
    vol = F.std() * np.sqrt(252)

    too_big = vol[vol > 1.0]
    assert too_big.empty, f"these factors are not on a return scale: {dict(too_big.round(2))}"

    too_small = vol[vol < 0.002]
    assert too_small.empty, f"these factors are implausibly quiet: {dict(too_small.round(5))}"


def test_no_factor_is_a_duplicate_of_another(panels):
    """Two factors with correlation above 0.95 are one factor with two names, and
    would make the covariance matrix near-singular."""
    F, _, _ = panels
    corr = F.corr()
    np.fill_diagonal(corr.values, 0.0)

    worst = corr.abs().max().max()
    pair = corr.abs().stack().idxmax()
    assert worst < 0.95, f"{pair[0]} and {pair[1]} are nearly identical (corr {worst:.3f})"


def test_factors_are_approximately_orthogonal_to_their_declared_targets(panels):
    """Rolling orthogonalisation cannot be exact when the true loading moves, but a
    large residual correlation means the hierarchy is not doing its job and the
    attribution will be ambiguous."""
    from backend.pipeline import factor_defs

    F, _, _ = panels
    offenders = {}
    for spec in factor_defs.FACTORS:
        for target in spec.get("orth", []):
            if spec["id"] not in F.columns or target not in F.columns:
                continue
            j = _pair(F[spec["id"]], F[target])
            if len(j) > 250:
                c = abs(j.corr().iloc[0, 1])
                if c > 0.25:
                    offenders[f"{spec['id']}~{target}"] = round(float(c), 3)

    assert not offenders, f"residual correlation above 0.25: {offenders}"
