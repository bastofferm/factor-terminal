"""FX legs on the NY-close clock.

Every factor is a NY-close return. Yahoo's `=X` pairs are not: the close filed under
date t is the rate at London midnight starting t, and since 2010-08-04 that is the
NY move of t-1. On it fx_jpy correlated +0.01 with eq_global on the same day and
-0.15 on the next, and after orthogonalisation still -0.29 with yesterday's fx_usd.
The legs now take their timing from the NY-listed CurrencyShares trusts.

The second failure was quieter. The warehouse FX table stopped on 2026-07-09 and was
filled forward without limit, so rt_uk's GBP leg was exactly zero for three months -
a gilt factor in pounds presenting itself as one in dollars.

The synthetic tests pin the mechanism; the integration tests at the bottom check the
regression on the real database, against the trusts and against a third
synchronous series, the dollar index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.pipeline import build_factors as bf
from backend.pipeline import seed

BREAK = pd.Timestamp(seed.YAHOO_FX_LAGGED_FROM)


def _world(seed_: int = 7, n: int = 1800, deposit_rate: float = 0.03):
    """A NY-close spot path, the trust that holds it, and Yahoo's copy of it.

    The trust earns a deposit rate net of its fee and adds a little tracking
    noise. Yahoo's level is synchronous before BREAK and the previous day's
    afterwards, as measured.
    """
    rng = np.random.default_rng(seed_)
    idx = pd.bdate_range("2008-01-01", periods=n)
    spot = pd.Series(rng.standard_normal(n) * 0.006, index=idx)
    deposit = pd.Series(deposit_rate / 252, index=idx)
    trust = spot + deposit + rng.standard_normal(n) * 0.0004
    level = np.exp(spot.cumsum())
    yahoo = level.where(idx < BREAK, level.shift(1))
    return idx, spot, trust, yahoo


def test_spot_from_the_trust_is_on_the_ny_clock():
    idx, spot, trust, yahoo = _world()
    s = bf.fx_spot_return(trust, yahoo)
    late = s.index >= BREAK + pd.Timedelta(days=400)
    assert s[late].corr(trust[late]) > 0.99
    assert abs(s[late].corr(trust.shift(1)[late])) < 0.1
    assert s[late].corr(spot[late]) > 0.99
    # The pair as stored is the one that is a day out.
    y = np.log(yahoo).diff()
    assert y[late].corr(spot[late]) < 0.1 < y[late].corr(spot.shift(1)[late])


def test_the_trusts_deposit_income_is_stripped():
    """sum(s) follows the spot move, not the trust's total return.

    The drift is a trailing one-year average, so a rate that moves is stripped
    with about half a year's lag; on the real trusts the yearly residual against
    spot is 0.2-0.35% a year, against 1-3% unstripped. Here the rate is constant.
    """
    idx, spot, trust, yahoo = _world()
    s = bf.fx_spot_return(trust, yahoo)
    late = s.index >= BREAK + pd.Timedelta(days=400)
    excess_trust = (trust - spot)[late].mean() * 252
    excess_s = (s - spot)[late].mean() * 252
    assert excess_trust > 0.025
    assert abs(excess_s) < 0.002


def test_yahoo_stands_in_only_while_it_was_synchronous():
    """Before the trust has a drift, the pair's own change is used - but only
    before BREAK. A missing trust day afterwards stays missing."""
    idx, spot, trust, yahoo = _world()
    trust = trust.copy()
    trust[idx < pd.Timestamp("2009-06-01")] = np.nan     # listed mid-2009
    hole = idx[idx >= BREAK][300]
    trust[hole] = np.nan
    s = bf.fx_spot_return(trust, yahoo)

    early = idx[(idx > idx[0]) & (idx < pd.Timestamp("2009-06-01"))]
    assert np.allclose(s[early], np.log(yahoo).diff()[early])
    assert s.loc[:BREAK - pd.Timedelta(days=1)].iloc[1:].notna().all()
    assert np.isnan(s[hole])


def test_a_stopped_spot_feed_leaves_gaps_not_zeros():
    """The rt_uk failure. A level carried forward is a currency that stopped
    moving; past the bound it must be missing, and so must the leg."""
    idx, spot, trust, yahoo = _world()
    stop = idx[-60]
    fx = pd.DataFrame({"GBP": yahoo[yahoo.index <= stop]})
    aligned = bf.align_spot(fx, idx)["GBP"]
    after = aligned[aligned.index > stop]
    assert after.iloc[:seed.FX_FFILL_LIMIT].notna().all()
    assert after.iloc[seed.FX_FFILL_LIMIT:].isna().all()

    # The trust keeps trading: the leg holds its drift for a few weeks, then stops.
    s = bf.fx_spot_return(trust, aligned)
    assert s.iloc[-30:].isna().all()
    # And the conversion of a still-trading local asset is missing there too,
    # not the local return with a zero FX move added.
    returns = pd.DataFrame({"FXB": trust, "IGLT.L": spot * 0.5}, index=idx)
    cash = pd.Series(0.0, index=idx)
    p = bf.Panels(returns, pd.DataFrame(index=idx), pd.DataFrame({"GBP": aligned}), cash)
    converted = p.in_base_ccy(p.ret("IGLT.L"), "GBP")
    assert converted.iloc[-30:].isna().all()
    assert converted.iloc[-300:-100].notna().all()


def test_spot_levels_read_every_pair_as_usd_per_unit():
    rows = pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02"],
        "instrument_id": ["USDJPY=X", "GBPUSD=X"],
        "close": [100.0, 1.25],
    })
    lv = bf.spot_levels(rows)
    assert lv.loc["2020-01-02", "JPY"] == pytest.approx(0.01)
    assert lv.loc["2020-01-02", "GBP"] == pytest.approx(1.25)


def test_every_currency_names_a_trust_and_its_pair():
    for ccy, (trust, pair) in seed.FX_SPOT.items():
        assert pair.endswith("=X") and ccy in pair, ccy
        assert trust.startswith("FX") and len(trust) == 3, ccy
    assert seed.fx_instruments(["GBP", "JPY"]) == ["FXB", "GBPUSD=X", "FXY", "USDJPY=X"]
    assert seed.fx_instruments("GBP", trusts_only=True) == ["FXB"]


# ---------------------------------------------------------------------------
# the real database
# ---------------------------------------------------------------------------

POST = slice("2011-01-01", None)


@pytest.fixture(scope="module")
def panels():
    from backend.pipeline.dbsync import connect
    try:
        with connect() as conn, conn.cursor() as cur:
            return bf.load_panels(cur)
    except Exception as exc:          # noqa: BLE001 - no database is a skip, not a failure
        pytest.skip(f"factors database unavailable: {exc}")


def _same_vs_lag(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a, b = a.loc[POST], b.loc[POST]
    return a.corr(b), a.corr(b.shift(1))


@pytest.mark.integration
@pytest.mark.parametrize("ccy", sorted(seed.FX_SPOT))
def test_each_fx_leg_moves_with_its_ny_trust_on_the_same_day(panels, ccy):
    """The regression the review asked for. As stored, the Yahoo pair fails it by
    a mile (about 0.05 same day, 0.9 the day before)."""
    trust, _ = seed.FX_SPOT[ccy]
    same, lag = _same_vs_lag(panels.fx_spot(ccy), panels.ret(trust))
    assert same > lag, f"{ccy}: same-day {same:.2f} <= previous-day {lag:.2f}"
    assert same > 0.9, f"{ccy}: same-day correlation with {trust} only {same:.2f}"


@pytest.mark.integration
@pytest.mark.parametrize("ccy", ["EUR", "JPY", "GBP", "CHF"])
def test_each_fx_leg_moves_with_the_dollar_index_on_the_same_day(panels, ccy):
    """Not circular: DX-Y.NYB is an ICE index on the NY clock, built from neither
    the trusts nor Yahoo's pairs. A stronger dollar is a weaker leg."""
    same, lag = _same_vs_lag(panels.fx_spot(ccy), panels.ret("DX-Y.NYB"))
    assert abs(same) > abs(lag) + 0.3, f"{ccy}: |{same:.2f}| vs |{lag:.2f}|"
    assert same < 0


def _stored(cur, factor_id: str) -> pd.Series:
    cur.execute("SELECT date, ret_excess FROM fact_factor_return "
                "WHERE factor_id = %s AND ret_excess IS NOT NULL ORDER BY date",
                (factor_id,))
    rows = cur.fetchall()
    s = pd.Series([float(v) for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
    return s


@pytest.mark.integration
def test_the_stored_factors_carry_the_synchronous_legs(panels):
    """What the rebuild wrote, not only what the code would build today."""
    from backend.pipeline.dbsync import connect
    with connect() as conn, conn.cursor() as cur:
        jpy = _stored(cur, "fx_jpy").reindex(panels.index)
        uk = _stored(cur, "rt_uk").reindex(panels.index)

    same, lag = _same_vs_lag(jpy, panels.ret("FXY"))
    assert same > 0.9 and same > lag, f"fx_jpy vs FXY: {same:.2f} / {lag:.2f}"

    leg = uk - panels.excess("IGLT.L")
    same, lag = _same_vs_lag(leg, panels.ret("FXB"))
    assert same > 0.9 and same > lag, f"rt_uk FX leg vs FXB: {same:.2f} / {lag:.2f}"
    # The frozen leg: after the warehouse table stopped it had a std of 4e-6.
    assert leg.dropna().iloc[-60:].std() > 1e-3


# ---------------------------------------------------------------------------
# a foreign security marked at the NY close
# ---------------------------------------------------------------------------

def test_a_tokyo_return_gets_every_ny_fx_move_exactly_once():
    """r_usd,t = r_local,t + sum of the NY-close FX moves since the previous Tokyo
    close. Tokyo trades on a US holiday (no NY move then), NY trades through a
    Tokyo holiday (its move lands on the next Tokyo day), a missing FX day drops the
    row it falls in, and a Tokyo day past the last NY close is not converted yet."""
    from backend.pipeline import sync

    ny = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                           "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
    s = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04, np.nan, 0.06, 0.07], index=ny)
    level = pd.Series(0.007, index=ny)
    # Tokyo: trades 01-06 (a Saturday stands in for a US-only holiday), shut 01-05
    # and 01-09, and has 01-12 before NY has closed it.
    tk = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-06", "2024-01-08",
                           "2024-01-10", "2024-01-11", "2024-01-12"])
    local = pd.DataFrame({"close": 100.0, "adj_close": 100.0, "log_return": 0.001,
                          "volume": 1}, index=tk)
    out = sync.convert_to_base(local, s, level)

    fx = (out["ret_log"] - 0.001).round(12)
    assert fx.loc["2024-01-04"] == 0.02
    assert fx.loc["2024-01-06"] == 0.03          # NY 01-05 moved, Tokyo shut that day
    assert fx.loc["2024-01-08"] == 0.04          # nothing in NY between 01-06 and 01-08
    assert "2024-01-10" not in fx.index.strftime("%Y-%m-%d")   # spans the missing 01-09
    assert fx.loc["2024-01-11"] == 0.07
    assert "2024-01-12" not in fx.index.strftime("%Y-%m-%d")   # NY 01-12 not closed yet
    assert "2024-01-03" not in fx.index.strftime("%Y-%m-%d")   # no previous Tokyo day
    # Prices convert at a level that cumulates the same moves (the local price is
    # flat here, so the whole log change of the USD price is the FX leg).
    r = np.log(out["adj_close"]).diff()
    for d in ["2024-01-06", "2024-01-08"]:
        assert r.loc[d] == pytest.approx(fx.loc[d], abs=1e-12)
