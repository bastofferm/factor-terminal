"""Check that the rendered formulas are the arithmetic the builders actually run.

A formula module is only worth having if it cannot drift from the code it claims to
describe, and prose comparison does not catch drift. So each test here evaluates
the rendered formula *independently* -- written out in the shape the formula states,
not by calling the builder's helpers -- and asserts it reproduces
`build_factors.build_one` to floating-point tolerance on simulated inputs.

Where that would be circular (the duration and convexity closed forms, say) the
independent version writes the closed form out by hand rather than importing
`transforms`, which is the whole point.

The same treatment is applied to the orthogonalisation equation: the stated argmin
over a trailing window, refitted every 21 observations, is implemented directly from
the rendered formula and checked against `orthogonalize.orthogonalize`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.core import orthogonalize as og
from backend.pipeline import build_factors as bf
from backend.pipeline import factor_defs, formula, seed

SEED = 20260914
N_DAYS = 900
TRADING_DAYS = 252

# Every ticker any tested factor reads, plus the ARP universe.
TICKERS = sorted({
    "ACWI", "IVV", "VGK", "EWJ", "EEM", "IWM", "IWB", "IGLT.L", "USDJPY=X",
    "EURUSD=X", "USDCHF=X", "DX-Y.NYB", "CEW", "TLT", "UUP", "GLD", "HYG", "SPY",
    "EFA", "IEF", "DBC", "USO", "LQD", "FXE", "FXY", "VIXY", "DBA", "CPER", "EMB",
    "EMLC", "BKLN", "VLUE", "MTUM", "QUAL", "USMV", "XLI", "XLB", "XLE", "XLF",
    "XLY", "XLP", "XLU", "XLV",
})

CURVE_SERIES = [sid for sid, _ in seed.CURVES["US_TSY"]]
LEVEL_SERIES = sorted(set(
    CURVE_SERIES
    + [seed.CASH_RATE_SERIES, "FRED:T10YIE", "FRED:SOFR", "FRED:EFFR", "FRED:VIXCLS"]
    + list(seed.POLICY_RATES.values())
))

# NFCI prints weekly; the sparse path is only taken below DAILY_COVERAGE_THRESHOLD.
SPARSE_SERIES = "FRED:NFCI"


@pytest.fixture(scope="module")
def panels() -> bf.Panels:
    """A synthetic but structurally faithful panel: daily returns, yield levels in
    percent, an FX rate, and a weekly series with genuine gaps."""
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2019-01-01", periods=N_DAYS)

    returns = pd.DataFrame(
        rng.standard_normal((N_DAYS, len(TICKERS))) * 0.01,
        index=idx, columns=TICKERS,
    )

    levels = pd.DataFrame(index=idx, dtype=float)
    for sid in LEVEL_SERIES:
        # A yield-like level in percent: positive, slowly drifting, never degenerate.
        walk = np.cumsum(rng.standard_normal(N_DAYS) * 0.03)
        levels[sid] = 2.5 + walk - walk.mean()
    levels["FRED:VIXCLS"] = 18.0 + np.abs(levels["FRED:VIXCLS"] - 2.5) * 5.0

    weekly = np.full(N_DAYS, np.nan)
    weekly[::5] = np.cumsum(rng.standard_normal(len(weekly[::5])) * 0.1)
    levels[SPARSE_SERIES] = weekly

    fx = pd.DataFrame({"GBP": 1.3 * np.exp(np.cumsum(
        rng.standard_normal(N_DAYS) * 0.004))}, index=idx)

    cash = levels[seed.CASH_RATE_SERIES]
    return bf.Panels(returns, levels, fx, cash)


def spec(factor_id: str) -> dict:
    return factor_defs.by_id()[factor_id]


def rendered(factor_id: str) -> dict:
    return formula.for_factor(spec(factor_id))


def built(panels: bf.Panels, factor_id: str) -> pd.Series:
    return bf.build_one(panels, spec(factor_id))


def same(a, b, tol: float = 1e-10) -> None:
    """Equal wherever both are defined, and defined on the same days."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape
    ok_a, ok_b = np.isfinite(a), np.isfinite(b)
    assert (ok_a == ok_b).all(), (
        f"defined on different days: builder {ok_a.sum()}, formula {ok_b.sum()}")
    assert np.allclose(a[ok_a], b[ok_b], rtol=0, atol=tol)


# --- helpers written from the formulas, not imported from the builders -----

def cash_leg(panels: bf.Panels) -> np.ndarray:
    """c_{t-1} = DFF_{t-1} / 100 / 252, as the preamble defines it."""
    c = panels.cash.to_numpy() / 100.0 / TRADING_DAYS
    return np.concatenate([[np.nan], c[:-1]])


def modified_duration(T: float, y: np.ndarray) -> np.ndarray:
    """D = (1 - (1 + y/2)^(-2T)) / y, with the y -> 0 limit of T."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (1.0 - (1.0 + y / 2.0) ** (-2.0 * T)) / y
    return np.where(np.abs(y) < 1e-8, T, d)


def convexity(T: float, y: np.ndarray) -> np.ndarray:
    """C = (2/y^2)(1 - disc) - 2T*disc / (y(1 + y/2)), floored at zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = (1.0 + y / 2.0) ** (-2.0 * T)
        c = (2.0 / y**2) * (1.0 - disc) - (2.0 * T * disc) / (y * (1.0 + y / 2.0))
    c = np.where(np.abs(y) < 1e-8, T * (T + 0.5), c)
    return np.maximum(c, 0.0)


def par_bond_return(panels: bf.Panels, series_id: str, T: float,
                    carry: bool = True) -> np.ndarray:
    """b_t = -D_{t-1} dy_t + 0.5 C_{t-1} dy_t^2 + (y_{t-1} - c_{t-1}) / 252."""
    y = panels.level(series_id).to_numpy() / 100.0
    dy = np.diff(y, prepend=np.nan)
    y_prev = np.concatenate([[np.nan], y[:-1]])
    r = (-modified_duration(T, y_prev) * dy
         + 0.5 * convexity(T, y_prev) * dy**2)
    return r + (y_prev / TRADING_DAYS - cash_leg(panels)) if carry else r


def trailing_sd(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    for t in range(x.size):
        w = x[max(0, t - window + 1): t + 1]
        w = w[np.isfinite(w)]
        if w.size >= min_periods:
            sd = np.std(w, ddof=1)
            if sd > 0:
                out[t] = sd
    return out


def rescaled(x: np.ndarray, target_vol: float) -> np.ndarray:
    """f <- k f with k = (target / sqrt(252)) / sd(f), one constant for the sample."""
    sd = np.nanstd(x, ddof=1)
    return x * (target_vol / np.sqrt(TRADING_DAYS) / sd)


# ---------------------------------------------------------------------------
# construction: one test per method, formula against builder
# ---------------------------------------------------------------------------

def test_single_is_the_excess_return(panels):
    """f_t = x_t^ACWI"""
    assert rendered("eq_global")["construction"]["plain"] == "f_t = x_t^ACWI"
    expected = panels.ret("ACWI").to_numpy() - cash_leg(panels)
    same(built(panels, "eq_global").to_numpy(), expected)


def test_single_converts_to_base_currency_by_adding_the_log_fx_change(panels):
    """f_t = (x_t^IGLT.L + dln(e_t^GBP))"""
    plain = rendered("rt_uk")["construction"]["plain"]
    assert plain == "f_t = (x_t^IGLT.L + dln(e_t^GBP))"

    e = panels.fx["GBP"].to_numpy()
    d_ln_e = np.diff(np.log(e), prepend=np.nan)
    expected = panels.ret("IGLT.L").to_numpy() - cash_leg(panels) + d_ln_e
    same(built(panels, "rt_uk").to_numpy(), expected)


def test_single_applies_the_sign_and_skips_the_cash_leg(panels):
    """f_t = -1 * r_t^USDJPY=X -- unfunded, so no cash leg, and flipped."""
    assert rendered("fx_jpy")["construction"]["plain"] == "f_t = -1 * r_t^USDJPY=X"
    same(built(panels, "fx_jpy").to_numpy(), -panels.ret("USDJPY=X").to_numpy())


def test_spread_has_no_cash_leg(panels):
    """f_t = r_t^IWM - r_t^IWB. The funding cancels between the legs."""
    assert rendered("eq_size")["construction"]["plain"] == "f_t = r_t^IWM - r_t^IWB"
    expected = panels.ret("IWM").to_numpy() - panels.ret("IWB").to_numpy()
    same(built(panels, "eq_size").to_numpy(), expected)
    # And the cash leg really is absent: adding it back would move the series.
    assert not np.allclose(expected, expected + cash_leg(panels), equal_nan=True)


def test_basket_is_equal_weighted_on_each_side(panels):
    """f_t = (TLT + UUP + GLD)/3 - (HYG + EEM)/2"""
    plain = rendered("liq_risk_off")["construction"]["plain"]
    assert plain == "f_t = (r_t^TLT + r_t^UUP + r_t^GLD) / 3 - (r_t^HYG + r_t^EEM) / 2"

    longs = sum(panels.ret(t).to_numpy() for t in ("TLT", "UUP", "GLD")) / 3.0
    shorts = sum(panels.ret(t).to_numpy() for t in ("HYG", "EEM")) / 2.0
    same(built(panels, "liq_risk_off").to_numpy(), longs - shorts)


def test_curve_slope_is_duration_neutral_and_scaled_to_five_years(panels):
    """f_t = 5 * (u_t^(10y) - u_t^(2y)), u = b / D."""
    plain = rendered("rt_us_slope")["construction"]["plain"]
    assert plain.splitlines()[0] == "f_t = 5 * (u_t^(10y) - u_t^(2y))"

    members = {tenor: sid for sid, tenor in seed.CURVES["US_TSY"]}
    unit = {}
    for T in (2.0, 10.0):
        y = panels.level(members[T]).to_numpy() / 100.0
        y_prev = np.concatenate([[np.nan], y[:-1]])
        unit[T] = par_bond_return(panels, members[T], T) / modified_duration(T, y_prev)

    expected = formula.CURVE_REF_DURATION * (unit[10.0] - unit[2.0])
    same(built(panels, "rt_us_slope").to_numpy(), expected)


def test_curve_level_averages_the_unit_duration_legs(panels):
    members = {tenor: sid for sid, tenor in seed.CURVES["US_TSY"]}
    tenors = [2.0, 5.0, 10.0, 30.0]
    legs = []
    for T in tenors:
        y = panels.level(members[T]).to_numpy() / 100.0
        y_prev = np.concatenate([[np.nan], y[:-1]])
        legs.append(par_bond_return(panels, members[T], T) / modified_duration(T, y_prev))

    expected = formula.CURVE_REF_DURATION * np.mean(legs, axis=0)
    same(built(panels, "rt_us_level").to_numpy(), expected)


def test_breakeven_carry_terms_cancel_exactly(panels):
    """The rendered formula has no carry term, and that is not an omission.

    The builder adds y_{t-1}/252 and then subtracts it again when carry is off, so
    the factor really is the pure duration-scaled change the formula states.
    """
    plain = rendered("rt_breakeven")["construction"]["plain"]
    assert plain == "f_t = -D_{t-1} * dy_t + 0.5 * C_{t-1} * (dy_t)^2"
    assert "c_{t-1}" not in plain

    expected = par_bond_return(panels, "FRED:T10YIE", 10.0, carry=False)
    same(built(panels, "rt_breakeven").to_numpy(), expected)


def test_spread_level_standardises_on_a_trailing_window(panels):
    """s = SOFR - EFFR; f = ds / trailing sd(ds); then rescaled to 10% vol."""
    plain = rendered("liq_funding")["construction"]["plain"]
    assert plain.splitlines()[0] == "s_t = FRED:SOFR - FRED:EFFR"

    s = (panels.level("FRED:SOFR") - panels.level("FRED:EFFR")).to_numpy()
    ds = np.diff(s, prepend=np.nan)
    expected = rescaled(ds / trailing_sd(ds, 252, 60), factor_defs.STANDARDIZED_TARGET_VOL)
    same(built(panels, "liq_funding").to_numpy(), expected)


def test_sparse_release_factor_is_zero_between_publications(panels):
    """The weekly series becomes an event factor, never a forward fill."""
    plain = rendered("liq_conditions")["construction"]["plain"]
    assert "if t is a release day" in plain
    assert "f_t = 0" in plain

    v = panels.level(SPARSE_SERIES).to_numpy()
    obs = np.flatnonzero(np.isfinite(v))
    changes = np.diff(v[obs], prepend=np.nan)
    scaled = changes / trailing_sd(changes, 52, 12)

    out = np.full(v.size, np.nan)
    first = int(np.flatnonzero(np.isfinite(scaled))[0])
    out[obs[first]:] = 0.0
    for k in range(first, scaled.size):
        if np.isfinite(scaled[k]):
            out[obs[k]] = scaled[k]
    expected = rescaled(out, factor_defs.STANDARDIZED_TARGET_VOL)

    actual = built(panels, "liq_conditions").to_numpy()
    same(actual, expected)

    # And the defining property: zero on the overwhelming majority of days.
    live = actual[np.isfinite(actual)]
    assert (live == 0.0).mean() > 0.7


def test_variance_premium_lags_the_implied_leg(panels):
    """f_t = (VIX_{t-1}/100)^2 / 252 - (r_t^SPY)^2"""
    plain = rendered("vol_variance_premium")["construction"]["plain"]
    assert plain.splitlines()[0] == "f_t = (VIX_{t-1} / 100)^2 / 252 - (r_t^SPY)^2"

    vix = panels.level("FRED:VIXCLS").to_numpy()
    implied = np.concatenate([[np.nan], (vix[:-1] / 100.0) ** 2 / TRADING_DAYS])
    realized = panels.ret("SPY").to_numpy() ** 2
    expected = rescaled(implied - realized, factor_defs.STANDARDIZED_TARGET_VOL)
    same(built(panels, "vol_variance_premium").to_numpy(), expected)


def test_realized_vol_change_differences_an_annualised_volatility(panels):
    plain = rendered("vol_rates")["construction"]["plain"]
    assert plain.splitlines()[0].startswith("v_t = sqrt(252) * sd(r_{t-20}^TLT")

    r = panels.ret("TLT")
    v = (r.rolling(21, min_periods=10).std() * np.sqrt(TRADING_DAYS)).to_numpy()
    dv = np.diff(v, prepend=np.nan)
    expected = rescaled(dv / trailing_sd(dv, 252, 60),
                        factor_defs.STANDARDIZED_TARGET_VOL)
    same(built(panels, "vol_rates").to_numpy(), expected)


def test_tsmom_signal_uses_no_information_from_its_own_day(panels):
    """S from t-252..t-22, v from t-63..t-1, applied to r_t."""
    plain = rendered("arp_trend")["construction"]["plain"]
    assert "S_it = sign( sum of r_iu for u = t-252 .. t-22 )" in plain

    universe = [u for u in spec("arp_trend")["inputs"]["universe"]
                if u in panels.returns.columns]
    R = panels.returns[universe]

    signal = np.sign(R.shift(22).rolling(231, min_periods=115).sum())
    vol = R.shift(1).rolling(63, min_periods=31).std()
    w = signal / vol.replace(0.0, np.nan)
    expected = ((w * R).sum(axis=1) / w.abs().sum(axis=1).replace(0.0, np.nan))

    same(built(panels, "arp_trend").to_numpy(), expected.to_numpy())


def test_xs_momentum_is_dollar_neutral(panels):
    plain = rendered("arp_xs_momentum")["construction"]["plain"]
    assert "w_it = 1{q_it > 2/3} / n_long  -  1{q_it < 1/3} / n_short" in plain

    universe = [u for u in spec("arp_xs_momentum")["inputs"]["universe"]
                if u in panels.returns.columns]
    R = panels.returns[universe]

    ranks = R.shift(22).rolling(231, min_periods=115).sum().rank(axis=1, pct=True)
    long_leg = (ranks > 2 / 3).astype(float)
    short_leg = (ranks < 1 / 3).astype(float)
    n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
    n_short = short_leg.sum(axis=1).replace(0.0, np.nan)
    w = long_leg.div(n_long, axis=0) - short_leg.div(n_short, axis=0)
    expected = (w * R).sum(axis=1).where(n_long.notna() & n_short.notna())

    same(built(panels, "arp_xs_momentum").to_numpy(), expected.to_numpy())
    # Weights net to zero by construction, which is what "dollar-neutral" means.
    live = w.dropna(how="all")
    assert np.allclose(live.sum(axis=1).dropna().to_numpy(), 0.0, atol=1e-12)


def test_fx_carry_lags_the_rate_differential(panels):
    plain = rendered("fx_carry")["construction"]["plain"]
    assert "k_jt   = (i_j,t-1 - i_USD,t-1) / 100 / 252" in plain

    inputs = spec("fx_carry")["inputs"]
    inverted = set(inputs["inverted"])
    usd = panels.level(seed.POLICY_RATES["USD"]).to_numpy()

    legs = []
    for ccy, ticker in inputs["pairs"].items():
        foreign = panels.level(seed.POLICY_RATES[ccy]).to_numpy()
        diff = (foreign - usd) / 100.0 / TRADING_DAYS
        k = np.concatenate([[np.nan], diff[:-1]])
        s = -1.0 if ccy in inverted else 1.0
        legs.append(np.sign(k) * (s * panels.ret(ticker).to_numpy() + k))

    same(built(panels, "fx_carry").to_numpy(), np.nanmean(np.array(legs), axis=0))


# ---------------------------------------------------------------------------
# orthogonalisation
# ---------------------------------------------------------------------------

def test_rolling_orthogonalisation_matches_the_stated_argmin():
    """Implement the rendered equation directly and reproduce the builder's residual.

    f~_t = f_t - a_tau - sum_j b_j,tau f~_t^gj, coefficients from least squares over
    s = tau-504 .. tau-1, refitted every 21 observations from the 252nd.
    """
    rng = np.random.default_rng(SEED)
    n = 1500
    g1 = rng.standard_normal(n) * 0.01
    g2 = rng.standard_normal(n) * 0.008
    f = 0.8 * g1 - 0.4 * g2 + rng.standard_normal(n) * 0.004
    X = np.column_stack([g1, g2])

    expected = np.full(n, np.nan)
    coef = None
    next_fit = formula.ORTH_MIN_OBS
    for t in range(n):
        if t >= next_fit:
            lo = max(0, t - formula.ORTH_WINDOW)
            ys, xs = f[lo:t], X[lo:t]
            if ys.size >= max(formula.ORTH_MIN_OBS, X.shape[1] + 2):
                design = np.column_stack([np.ones(ys.size), xs])
                coef = np.linalg.lstsq(design, ys, rcond=None)[0]
            next_fit = t + formula.ORTH_REFIT_EVERY
        if coef is not None:
            expected[t] = f[t] - coef[0] - X[t] @ coef[1:]

    same(og.orthogonalize(f, X, mode="rolling"), expected, tol=1e-12)


def test_a_factor_with_no_targets_is_identical_in_both_panels():
    """The claim the raw/orthogonalised comparison rests on for level-0 factors."""
    f = formula.orthogonalisation([])
    assert f.plain == "f~_t = f_t"

    rng = np.random.default_rng(SEED)
    y = rng.standard_normal(500)
    assert np.array_equal(og.orthogonalize(y, np.empty((500, 0))), y)


def test_orthogonalisation_names_the_targets_in_hierarchy_order():
    f = formula.orthogonalisation(["rt_us_level", "rt_us_slope", "eq_global"])
    assert "b1_tau * f~_t^rt_us_level" in f.plain
    assert "b2_tau * f~_t^rt_us_slope" in f.plain
    assert "b3_tau * f~_t^eq_global" in f.plain
    # The regressors are the targets' orthogonalised series, not their raw ones.
    assert any("*orthogonalised* series of rt_us_level" in s.meaning for s in f.where)


def test_orthogonalisation_modes_state_their_own_window():
    assert "tau-504" in formula.orthogonalisation(["eq_global"], "rolling").plain
    assert "every s < tau" in formula.orthogonalisation(["eq_global"], "expanding").plain
    assert "whole sample" in formula.orthogonalisation(["eq_global"], "full_sample").plain


# ---------------------------------------------------------------------------
# registry-wide structure
# ---------------------------------------------------------------------------

def test_every_factor_renders_both_formulas():
    all_rendered = formula.all_factors()
    assert len(all_rendered) == len(factor_defs.FACTORS)
    for fid, r in all_rendered.items():
        assert r["construction"]["plain"].startswith(("f_t", "s_t", "v_t", "S_it",
                                                      "M_it", "k_jt")), fid
        assert r["construction"]["latex"], fid
        assert r["orthogonalisation"]["plain"], fid
        assert r["construction"]["steps"], fid


def test_every_symbol_that_claims_a_source_names_one_the_builder_reads():
    """A formula that points at an instrument the factor does not touch is worse than
    no formula, because it looks authoritative.

    The check is against the factor's own construction inputs plus the constants the
    builder reaches for on its own -- the cash rate, the VIX level, the curve members
    behind a tenor list, the policy rates behind an FX pair. Nothing else may appear.
    """
    known_factors = {f["id"] for f in factor_defs.FACTORS}
    ambient_levels = ({seed.CASH_RATE_SERIES, "FRED:VIXCLS"}
                      | set(seed.POLICY_RATES.values())
                      | {sid for members in seed.CURVES.values()
                         for sid, _ in members})

    for f in factor_defs.FACTORS:
        declared: list[str] = []
        _collect(f["inputs"], declared)
        allowed = set(declared) | ambient_levels
        r = formula.for_factor(f)

        for block in ("construction", "orthogonalisation"):
            for sym in r[block]["where"]:
                ref, kind = sym["ref"], sym["kind"]
                if ref is None:
                    continue
                if kind == "factor":
                    assert ref in known_factors, f"{f['id']}: unknown factor {ref}"
                elif kind == "level" and len(ref) == 3:
                    # A currency code: FX lives in fact_input_fx, not as a level series.
                    assert ref == f["inputs"].get("fx"), f"{f['id']}: stray ccy {ref}"
                else:
                    assert ref in allowed, f"{f['id']}: {kind} {ref} is not an input"


def _collect(node, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for v in node:
            _collect(v, out)
    elif isinstance(node, dict):
        for v in node.values():
            _collect(v, out)


def test_orthogonalisation_matches_the_registry_for_every_factor():
    for f in factor_defs.FACTORS:
        r = formula.for_factor(f)
        targets = f.get("orth", [])
        assert r["orthogonalised_against"] == list(targets), f["id"]
        for g in targets:
            assert g in r["orthogonalisation"]["plain"], f"{f['id']} omits {g}"


def test_for_factor_accepts_a_database_row_shape():
    """The API passes a parsed ref_factor row, not a factor_defs entry."""
    row = {
        "factor_id": "eq_size",
        "construction": {"method": "spread", "inputs": {"long": "IWM", "short": "IWB"}},
        "orthogonalize_against": ["eq_global"],
    }
    assert (formula.for_factor(row)["construction"]["plain"]
            == formula.for_factor(spec("eq_size"))["construction"]["plain"])
    assert formula.for_factor(row)["orthogonalised_against"] == ["eq_global"]


def test_plain_rendering_is_ascii_so_it_survives_a_windows_console():
    for fid, r in formula.all_factors().items():
        for block in ("construction", "orthogonalisation"):
            text = r[block]["plain"] + " ".join(r[block]["steps"])
            text.encode("cp1252")  # raises if a stray minus sign crept in
            assert text.isascii(), fid


def test_latex_escapes_the_underscores_in_identifiers():
    r"""An unescaped underscore inside \text{} is a subscript, and KaTeX throws on it.

    Factor ids and several level-series ids carry them: eq_global, ECB:BUND_10Y,
    SNB:POLICY_RATE. Unescaped, the write-up renders BUND with a subscript and the
    profile popout shows a red error where the formula should be.
    """
    import re

    for fid, r in formula.all_factors().items():
        for block in ("construction", "orthogonalisation"):
            for body in re.findall(r"\\text\{(.*?)\}", r[block]["latex"]):
                assert "_" not in body.replace(r"\_", ""), f"{fid}: bare _ in {body}"


def test_latex_braces_balance():
    for fid, r in formula.all_factors().items():
        for block in ("construction", "orthogonalisation"):
            latex = r[block]["latex"]
            depth = 0
            for i, ch in enumerate(latex):
                if ch == "{" and (i == 0 or latex[i - 1] != "\\"):
                    depth += 1
                elif ch == "}" and (i == 0 or latex[i - 1] != "\\"):
                    depth -= 1
                assert depth >= 0, f"{fid} ({block}): unbalanced closing brace"
            assert depth == 0, f"{fid} ({block}): {depth} unclosed brace(s)"


def test_an_unknown_method_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="no formula renderer"):
        formula.construction("teleology", {})
