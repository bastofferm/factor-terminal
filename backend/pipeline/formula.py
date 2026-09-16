"""The exact arithmetic behind every factor, rendered as a formula.

A factor definition in `factor_defs` is a method name and a bag of inputs. That is
enough to build the series but not enough to read it: `{"method": "curve",
"inputs": {"curve": "US_TSY", "shape": "slope", "tenors": [2, 10]}}` does not tell
anyone that the factor is a duration-neutral steepener whose two legs are divided
by their own modified durations and then scaled back to five years.

This module turns that bag of inputs into the formula the builder actually
evaluates, symbol by symbol, plus the exact definition of the residualisation that
turns the raw factor into the orthogonalised one. It is transcribed from
`build_factors.py`, `core/transforms.py` and `core/orthogonalize.py` rather than
written from memory, and `backend/tests/test_formula.py` checks the transcription
by evaluating each rendered formula independently against the builder's own output
on simulated inputs.

Two renderings of the same thing:

    latex  for the write-up under Documentation/, where GitHub renders $$...$$
    plain  ASCII, for the terminal UI, the chatbot corpus and the CLI

Plain is deliberately ASCII-only. These strings reach a Windows console through the
pipeline CLIs, where a minus sign or a sigma raises UnicodeEncodeError under the
cp1252 default, and a formula is not worth crashing a build job for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.pipeline import factor_defs, seed

# Mirrors of the constants the builders use, imported from where they live rather
# than restated, so a change to one cannot leave the documentation behind.
TRADING_DAYS = 252
CURVE_REF_DURATION = factor_defs.CURVE_REF_DURATION
CASH_RATE_SERIES = seed.CASH_RATE_SERIES

# core/orthogonalize.py::orthogonalize defaults, as build_factors calls them.
ORTH_WINDOW = 504
ORTH_MIN_OBS = 252
ORTH_REFIT_EVERY = 21


@dataclass(frozen=True)
class Symbol:
    """One symbol in a formula and the concrete thing it stands for."""

    plain: str
    latex: str
    meaning: str
    kind: str                 # instrument | level | curve | factor | constant
    ref: str | None = None    # instrument_id or series_id, where one exists


@dataclass(frozen=True)
class Formula:
    plain: str
    latex: str
    where: list[Symbol] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "plain": self.plain,
            "latex": self.latex,
            "where": [s.__dict__ for s in self.where],
            "steps": list(self.steps),
        }


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------

# Every instrument return in the model is a log total return, and every excess
# return subtracts *yesterday's* overnight rate, because the rate earned over day t
# was set at t-1 (core/transforms.py::to_excess_return).
PREAMBLE: list[Symbol] = [
    Symbol("r_t^X", r"r^X_t",
           "log total return of instrument X: ln(P_t / P_{t-1}) on a price series "
           "already adjusted for dividends and splits", "constant"),
    Symbol("c_t", r"c_t",
           f"daily cash rate: {CASH_RATE_SERIES} at t, quoted as an annualised "
           f"percent, divided by 100 and by {TRADING_DAYS}", "level",
           CASH_RATE_SERIES),
    Symbol("x_t^X", r"x^X_t",
           "excess return of X over cash: x_t^X = r_t^X - c_{t-1}", "constant"),
    Symbol("f_t", r"f_t", "the raw factor, before any orthogonalisation", "constant"),
    Symbol("f~_t", r"\tilde{f}_t",
           "the orthogonalised factor, which is what the model consumes", "constant"),
]

_CASH = Symbol("c_t", r"c_t",
               f"daily cash rate: {CASH_RATE_SERIES}_t / 100 / {TRADING_DAYS}, lagged "
               f"one day because the overnight rate earned on day t is set at t-1",
               "level", CASH_RATE_SERIES)


def _tex(name: str) -> str:
    r"""Escape an identifier for \text{}.

    Factor ids and level-series ids carry underscores -- eq_global, ECB:BUND_10Y --
    and an unescaped underscore inside math mode is a subscript, so KaTeX either
    throws or silently renders `BUND` with a subscript. Everything else in these
    identifiers (:, ., -, =, ^) is safe in \text{}.
    """
    return name.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _instrument(ticker: str, role: str = "instrument return") -> Symbol:
    return Symbol(f"r_t^{ticker}", rf"r^{{\text{{{_tex(ticker)}}}}}_t",
                  f"{role}: log total return of {ticker}", "instrument", ticker)


def _level(series_id: str, meaning: str) -> Symbol:
    return Symbol(series_id, rf"\text{{{_tex(series_id)}}}", meaning, "level",
                  series_id)


def _duration_symbols(curve_label: str) -> list[Symbol]:
    return [
        Symbol("y_t^(T)", r"y^{(T)}_t",
               f"par yield of the {curve_label} curve at tenor T, in decimal "
               f"(the quoted percent divided by 100)", "curve"),
        Symbol("D_t^(T)", r"D^{(T)}_t",
               "modified duration of a par bond: (1 - (1 + y/2)^(-2T)) / y, "
               "evaluated at yesterday's yield so the return at t uses only "
               "information available at t-1", "constant"),
        Symbol("C_t^(T)", r"C^{(T)}_t",
               "convexity of the same par bond: (2/y^2)(1 - (1+y/2)^(-2T)) "
               "- 2T(1+y/2)^(-2T) / (y(1+y/2))", "constant"),
    ]


_BOND_RETURN_PLAIN = ("b_t^(T) = -D_{t-1}^(T) * dy_t^(T) "
                      "+ 0.5 * C_{t-1}^(T) * (dy_t^(T))^2 "
                      f"+ (y_{{t-1}}^(T) - c_{{t-1}}) / {TRADING_DAYS}")
_BOND_RETURN_LATEX = (r"b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t "
                      r"+ \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2 "
                      r"+ \frac{y^{(T)}_{t-1} - c_{t-1}}{252}")


def _rescale_note(target_vol: float) -> tuple[str, str, Symbol, str]:
    """The constant that puts a z-scored series back on a return scale.

    One number for the whole sample, which is why it is a units convention and not
    lookahead: it maps beta -> beta/k and sigma -> k*sigma and leaves every
    variance, correlation, t-statistic and R-squared unchanged
    (build_factors.py::_rescale).
    """
    plain = f"f_t <- k * f_t,   k = ({target_vol:g} / sqrt({TRADING_DAYS})) / sd(f)"
    latex = (rf"f_t \leftarrow k\,f_t, \qquad "
             rf"k = \frac{{{target_vol:g} / \sqrt{{252}}}}{{\operatorname{{sd}}(f)}}")
    sym = Symbol("k", "k",
                 f"one constant for the whole sample, chosen so the factor's "
                 f"annualised volatility is {target_vol:.0%}. A constant rescaling "
                 f"cannot change any risk number; it only puts this factor's betas "
                 f"on the same readable scale as the return-based blocks",
                 "constant")
    step = (f"Rescale to {target_vol:.0%} annualised volatility. The transform above "
            f"yields a z-score, and every factor in this model is a return.")
    return plain, latex, sym, step


# ---------------------------------------------------------------------------
# one renderer per construction method
# ---------------------------------------------------------------------------

def _f_single(inputs: dict) -> Formula:
    inst = inputs["instrument"]
    sign = float(inputs.get("sign", 1))
    excess = bool(inputs.get("excess", True))
    ccy = inputs.get("fx")

    core_plain = f"x_t^{inst}" if excess else f"r_t^{inst}"
    core_latex = (rf"x^{{\text{{{_tex(inst)}}}}}_t" if excess
                  else rf"r^{{\text{{{_tex(inst)}}}}}_t")

    where = [_instrument(inst)]
    steps: list[str] = []

    if excess:
        where.append(_CASH)
        steps.append(f"Subtract the overnight rate: x_t = r_t - c_(t-1). {inst} is a "
                     f"funded long position, so its return has to be measured over "
                     f"cash.")
    else:
        steps.append("No cash leg: this is not a funded position, so there is nothing "
                     "to fund.")

    if ccy:
        core_plain = f"({core_plain} + dln(e_t^{ccy}))"
        core_latex = (rf"\bigl({core_latex} "
                      rf"+ \Delta \ln e^{{\text{{{_tex(ccy)}}}}}_t\bigr)")
        where.append(Symbol(f"e_t^{ccy}", rf"e^{{\text{{{_tex(ccy)}}}}}_t",
                            f"USD per unit of {ccy}; its log change converts the local "
                            f"return into the base currency, which is exact in logs",
                            "level", ccy))
        steps.append(f"Convert to the base currency by adding the log change in "
                     f"{ccy}/USD. In logs the conversion is a sum, with no cross term.")

    if sign != 1:
        plain = f"f_t = {sign:g} * {core_plain}"
        latex = rf"f_t = {sign:g}\,{core_latex}"
        steps.append(f"Multiply by {sign:g} so the factor points in the stated "
                     f"direction.")
    else:
        plain = f"f_t = {core_plain}"
        latex = rf"f_t = {core_latex}"

    return Formula(plain, latex, where, steps)


def _f_spread(inputs: dict) -> Formula:
    long, short = inputs["long"], inputs["short"]
    return Formula(
        f"f_t = r_t^{long} - r_t^{short}",
        rf"f_t = r^{{\text{{{_tex(long)}}}}}_t - r^{{\text{{{_tex(short)}}}}}_t",
        [_instrument(long, "long leg"), _instrument(short, "short leg")],
        ["Self-financing, so no cash leg: the funding cost cancels between the two "
         "legs and subtracting it would understate the spread."],
    )


def _f_basket(inputs: dict) -> Formula:
    longs = list(inputs["long"])
    shorts = list(inputs.get("short", []))

    lp = " + ".join(f"r_t^{i}" for i in longs)
    ll = " + ".join(rf"r^{{\text{{{_tex(i)}}}}}_t" for i in longs)
    plain = f"f_t = ({lp}) / {len(longs)}"
    latex = rf"f_t = \frac{{{ll}}}{{{len(longs)}}}"

    where = [_instrument(i, "long leg") for i in longs]
    steps = [f"Equally weighted long basket of {len(longs)}: each leg carries "
             f"1/{len(longs)} of the notional."]

    if shorts:
        sp = " + ".join(f"r_t^{i}" for i in shorts)
        sl = " + ".join(rf"r^{{\text{{{_tex(i)}}}}}_t" for i in shorts)
        plain += f" - ({sp}) / {len(shorts)}"
        latex += rf" - \frac{{{sl}}}{{{len(shorts)}}}"
        where += [_instrument(i, "short leg") for i in shorts]
        steps.append(f"Minus an equally weighted short basket of {len(shorts)}. Both "
                     f"sides carry the market direction, so it cancels and what "
                     f"survives is the spread between them.")
        steps.append("Self-financing, so no cash leg.")

    return Formula(plain, latex, where, steps)


def _f_curve(inputs: dict) -> Formula:
    curve = inputs["curve"]
    shape = inputs["shape"]
    tenors = sorted(float(t) for t in inputs["tenors"])
    members = {tenor: sid for sid, tenor in seed.CURVES[curve]}
    k = CURVE_REF_DURATION

    if shape == "level":
        shape_plain = f"({' + '.join(f'u_t^({t:g}y)' for t in tenors)}) / {len(tenors)}"
        shape_latex = rf"\frac{{1}}{{{len(tenors)}}}\sum_{{T}} u^{{(T)}}_t"
        shape_step = (f"Level: an equal unit-duration long at every tenor "
                      f"({', '.join(f'{t:g}y' for t in tenors)}).")
    elif shape == "slope":
        lo, hi = tenors[0], tenors[-1]
        shape_plain = f"(u_t^({hi:g}y) - u_t^({lo:g}y))"
        shape_latex = rf"\bigl(u^{{({hi:g}y)}}_t - u^{{({lo:g}y)}}_t\bigr)"
        shape_step = (f"Slope: long {hi:g}y against short {lo:g}y in equal units of "
                      f"duration, so a parallel shift nets to zero and only the "
                      f"steepening survives.")
    elif shape == "curvature":
        lo, belly, hi = tenors[0], tenors[1], tenors[-1]
        shape_plain = f"(2 * u_t^({belly:g}y) - u_t^({lo:g}y) - u_t^({hi:g}y))"
        shape_latex = (rf"\bigl(2u^{{({belly:g}y)}}_t - u^{{({lo:g}y)}}_t "
                       rf"- u^{{({hi:g}y)}}_t\bigr)")
        shape_step = (f"Curvature: long two units of the {belly:g}y belly against one "
                      f"unit each of the {lo:g}y and {hi:g}y wings. Neutral to both a "
                      f"parallel shift and a steepening.")
    else:  # pragma: no cover - factor_defs.validate would have caught it
        raise ValueError(f"unknown curve shape {shape!r}")

    plain = (f"f_t = {k:g} * {shape_plain}\n"
             f"u_t^(T) = b_t^(T) / D_{{t-1}}^(T)\n"
             f"{_BOND_RETURN_PLAIN}")
    latex = (rf"f_t = {k:g} \cdot {shape_latex}, \qquad "
             rf"u^{{(T)}}_t = \frac{{b^{{(T)}}_t}}{{D^{{(T)}}_{{t-1}}}}"
             "\n\n" + _BOND_RETURN_LATEX)

    where = _duration_symbols(curve) + [
        Symbol("b_t^(T)", r"b^{(T)}_t",
               "synthetic par-bond excess return at tenor T", "constant"),
        Symbol("u_t^(T)", r"u^{(T)}_t",
               "the same return per year of duration, which is what makes the legs "
               "comparable across tenors", "constant"),
        Symbol(f"{k:g}", f"{k:g}",
               f"reference duration in years. The shape is formed in unit-duration "
               f"terms and then scaled back to {k:g}y, roughly the duration of a broad "
               f"aggregate index, so the factor comes out at a familiar magnitude",
               "constant"),
        _CASH,
    ] + [_level(members[t], f"{curve} par yield at {t:g}y") for t in tenors
         if t in members]

    steps = [
        "Convert each tenor's yield change into a bond return, with duration and "
        "convexity evaluated at yesterday's yield. Section 2.2 requires every factor "
        "to be a return, and a yield change is not one.",
        "Divide each leg by its own modified duration. Five basis points on a 30y is "
        "not the same trade as five on a 2y, and without this the long end would "
        "dominate every shape.",
        shape_step,
        f"Scale back by {k:g} years of duration.",
    ]
    return Formula(plain, latex, where, steps)


def _f_synthetic_bond(inputs: dict) -> Formula:
    series = inputs["series"]
    tenor = float(inputs["tenor"])
    carry = bool(inputs.get("carry", True))

    if carry:
        plain = (f"f_t = -D_{{t-1}} * dy_t + 0.5 * C_{{t-1}} * (dy_t)^2 "
                 f"+ (y_{{t-1}} - c_{{t-1}}) / {TRADING_DAYS}")
        latex = (r"f_t = -D_{t-1}\,\Delta y_t "
                 r"+ \tfrac{1}{2} C_{t-1}\,(\Delta y_t)^2 "
                 r"+ \frac{y_{t-1} - c_{t-1}}{252}")
        steps = [f"Carry is the {tenor:g}y yield earned for a day, less cash, so the "
                 f"factor is an excess return."]
        where_extra = [_CASH]
    else:
        plain = "f_t = -D_{t-1} * dy_t + 0.5 * C_{t-1} * (dy_t)^2"
        latex = (r"f_t = -D_{t-1}\,\Delta y_t "
                 r"+ \tfrac{1}{2} C_{t-1}\,(\Delta y_t)^2")
        steps = ["No carry leg. A breakeven is the difference between two yields and "
                 "earns no coupon of its own, so the builder adds the carry term and "
                 "subtracts it again; the two cancel exactly and the factor is a pure "
                 "duration-scaled change."]
        where_extra = []

    where = [
        _level(series, "quoted yield series, in percent; y_t is that divided by 100"),
        Symbol("D_{t-1}", r"D_{t-1}",
               f"modified duration of a {tenor:g}y par bond at yesterday's yield: "
               f"(1 - (1 + y/2)^(-2*{tenor:g})) / y", "constant"),
        Symbol("C_{t-1}", r"C_{t-1}",
               f"convexity of the same {tenor:g}y par bond", "constant"),
    ] + where_extra

    steps.insert(0, f"Treat the series as the yield of a {tenor:g}y par bond and price "
                    f"a day's move on it, with duration and convexity taken at t-1 so "
                    f"nothing about today's move leaks into its own pricing.")
    return Formula(plain, latex, where, steps)


def _f_spread_level(inputs: dict) -> Formula:
    a, b = inputs["minuend"], inputs["subtrahend"]
    transform = inputs.get("transform", "diff")
    plain = f"s_t = {a} - {b}\n"
    latex = rf"s_t = \text{{{_tex(a)}}}_t - \text{{{_tex(b)}}}_t"

    if transform == "diff_std":
        plain += (f"f_t = (s_t - s_{{t-1}}) / sd(ds over the trailing {TRADING_DAYS} "
                  f"observations, at least 60)")
        latex += ("\n\n" + r"f_t = \frac{\Delta s_t}"
                  r"{\operatorname{sd}\bigl(\Delta s_{t-251},\dots,\Delta s_t\bigr)}")
        step = ("Divide the daily change by its own trailing volatility. Basis points "
                "are not comparable with the return-scaled blocks, and the scaling "
                "window is strictly trailing: a full-sample standard deviation would "
                "leak future volatility into every historical observation.")
    else:
        plain += "f_t = s_t - s_{t-1}"
        latex += "\n\n" + r"f_t = \Delta s_t"
        step = "Difference the spread: the level itself is not stationary."

    return Formula(plain, latex,
                   [_level(a, "first leg of the spread, in percent"),
                    _level(b, "second leg of the spread, in percent")],
                   ["Form the spread between the two quoted levels.", step])


def _f_level_transform(inputs: dict) -> Formula:
    series = inputs["series"]
    transform = inputs.get("transform", "diff")
    sparse = bool(inputs.get("sparse"))

    if sparse:
        plain = ("f_t = ds_k / sd(ds over the last 52 releases, at least 12)"
                 "    if t is a release day\n"
                 "f_t = 0"
                 "                                                       otherwise\n"
                 "ds_k = s_k - s_{k-1}, over consecutive releases, not consecutive days")
        latex = (r"f_t = \begin{cases}"
                 r"\dfrac{\Delta s_k}"
                 r"{\operatorname{sd}(\Delta s_{k-51},\dots,\Delta s_k)}"
                 r" & t = \tau_k,\ \text{a release day} \\[2ex]"
                 r"0 & \text{otherwise}\end{cases}")
        steps = [
            "The series prints weekly, not daily.",
            "It is NOT forward filled. A filled series carries no new information "
            "between releases, which manufactures autocorrelation, understates "
            "standard errors and can smuggle in lookahead.",
            "Instead the standardised change lands on the publication day and the "
            "factor is exactly zero in between: a release-event factor, sparse "
            "daily.",
            "The scaling window counts releases rather than days, so 52 of them is "
            "about a year of the series' own history.",
        ]
    elif transform == "diff_std":
        plain = (f"f_t = (s_t - s_{{t-1}}) / sd(ds over the trailing {TRADING_DAYS} "
                 f"observations, at least 60)")
        latex = (r"f_t = \frac{\Delta s_t}"
                 r"{\operatorname{sd}(\Delta s_{t-251},\dots,\Delta s_t)}")
        steps = ["Difference the level and divide by its own trailing volatility."]
    elif transform == "log_diff":
        plain = "f_t = ln(s_t) - ln(s_{t-1})"
        latex = r"f_t = \Delta \ln s_t"
        steps = ["Log-difference the level."]
    else:
        plain = "f_t = s_t - s_{t-1}"
        latex = r"f_t = \Delta s_t"
        steps = ["Difference the level: the level itself is not stationary."]

    return Formula(plain, latex, [_level(series, "the published level series")], steps)


def _f_variance_premium(inputs: dict) -> Formula:
    under = inputs["underlying"]
    plain = f"f_t = (VIX_{{t-1}} / 100)^2 / {TRADING_DAYS} - (r_t^{under})^2"
    latex = (r"f_t = \frac{1}{252}\left(\frac{\text{VIX}_{t-1}}{100}\right)^2 "
             rf"- \bigl(r^{{\text{{{_tex(under)}}}}}_t\bigr)^2")
    return Formula(
        plain, latex,
        [_level("FRED:VIXCLS", "VIX close in index points; divided by 100 it is an "
                               "annualised volatility in decimal"),
         _instrument(under, "underlying whose realised variance is delivered")],
        ["Yesterday's implied variance is what a variance swap struck at t-1 pays "
         "against, so the lag is the contract and not a modelling choice.",
         f"Today's realised variance is the squared {under} return, which is what the "
         f"daily leg of such a swap settles on.",
         "The difference is the daily payoff of a short variance position, positive on "
         "average because implied sits above realised.",
         "The payoff is convex in the underlying, so a linear beta on it is a first "
         "approximation and nothing more."],
    )


def _f_realized_vol_change(inputs: dict) -> Formula:
    inst = inputs["instrument"]
    window = int(inputs.get("window", 21))
    plain = (f"v_t = sqrt({TRADING_DAYS}) * sd(r_{{t-{window - 1}}}^{inst}, ..., "
             f"r_t^{inst})\n"
             f"f_t = (v_t - v_{{t-1}}) / sd(dv over the trailing {TRADING_DAYS} "
             f"observations, at least 60)")
    latex = (rf"v_t = \sqrt{{252}}\;\operatorname{{sd}}"
             rf"\bigl(r^{{\text{{{_tex(inst)}}}}}_{{t-{window - 1}}},\dots,"
             rf"r^{{\text{{{_tex(inst)}}}}}_t\bigr)"
             "\n\n"
             r"f_t = \frac{\Delta v_t}"
             r"{\operatorname{sd}(\Delta v_{t-251},\dots,\Delta v_t)}")
    return Formula(
        plain, latex,
        [_instrument(inst, "instrument whose realised volatility is tracked")],
        [f"Realised volatility over a trailing {window} days, annualised.",
         "Differenced, because a volatility level is not stationary.",
         "Standardised on a trailing window, because a change in volatility is not "
         "measured in return units.",
         "This is a change in a risk measure, not a tradable return. It stands in for "
         "a rates-volatility series because CBOE discontinued ^TYVIX in 2020."],
    )


def _f_tsmom(inputs: dict) -> Formula:
    universe = list(inputs["universe"])
    lookback = int(inputs.get("lookback", 252))
    skip = int(inputs.get("skip", 21))
    vol_window = int(inputs.get("vol_window", 63))

    plain = (
        f"S_it = sign( sum of r_iu for u = t-{lookback} .. t-{skip + 1} )\n"
        f"v_it = sd( r_i,t-{vol_window} , ... , r_i,t-1 )\n"
        f"w_it = S_it / v_it\n"
        f"f_t  = sum_i (w_it * r_it) / sum_i |w_it|"
    )
    latex = (
        rf"S_{{i,t}} = \operatorname{{sign}}"
        rf"\Bigl(\sum_{{u=t-{lookback}}}^{{t-{skip + 1}}} r_{{i,u}}\Bigr), \qquad "
        rf"v_{{i,t}} = \operatorname{{sd}}\bigl(r_{{i,t-{vol_window}}},\dots,"
        rf"r_{{i,t-1}}\bigr)"
        "\n\n"
        r"w_{i,t} = \frac{S_{i,t}}{v_{i,t}}, \qquad "
        r"f_t = \frac{\sum_i w_{i,t}\,r_{i,t}}{\sum_i \lvert w_{i,t}\rvert}"
    )
    return Formula(
        plain, latex,
        [_instrument(u) for u in universe] + [
            Symbol("S_it", r"S_{i,t}",
                   "position sign: +1 if the trailing return was positive, -1 if "
                   "negative", "constant"),
            Symbol("v_it", r"v_{i,t}",
                   f"trailing {vol_window}-day volatility, used to size the position",
                   "constant"),
        ],
        [f"The signal at t reads returns through t-{skip + 1} only, and the position "
         f"it implies is applied to the return at t. Nothing about day t enters its "
         f"own weight.",
         f"Skipping the most recent {skip} days is the standard "
         f"{lookback // 21}-minus-1 construction, which keeps short-term reversal out "
         f"of the trend signal.",
         f"Inverse-volatility sizing puts each of the {len(universe)} instruments on a "
         f"comparable risk footing; without it the equity legs would drown out the "
         f"bond legs.",
         "Dividing by gross exposure makes the factor a return on one unit of notional "
         "rather than a sum that grows with the size of the universe."],
    )


def _f_xs_momentum(inputs: dict) -> Formula:
    universe = list(inputs["universe"])
    lookback = int(inputs.get("lookback", 252))
    skip = int(inputs.get("skip", 21))

    plain = (
        f"M_it = sum of r_iu for u = t-{lookback} .. t-{skip + 1}\n"
        "q_it = cross-sectional percentile rank of M_it within the universe at t\n"
        "w_it = 1{q_it > 2/3} / n_long  -  1{q_it < 1/3} / n_short\n"
        "f_t  = sum_i w_it * r_it"
    )
    latex = (
        rf"M_{{i,t}} = \sum_{{u=t-{lookback}}}^{{t-{skip + 1}}} r_{{i,u}}, \qquad "
        r"q_{i,t} = \operatorname{rank}_\text{pct}\bigl(M_{\cdot,t}\bigr)_i"
        "\n\n"
        r"w_{i,t} = \frac{\mathbf{1}\{q_{i,t} > 2/3\}}{n^{\text{long}}_t} - "
        r"\frac{\mathbf{1}\{q_{i,t} < 1/3\}}{n^{\text{short}}_t}, \qquad "
        r"f_t = \sum_i w_{i,t}\,r_{i,t}"
    )
    return Formula(
        plain, latex,
        [_instrument(u) for u in universe] + [
            Symbol("q_it", r"q_{i,t}",
                   "percentile rank of the trailing return within the universe on "
                   "that day", "constant"),
        ],
        [f"Rank the {len(universe)} instruments by trailing return, again skipping the "
         f"most recent {skip} days.",
         "Long the top third, short the bottom third, equally weighted within each "
         "leg.",
         "Dollar-neutral by construction: the long and the short weights each sum to "
         "one, so the common market direction cancels and what is left is the spread "
         "between winners and losers."],
    )


def _f_fx_carry(inputs: dict) -> Formula:
    pairs: dict[str, str] = dict(inputs["pairs"])
    inverted = set(inputs.get("inverted", []))
    usd_rate = seed.POLICY_RATES["USD"]

    legs = ", ".join(sorted(pairs))
    plain = (
        f"k_jt   = (i_j,t-1 - i_USD,t-1) / 100 / {TRADING_DAYS}\n"
        "p_jt   = sign(k_jt)\n"
        "leg_jt = p_jt * (s_j * r_t^Xj + k_jt)\n"
        f"f_t    = (1/{len(pairs)}) * sum over j of leg_jt"
    )
    latex = (
        r"\kappa_{j,t} = \frac{i_{j,t-1} - i_{\text{USD},t-1}}{100 \cdot 252}, \qquad "
        r"p_{j,t} = \operatorname{sign}(\kappa_{j,t})"
        "\n\n"
        rf"f_t = \frac{{1}}{{{len(pairs)}}}\sum_j p_{{j,t}}"
        r"\bigl(s_j\,r^{X_j}_t + \kappa_{j,t}\bigr)"
    )

    where: list[Symbol] = []
    for ccy, ticker in sorted(pairs.items()):
        sid = seed.POLICY_RATES.get(ccy)
        flipped = ccy in inverted
        where.append(_instrument(
            ticker,
            f"{ccy} spot leg" + (", sign flipped so the quoted pair reads as long the "
                                 "foreign currency against the dollar" if flipped
                                 else "")))
        if sid:
            where.append(_level(sid, f"{ccy} policy rate i_{ccy}, annualised percent"))
    where.append(_level(usd_rate, "USD policy rate, the funding leg"))
    where.append(Symbol("s_j", "s_j",
                        "+1 or -1, whichever orients the quoted pair as long the "
                        "foreign currency against the dollar", "constant"))

    return Formula(
        plain, latex, where,
        [f"Carry is approximated from policy-rate differentials ({legs} against USD) "
         f"because the warehouse holds no forward points. Covered interest parity says "
         f"the forward discount equals the rate differential, so this stands in for "
         f"it.",
         "The differential is lagged one day: the carry earned over day t is fixed by "
         "the rates set at t-1.",
         "Each leg is held long when it yields more than the dollar and short when it "
         "yields less, and the basket is the equally weighted average of the legs.",
         "APPROXIMATION. Three non-USD legs is narrower and noisier than a real G10 "
         "carry basket; treat its loading with corresponding scepticism."],
    )


_RENDERERS = {
    "single": _f_single,
    "spread": _f_spread,
    "basket": _f_basket,
    "curve": _f_curve,
    "synthetic_bond": _f_synthetic_bond,
    "spread_level": _f_spread_level,
    "level_transform": _f_level_transform,
    "variance_premium": _f_variance_premium,
    "realized_vol_change": _f_realized_vol_change,
    "tsmom": _f_tsmom,
    "xs_momentum": _f_xs_momentum,
    "fx_carry": _f_fx_carry,
}


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def construction(method: str, inputs: dict[str, Any] | None) -> Formula:
    """The formula the builder evaluates for one factor, over named instruments."""
    inputs = dict(inputs or {})
    render = _RENDERERS.get(method)
    if render is None:
        raise ValueError(f"no formula renderer for construction method {method!r}")

    f = render(inputs)
    target = inputs.get("scale_to_vol")
    if not target:
        return f

    plain, latex, sym, step = _rescale_note(float(target))
    return Formula(f"{f.plain}\n{plain}", f"{f.latex}\n\n{latex}",
                   list(f.where) + [sym], list(f.steps) + [step])


def orthogonalisation(targets: list[str] | None, mode: str = "rolling") -> Formula:
    """The exact residualisation that turns the raw factor into the model's factor.

    Transcribed from core/orthogonalize.py::residualize_rolling as build_factors
    calls it. Two details are load-bearing and easy to miss when reading the code:

      * the regressors are the *already orthogonalised* series of the targets, not
        their raw versions, because build_factors stores each residual back into
        `panels.built`. The hierarchy therefore compounds down the levels rather
        than each level starting again from raw.
      * the fitting window ends at tau-1. Day t is never in the regression that
        residualises it.
    """
    targets = list(targets or [])
    if not targets:
        return Formula(
            "f~_t = f_t",
            r"\tilde{f}_t = f_t",
            [],
            ["This factor sits at the top of its block hierarchy and is residualised "
             "against nothing. Its orthogonalised series is identical to its raw "
             "series, value for value, so the two panels estimate the same thing "
             "here."],
        )

    names = ", ".join(targets)
    terms_plain = " - ".join(f"b{j}_tau * f~_t^{g}" for j, g in enumerate(targets, 1))
    terms_latex = " - ".join(
        rf"\hat{{b}}_{{{j},\tau}}\,\tilde{{f}}^{{\text{{{_tex(g)}}}}}_t"
        for j, g in enumerate(targets, 1))

    # Broken across lines rather than written as one expression: the panel that
    # renders this is 620px wide, and an argmin with its limits inline overflows it
    # into a horizontal scrollbar, which is a poor way to read the definition that
    # separates the two panels.
    if mode == "rolling":
        fit_plain = (f"where a_tau, b_tau minimise\n"
                     f"    sum over s = tau-{ORTH_WINDOW} .. tau-1 of\n"
                     f"        ( f_s - a - sum_j b_j * f~_s^gj )^2")
        fit_latex = (r"(\hat{a}_\tau, \hat{b}_\tau) = "
                     r"\arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} "
                     r"\Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2")
        window_step = (f"Coefficients come from ordinary least squares on the trailing "
                       f"{ORTH_WINDOW} observations ({ORTH_WINDOW // 252} years), "
                       f"refitted every {ORTH_REFIT_EVERY} observations and applied "
                       f"forward until the next refit.")
    elif mode == "expanding":
        fit_plain = ("where a_tau, b_tau minimise\n"
                     "    sum over every s < tau of\n"
                     "        ( f_s - a - sum_j b_j * f~_s^gj )^2")
        fit_latex = (r"(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} "
                     r"\sum_{s < \tau} \Bigl(f_s - a - \sum_j b_j\,"
                     r"\tilde{f}^{g_j}_s\Bigr)^2")
        window_step = ("Coefficients use all history to date. Causal, but an expanding "
                       "window never forgets, so a loading that was high in one crisis "
                       "stays high for a decade afterwards.")
    else:
        fit_plain = ("where a, b minimise\n"
                     "    sum over the whole sample of\n"
                     "        ( f_s - a - sum_j b_j * f~_s^gj )^2")
        fit_latex = (r"(\hat{a}, \hat{b}) = \arg\min_{a,b} \sum_{s} "
                     r"\Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2")
        window_step = ("Coefficients are fitted once on the whole sample. Exactly "
                       "orthogonal, but every historical factor value then contains "
                       "information from its own future.")

    plain = f"f~_t = f_t - a_tau - {terms_plain}\n{fit_plain}"
    latex = (rf"\tilde{{f}}_t = f_t - \hat{{a}}_\tau - {terms_latex}"
             "\n\n" + fit_latex)

    where = [
        Symbol("f_t", "f_t", "the raw factor, exactly as the formula above builds it",
               "constant"),
        Symbol("f~_t", r"\tilde{f}_t",
               "the orthogonalised factor, which is what the regression, the "
               "covariance matrix and the risk forecast consume", "constant"),
    ] + [
        Symbol(f"f~_t^{g}", rf"\tilde{{f}}^{{\text{{{_tex(g)}}}}}_t",
               f"the *orthogonalised* series of {g}, not its raw version",
               "factor", g)
        for g in targets
    ] + [
        Symbol("tau", r"\tau",
               f"the most recent refit at or before t. Refits begin once "
               f"{ORTH_MIN_OBS} complete observations exist and then happen every "
               f"{ORTH_REFIT_EVERY}", "constant"),
    ]

    steps = [
        f"Residualise against {names}, in that order of the block hierarchy.",
        window_step,
        "The fitting window ends at tau-1, so day t is never part of the regression "
        "that residualises it. That is the whole reason for the rolling refit: a "
        "full-sample residual would put future information into historical factor "
        "values, and this project's headline output is a predicted-versus-realised "
        "risk comparison, which such a leak would flatter exactly where the model is "
        "being judged.",
        "The regressors are the targets' own orthogonalised series, so the hierarchy "
        "compounds: by the time a level-3 factor is residualised, the level-0 factors "
        "it sees have already had everything above them removed.",
        f"Before the first refit, with fewer than {ORTH_MIN_OBS} usable observations "
        f"or a missing regressor on the day, the orthogonalised value is NULL rather "
        f"than a silent fallback to the raw value. A factor that quietly stops being "
        f"orthogonal on some dates is worse than one with a documented gap.",
    ]
    return Formula(plain, latex, where, steps)


# Helper functions each builder leans on, worth showing beside it because the
# arithmetic that matters is often in them rather than in the builder itself: a
# curve factor's five readable lines call out to duration, convexity and a
# synthetic par bond, and reading only the caller tells you nothing.
_BUILDER_HELPERS: dict[str, list[tuple[str, str]]] = {
    "curve": [("backend.pipeline.build_factors", "_synthetic_bond_returns"),
              ("backend.core.transforms", "yield_change_to_return"),
              ("backend.core.transforms", "par_bond_modified_duration")],
    "synthetic_bond": [("backend.core.transforms", "yield_change_to_return"),
                       ("backend.core.transforms", "par_bond_modified_duration")],
    "level_transform": [("backend.core.transforms", "sparse_release_change"),
                        ("backend.core.transforms", "apply_transform")],
    "spread_level": [("backend.core.transforms", "diff_standardized")],
    "realized_vol_change": [("backend.core.transforms", "diff_standardized")],
    "single": [("backend.core.transforms", "to_excess_return")],
}


def _source_of(module_name: str, func_name: str) -> dict | None:
    """One function's source, with the file and line it starts at.

    Read with `inspect.getsource` rather than stored as a string: a snippet that
    is copied cannot be wrong at the moment it is written and cannot be right for
    long. This one is the running code by construction.
    """
    import importlib
    import inspect

    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name)
        code = inspect.getsource(fn)
        _, line = inspect.getsourcelines(fn)
    except Exception:
        return None
    return {
        "name": func_name,
        "module": module_name,
        "path": module_name.replace(".", "/") + ".py",
        "line": line,
        "code": code.rstrip(),
    }


def source_code(method: str, inputs: dict[str, Any] | None = None) -> dict:
    """The Python that builds a factor of this method, as it actually runs.

    The formula says what the arithmetic is; this says where it lives, which is the
    question anyone who doubts a number asks next. `call` shows the builder invoked
    with this factor's own inputs, so the snippet is not generic.
    """
    import json as _json

    entry = "_variance_premium" if method == "variance_premium" else None
    if entry is None:
        entry = {
            "single": "_single", "spread": "_spread", "basket": "_basket",
            "curve": "_curve", "synthetic_bond": "_synthetic_bond",
            "spread_level": "_spread_level", "level_transform": "_level_transform",
            "realized_vol_change": "_realized_vol_change", "tsmom": "_tsmom",
            "xs_momentum": "_xs_momentum", "fx_carry": "_fx_carry",
        }.get(method)
    if entry is None:
        return {"method": method, "builder": None, "helpers": [], "call": ""}

    builder = _source_of("backend.pipeline.build_factors", entry)
    helpers = [h for h in (_source_of(m, f)
                          for m, f in _BUILDER_HELPERS.get(method, []))
               if h is not None]

    rendered = _json.dumps(inputs or {}, indent=4, sort_keys=True)
    call = (f"# backend/pipeline/build_factors.py, in build_one()\n"
            f"inputs = {rendered}\n"
            f"series = {entry}(panels, inputs)")
    if (inputs or {}).get("scale_to_vol"):
        call += (f"\nseries = _rescale(series, "
                 f"{float(inputs['scale_to_vol']):g})   # to a return scale")

    return {"method": method, "builder": builder, "helpers": helpers, "call": call}


def for_factor(spec: dict, mode: str = "rolling") -> dict:
    """Both formulas for one entry of the factor registry.

    Accepts either a `factor_defs` entry or a `ref_factor` row whose `construction`
    JSON has been parsed, so the API and the docs can share this without either one
    reshaping its data first.
    """
    construction_spec = spec.get("construction") or spec
    method = construction_spec.get("method") or spec.get("method")
    inputs = construction_spec.get("inputs") or spec.get("inputs") or {}
    targets = (spec.get("orth")
               if spec.get("orth") is not None
               else spec.get("orthogonalize_against")) or []

    return {
        "method": method,
        "construction": construction(method, inputs).as_dict(),
        "orthogonalisation": orthogonalisation(list(targets), mode).as_dict(),
        "orthogonalised_against": list(targets),
        "preamble": [s.__dict__ for s in PREAMBLE],
    }


def all_factors(mode: str = "rolling") -> dict[str, dict]:
    """Every factor in the registry, rendered. Used by the docs and the corpus."""
    return {f["id"]: for_factor(f, mode) for f in factor_defs.FACTORS}


def main() -> int:
    """Print the whole appendix. `python -m backend.pipeline.formula`."""
    for f in factor_defs.FACTORS:
        rendered = for_factor(f)
        print(f"\n{'=' * 78}\n{f['id']}  --  {f['name']}  [{f['block']}]\n{'=' * 78}")
        print("\nConstruction\n")
        print("    " + rendered["construction"]["plain"].replace("\n", "\n    "))
        print("\nOrthogonalisation\n")
        print("    " + rendered["orthogonalisation"]["plain"].replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
