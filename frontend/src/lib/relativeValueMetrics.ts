/**
 * What every number on the Relative Value & Timing page means and how it is
 * computed.
 *
 * Same shape and rule as metrics.ts and returnMetrics.ts: one definition per
 * quantity. Transcribed from backend/core/relative_value.py and
 * backend/pipeline/relative_value.py.
 */

import { Metric } from "@/lib/metrics";

export const RV_METRICS: Record<string, Metric> = {
  value: {
    title: "Value signal",
    what: "How cheap the long leg is against the short leg, compared with the pair's own "
        + "history, as a z-score.",
    how: "Each view's valuation spread, oriented so that higher means the long leg is cheaper: "
       + "US equities vs TIPS use Shiller's excess CAPE yield; the regional pairs the gap "
       + "between CAPE earnings yields (Barclays) or distribution yields (EM); high yield vs "
       + "investment grade the HYG-LQD distribution-yield gap; TIPS the breakeven, inverted; "
       + "duration the ACM term premium. z against every month up to the current one, "
       + "clipped at ±3; at least 60 months of history.",
    formula: String.raw`z_t = \operatorname{clip}\!\left(\frac{x_t - \bar x_{\le t}}{s_{\le t}},\ \pm 3\right)`,
    read: "Value sets the level of a long-run return, not its date: over one year it explains "
        + "next to nothing. A large z is a reason to look, and a weak reason to act now. The "
        + "percentile beside it says how unusual the level is without assuming normality.",
  },
  carry: {
    title: "Carry signal",
    what: "What the pair earns if prices do not move, against its own history.",
    how: "Each leg's carry over cash: equities, REITs and credit funds their trailing "
       + "distribution yield; Treasury funds the curve yield at their maturity plus roll-down; "
       + "TIPS the real yield plus three-month CPI inflation; gold minus the cash rate. A "
       + "hedged leg is measured over its own currency's short rate. Pair carry = long − w × "
       + "short at the volatility-neutral weight w, then z-scored like value.",
    formula: String.raw`c_{\text{pair}} = c_{\text{long}} - w\,c_{\text{short}}, \quad w = \sigma_{\text{long}}/\sigma_{\text{short}}`,
    read: "Carry is earned only while prices hold still. A high-carry pair with a falling "
        + "trend is the classic setup for a carry crash. No carry is shown for commodities "
        + "(no futures curve) or for EM vs US (the yield gap is already the value signal).",
  },
  trend: {
    title: "Trend signal",
    what: "The pair's return from twelve months ago to one month ago, over its volatility.",
    how: "Daily log excess returns of a volatility-neutral pair (the short leg scaled to the "
       + "long leg's trailing one-year volatility, the weight lagged a day). With no drift the "
       + "ratio is standard normal, so it needs no history. Clipped at ±3.",
    formula: String.raw`z_t = \frac{\ln P_{t-1} - \ln P_{t-12}}{\sigma_t\sqrt{11/12}}`,
    read: "Trend is here to keep a value view out of a value trap, not to forecast. The tick "
        + "beside it says whether the pair's total-return level is above its ten-month "
        + "average (Faber 2007).",
    ref: "Moskowitz, Ooi and Pedersen (2012); Faber (2007).",
  },
  macro: {
    title: "Macro signal",
    what: "Growth and inflation momentum, signed by which way the pair gains in each.",
    how: "Growth: minus the three-month change in the Chicago Fed NFCI (first releases on "
       + "their publication dates). Inflation: the three-month change in the 10-year "
       + "breakeven. Each z-scored against its history, then (s_g z_g + s_π z_π) / (|s_g| + |s_π|) "
       + "with the pair's signs from the growth × inflation quadrants.",
    read: "Both inputs are market proxies; no activity or CPI release is in the database. The "
        + "signs are set by hand, not estimated. The regional equity pairs have none, so they "
        + "carry no macro signal rather than a neutral one.",
  },
  score: {
    title: "Composite score",
    what: "The weighted average of the signals a view has.",
    how: "S = Σ w_k z_k / Σ w_k over the available signals, equal weights by default. A "
       + "missing signal is left out rather than counted as zero, which would read as "
       + "'fairly valued'.",
    formula: String.raw`S = \frac{\sum_k w_k z_k}{\sum_{k\,\text{available}} w_k}`,
    read: "Stance: |S| < 0.25 neutral, up to 0.75 a lean, beyond it a full over- or "
        + "underweight (editable). The sparkline is the score month by month over three years, "
        + "each month using only what was known then.",
  },
  change: {
    title: "One-month change and the no-trade band",
    what: "How far the score moved since the previous month-end, and whether that is enough "
        + "to act.",
    how: "Act when |S_t − S_{t−1}| exceeds the band (0.25 by default); otherwise hold.",
    read: "The band keeps a noisy score from turning the portfolio over. 'Hold' does not mean "
        + "the view is weak; it means it has not changed.",
  },
  tilt: {
    title: "Suggested tilt and its tracking error",
    what: "The active weight on the long leg the score maps to, with the short leg at the "
        + "volatility-neutral weight.",
    how: "a = clip(k S, ±a_max), k = 2 pp per unit of score and a_max = 5 pp by default. TE = "
       + "|a| × the pair's volatility. When the total, √Σ TE², exceeds the budget, every tilt "
       + "is scaled by the same factor.",
    read: "A heuristic, not an optimisation. The total treats the views as independent; three "
        + "of them are against US equities, so the true tracking error is higher.",
  },
  anchor: {
    title: "Ten-year anchor",
    what: "The building blocks' expected excess return of the pair over the next decade, and "
        + "per unit of the pair's volatility.",
    how: "From the latest Ten-Year Forecasts run: long excess − w × short excess. TIPS, high "
       + "yield and investment grade are not classes there; they get the bond building block "
       + "with the yield unchanged - real yield + the forecast's inflation, Treasury yield + "
       + "OAS − PD × LGD.",
    read: "The anchor is the slow view the signals sit beside. Where it and the score point "
        + "opposite ways, the disagreement is the information.",
  },
  per_risk: {
    title: "Excess return per unit of risk",
    what: "Ten-year expected excess return over cash divided by volatility: a Sharpe ratio of "
        + "the building blocks.",
    how: "Expected compound return − expected cash, over the strategic (monthly) volatility "
       + "of the run, or the fund's own monthly volatility for the three extra legs.",
    read: "A ranking device. The expected return under it is a ten-year central value with a "
        + "wide band (Ten-Year Forecasts).",
  },
  regime: {
    title: "Macro regime",
    what: "The growth × inflation quadrant the two momentum signals point to.",
    how: "Growth z > 0 and inflation z ≤ 0: goldilocks; both > 0: reflation; growth ≤ 0 and "
       + "inflation > 0: stagflation; both ≤ 0: recession.",
    read: "A momentum reading of market proxies, not a nowcast of GDP and CPI. Near zero on "
        + "both axes the quadrant is a coin toss.",
  },
  credit: {
    title: "Credit after expected losses",
    what: "What an OAS pays after expected defaults, the default rate that would eat it, and "
        + "the widening over a year that would.",
    how: "EL = PD × LGD; excess = s − EL; PD* = s / LGD; cushion = (s − EL) / spread duration; "
       + "cover = PD* / PD.",
    read: "Moody's speculative-grade default rate counts issuers and includes loans, so it "
        + "overstates a high-yield bond index; the second line shows the same arithmetic at a "
        + "lower default rate. The OAS on FRED starts in May 2023, too short for a norm.",
  },
};
