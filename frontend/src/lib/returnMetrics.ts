/**
 * What every number on the Return Forecasts page means and how it is computed.
 *
 * Same shape and rule as metrics.ts: one definition per quantity. Transcribed
 * from backend/core/returns.py, backend/pipeline/return_forecast.py and
 * backend/pipeline/return_track_record.py.
 */

import { Metric } from "@/lib/metrics";

export const RETURN_METRICS: Record<string, Metric> = {
  total: {
    title: "Ten-year expected return",
    what: "The compound annual return the class is expected to earn over the next ten "
        + "years, in dollars, as the sum of its building blocks.",
    how: "Income + net buybacks + real growth + inflation + valuation change + other "
       + "(roll-down, hedge carry, default losses). Exact in logs, and good to the second "
       + "order in simple returns, which is the convention of every published "
       + "building-block model.",
    formula: String.raw`E[r] \approx \underbrace{D/P}_{\text{income}} + \underbrace{b}_{\text{buybacks}} + \underbrace{g}_{\text{real growth}} + \underbrace{\pi}_{\text{inflation}} + \underbrace{\tfrac{\min(H,T)}{T}\,\tfrac{\ln(\text{norm}) - \ln(\text{CAPE})}{H}}_{\text{valuation}}`,
    read: "A central value, not a promise: the band beside it is the honest summary. The "
        + "valuation block is where nearly all the uncertainty sits, which is why the "
        + "breakdown matters more than the sum.",
    ref: "Brightman (2012); J.P. Morgan LTCMA methodology; Invesco CMA (2024).",
  },
  income: {
    title: "Income",
    what: "Equities and REITs: the fund's distribution yield. Bonds: the average of the "
        + "yield today and the yield assumed at the end of the horizon.",
    how: "Distribution yield = log(adj. close ratio) − log(close ratio) over 252 days, "
       + "the median of the last three years so one special distribution cannot set it. "
       + "Bond yields are read off the Treasury, euro-area AAA and JGB curves at the "
       + "fund's maturity.",
    read: "The one block that is observable today. For bonds it dominates, which is why "
        + "bond forecasts have a far better record than equity forecasts.",
  },
  real_growth: {
    title: "Real growth per share",
    what: "How fast real earnings per share have grown, measured on Shiller's record "
        + "from 1871.",
    how: "Ten-year average real earnings at both ends of the record, log difference over "
       + "the years between their midpoints. Its standard error comes from the "
       + "non-overlapping decade growth rates.",
    read: "Per-share growth, not GDP growth: dilution by new issues is already in it. The "
        + "same rate is used for every equity region; REITs, which pay out most of their "
        + "earnings, get none.",
  },
  inflation: {
    title: "Expected inflation",
    what: "The 10-year breakeven inflation rate (FRED T10YIE).",
    how: "The forecast is in dollars, so a foreign market's local inflation and its "
       + "currency's drift net out to US inflation under relative purchasing-power parity.",
    read: "PPP holds over decades only roughly; the currency is the largest unmodelled "
        + "source of error for the foreign classes.",
  },
  valuation: {
    title: "Valuation change",
    what: "The annual return from the valuation ratio drifting toward its own history.",
    how: "US: Shiller CAPE toward its geometric mean since 1881. Europe and Japan: "
       + "Barclays' country CAPE toward its median since 1982. EM and REITs: the "
       + "dividend yield toward its mean since the fund started. The gap closes linearly "
       + "in logs over the reversion period (15 years), so two thirds of it falls inside "
       + "a ten-year horizon.",
    read: "Which history defines 'normal' is a judgement with large leverage: the US norm "
        + "is 16 since 1881 and 27 since 1990. The band draws among the samples. A full "
        + "reversion to a historical mean also assumes rates revert to the level that "
        + "shaped that mean.",
  },
  other: {
    title: "Other",
    what: "Bonds: roll-down less default losses, plus the hedge carry for international "
        + "bonds. Equities: net buybacks.",
    how: "Roll-down = duration × (y(M) − y(M−1)) on today's curve. Default losses are "
       + "assumptions: 0.10% a year on the IG share of AGG, 1.0% on EM sovereigns.",
    read: "Small next to the yield, and the part that depends most on assumptions.",
  },
  arithmetic: {
    title: "Arithmetic vs geometric",
    what: "The expected one-year return, as mean-variance optimisation needs it, beside "
        + "the compound return an investor's wealth grows at.",
    how: "(1 + g)·exp(σ²/2) − 1, about g + σ²/2, with σ from the strategic covariance.",
    formula: String.raw`\mu_{\text{arith}} = (1+g)\,e^{\sigma^2/2} - 1 \approx g + \tfrac{\sigma^2}{2}`,
    read: "The gap is volatility drag: two assets with the same arithmetic mean end with "
        + "different wealth, and the more volatile ends with less (Markowitz on Kelly).",
  },
  band: {
    title: "Ten-year band",
    what: "The 5th to 95th percentile of the simulated compound annual return over ten "
        + "years.",
    how: "20,000 paths. Each draws the block assumptions - the valuation norm's sample, "
       + "a reversion period of 10 to 20 years, growth within its standard error, where "
       + "yields end - and then ten years of correlated return noise from the strategic "
       + "covariance.",
    read: "Wide by design. The median can differ from the central value because the band "
        + "draws the valuation norm from several samples while the central value uses "
        + "the first.",
  },
  vol: {
    title: "Strategic volatility",
    what: "Annualised volatility from the funds' own monthly returns since 2000 - "
        + "crises included.",
    how: "Equal-weighted pairwise sample of monthly log total returns, × 12, repaired to "
       + "positive semi-definite. The quarterly figure is the check that monthly "
       + "sampling is not the story; the factor-model figure is the risk model's daily "
       + "covariance on the same funds.",
    read: "Deliberately not the risk page's fast covariance: that answers 'how risky "
        + "tomorrow', this answers 'how do these move together over a decade'. The daily "
        + "factor-model figure runs higher because daily crisis noise partly reverses "
        + "within the month.",
  },
  history: {
    title: "Rear-view mirror",
    what: "The fund's own compound annual total return over its whole history, and over "
        + "the last ten years.",
    how: "exp(sum of daily log returns / years) − 1.",
    read: "What a forecast by extrapolation says. On Shiller's record the last decade's "
        + "return correlates −0.37 with the next decade's: the rear-view mirror is not "
        + "just noisy, it points the wrong way.",
  },
  implied: {
    title: "Implied (reverse-optimised) returns",
    what: "The excess returns under which a portfolio would be the mean-variance optimum.",
    how: "μ = λΣw. λ is set so that the market portfolio's implied excess return equals "
       + "its building-block excess return; the same λ serves every implied vector, so "
       + "they are comparable.",
    formula: String.raw`\mu_{\text{implied}} = \lambda\,\Sigma\,w, \qquad \lambda = \frac{w_m^{\prime}\mu_{\text{BB}}}{w_m^{\prime}\Sigma\,w_m}`,
    read: "Not a forecast but a place to shrink toward when the forecasts are doubted. "
        + "The minimum-variance vector is the same number for every asset it holds - "
        + "unconstrained, Σw_mv is proportional to a vector of ones - and higher for the "
        + "assets it leaves out.",
    ref: "Black and Litterman (1992); Kan, Wang and Zhou (2016).",
  },
  lambda: {
    title: "Risk aversion λ",
    what: "The price of variance that turns a portfolio into a return vector.",
    how: "w_m′μ_BB / w_m′Σw_m for the market portfolio w_m.",
    read: "Low when the building blocks expect little from the market: the implied "
        + "vectors then expect little too. If the market's building-block excess return "
        + "is not positive, λ falls back to 2.5.",
  },
  gap: {
    title: "Gap to the frontier",
    what: "How far a portfolio's expected excess return falls below the frontier of a "
        + "given vector at the same risk.",
    how: "Frontier return at the portfolio's volatility minus the portfolio's expected "
       + "excess return, both under that vector.",
    read: "Zero for the vector's own optimum; for every other vector it is the price of "
        + "having been wrong. Read across a row, what holding the portfolio costs under "
        + "each view; down a column, which portfolios a view forgives (Broadie 1993).",
  },
  oosR2: {
    title: "Out-of-sample R²",
    what: "How much of the forecast error of the historical mean the forecast removes.",
    how: "1 − Σ(realised − forecast)² / Σ(realised − expanding mean)², with every "
       + "forecast built from data known at its date.",
    formula: String.raw`R^2_{\text{OOS}} = 1 - \frac{\sum_t (r_t - \hat r_t)^2}{\sum_t (r_t - \bar r_t)^2}`,
    read: "Positive: the forecast beat the historical mean. Overlapping ten-year windows "
        + "share most of their outcome, so a long monthly sample holds about a dozen "
        + "independent decades; read the non-overlapping row for the honest sample size.",
    ref: "Campbell and Thompson (2008); Goyal and Welch (2008).",
  },
  dmT: {
    title: "Diebold-Mariano t",
    what: "Whether the forecast's squared errors are smaller than the historical mean's.",
    how: "Mean loss differential over its Bartlett long-run standard error, 119 lags for "
       + "overlapping monthly ten-year windows.",
    read: "Negative favours the forecast. Below about −2 is significant; on ten-year "
        + "horizons it rarely is, which is the point Böni et al. make about a decade "
        + "being too short to prove an advantage.",
  },
};
