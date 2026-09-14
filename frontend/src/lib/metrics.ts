/**
 * What each headline number means, and how it is computed.
 *
 * A row of nine figures across the top of a page is only useful to someone who
 * already knows what all nine are. "Max VIF 88" says nothing on its own: not what a
 * variance inflation factor is, not that it comes from regressing one factor on the
 * other thirty-nine, not that the risk pipeline gates on it at 100, and not that on
 * the raw panel it is a property of the factor set rather than a fault.
 *
 * So each card carries its own definition, opened on hover. Three fields, because
 * they answer three different questions and conflating them produces the kind of
 * tooltip that restates the label:
 *
 *   what   the quantity, in one sentence
 *   how    the arithmetic, close enough to reproduce it
 *   read   what a given value tells you, including where it is misleading
 *
 * Definitions live here rather than beside each page, for the same reason
 * `verdict.ts` owns which outcome is good: the condition number appears on the
 * Loadings Lab and on Covariance & PCA, and two pages describing it differently
 * would mean one of them is wrong. Where a figure has a documented caveat -- a
 * Mincer-Zarnowitz slope below 1 is partly mechanical, a high VIF is structural on
 * the raw panel -- `read` carries it, because that is the half a reader is most
 * likely to get wrong.
 */

export interface Metric {
  /** Full name, when the card label has to be short. */
  title?: string;
  what: string;
  how: string;
  read: string;
}

export const METRICS: Record<string, Metric> = {
  // --- the rolling regression -------------------------------------------
  windows: {
    title: "Estimation windows",
    what: "How many separate regressions this spec produced for the security.",
    how: "One regression per roll-forward step: the window slides by the step size "
       + "and refits, from the first date with enough history to the last.",
    read: "More windows means a longer history to judge stability over, not a "
        + "better fit. A 252-day window rolled monthly over twenty years gives "
        + "about 240.",
  },
  adjR2: {
    title: "Adjusted R²",
    what: "The share of the security's return variance the factors explain in the "
        + "latest window, after penalising the model for the number of regressors.",
    how: "1 − (1 − R²)·(n − 1)/(n − k − 1), with n observations and k factors. The "
       + "penalty matters here: forty factors on 252 days would raise plain R² "
       + "whether or not they carry information.",
    read: "For a single stock, 0.2–0.5 is normal — most of a stock's variance is "
        + "idiosyncratic by construction. A figure near 1 on a single name usually "
        + "means the factor set contains something close to the security itself.",
  },
  alpha: {
    title: "Alpha",
    what: "The intercept of the regression, annualised: the average return left "
        + "unexplained by the factors.",
    how: "The fitted constant term, multiplied by 252. Its t-statistic divides it "
       + "by its Newey-West standard error.",
    read: "Read the t-statistic, not the percentage. Alpha on a single name over "
        + "one window is dominated by noise, and |t| below about 2 means the data "
        + "cannot distinguish it from zero.",
  },
  residualVol: {
    title: "Residual volatility",
    what: "The annualised volatility of what the factors do not explain — the "
        + "security's idiosyncratic risk.",
    how: "Standard deviation of the regression residuals × √252. The risk model "
       + "shrinks this toward the cross-sectional median before using it, and "
       + "floors it at 2%, so a single quiet window cannot produce a falsely "
       + "precise idiosyncratic risk.",
    read: "This enters the risk forecast directly: predicted variance is β′Σβ plus "
        + "this squared. For a single stock it is usually the larger of the two.",
  },
  observations: {
    title: "Observations",
    what: "Daily returns in the latest estimation window.",
    how: "Days on which the security and every factor in the set all have a value. "
       + "Days with any missing regressor are dropped rather than filled.",
    read: "Below the window length means gaps in the panel. Well below the number "
        + "of factors means the design is under-determined and the window will be "
        + "excluded from risk scoring.",
  },
  durbinWatson: {
    title: "Durbin-Watson",
    what: "Whether the regression residuals are serially correlated.",
    how: "Σ(eₜ − eₜ₋₁)² / Σeₜ², over the residuals of the latest window. Two means "
       + "no first-order autocorrelation, zero means perfect positive.",
    read: "Away from 2 says the residual still carries structure the factors "
        + "missed — often stale pricing, or a missing factor. It does not "
        + "invalidate the betas, but it does mean the plain standard errors would "
        + "be too small, which is why these are Newey-West.",
  },
  conditionNumber: {
    title: "Condition number",
    what: "How close the design matrix is to singular — how much the factors "
        + "overlap with each other in this window.",
    how: "Ratio of the largest to the smallest singular value of the standardised "
       + "regressor matrix.",
    read: "The invariant that governs how far β′Σβ can blow up, so it means the "
        + "same thing on either factor panel. Above 200 the betas explode in "
        + "offsetting pairs and the window is excluded from risk scoring — the "
        + "failure that once produced a 360% predicted volatility for AAPL.",
  },
  maxVif: {
    title: "Maximum VIF",
    what: "The worst variance inflation factor in the design: how much one "
        + "factor's coefficient variance is inflated by its overlap with the rest.",
    how: "For each factor, regress it on all the others and take 1/(1 − R²). This "
       + "is the largest of those. The condition number says the design as a whole "
       + "is ill-conditioned; VIF says which factor is responsible.",
    read: "Above 100 is anomalous on the orthogonalised panel and the risk "
        + "pipeline gates on it there. On the raw panel it is the definition of "
        + "the factor set — eq_us regressed on the other thirty-nine has an R² "
        + "near 0.9998, because eq_global is one of them — so it is recorded and "
        + "not gated.",
  },
  fPValue: {
    title: "F p-value",
    what: "Whether the factors jointly explain anything at all.",
    how: "An F-test of the null that every slope coefficient is zero, against the "
       + "alternative that at least one is not.",
    read: "Below 0.05 means the factor set as a whole is doing work. It is a weak "
        + "test with forty regressors and says nothing about any individual beta.",
  },

  // --- the risk backtest -------------------------------------------------
  forecastsScored: {
    title: "Forecasts scored",
    what: "How many ex-ante forecasts were compared against what actually "
        + "happened.",
    how: "One per estimation window that had a usable covariance and a realised "
       + "outcome. Windows excluded by the conditioning gate are stored and "
       + "flagged, not scored.",
    read: "A backtest on a handful of forecasts says very little. The excluded "
        + "count beside it is worth reading: a spec losing most of its windows is "
        + "not really estimable.",
  },
  biasRatio: {
    title: "Bias ratio",
    what: "Whether the forecast gets the level of risk right on average.",
    how: "Mean of realised volatility divided by predicted volatility, across "
       + "every scored forecast.",
    read: "One is right. Above one means the model under-forecasts risk, which is "
        + "the dangerous direction. It says nothing about timing: a forecast that "
        + "is right on average and wrong in every individual period scores 1.0 "
        + "here and fails Mincer-Zarnowitz.",
  },
  biasStatistic: {
    title: "Bias statistic",
    what: "Whether the forecast gets the level right period by period, not just on "
        + "average.",
    how: "Standard deviation of the standardised returns rₜ / σ̂ₜ₋₁. If the "
       + "forecast were exactly right, those would be unit-variance.",
    read: "One is right. This is the stricter of the two level checks, because "
        + "over- and under-forecasting in different periods cannot cancel out the "
        + "way they can in the bias ratio.",
  },
  mzSlope: {
    title: "Mincer-Zarnowitz slope",
    what: "Whether the forecast tracks the variation in realised risk, not just "
        + "its level.",
    how: "Slope b from regressing realised variance on predicted variance: "
       + "σ²_realised = a + b·σ²_predicted. Unbiased means a = 0 and b = 1.",
    read: "A slope below 1 is partly mechanical whenever volatility moves faster "
        + "than the horizon: realised volatility over a forward window blends "
        + "regimes the forecast refers to singly. On simulated data with a perfect "
        + "forecast the slope comes out at 0.50 with 21-day regimes and 1.02 with "
        + "504-day ones — so a low slope is evidence, not proof the model "
        + "over-reacts.",
  },
  mzJoint: {
    title: "Mincer-Zarnowitz joint test",
    what: "Whether the intercept and slope together are consistent with an "
        + "unbiased forecast.",
    how: "An F-test of a = 0 and b = 1 jointly, on the same regression.",
    read: "Above 0.05 means unbiasedness cannot be rejected. Below it, read the "
        + "slope and the bias ratio to see which half is failing.",
  },
  mzR2: {
    title: "Mincer-Zarnowitz R²",
    what: "How much of the variation in realised risk the forecast explains — its "
        + "timing ability, as opposed to its level.",
    how: "R² of the same regression of realised on predicted variance.",
    read: "Low figures are normal: realised volatility over a short forward window "
        + "is itself a noisy estimate, which caps how much of it anything can "
        + "explain. Near zero means the forecast is delivering a level and no "
        + "timing.",
  },
  var95: {
    title: "VaR 95% coverage",
    what: "How often the daily loss exceeded the 95% value-at-risk the model "
        + "predicted, against how often it should have.",
    how: "Count of days where the return fell below −1.645·σ̂, against 5% of the "
       + "days scored. Kupiec tests whether that count is consistent with 5%.",
    read: "Too many breaches means the model under-states risk; too few means it "
        + "is too wide and would over-reserve capital. Both fail Kupiec, and the "
        + "count alone cannot see whether the breaches arrived together.",
  },
  var99: {
    title: "VaR 99% coverage",
    what: "The same count at the 99% level, where the tail assumption bites.",
    how: "Count of days below −2.326·σ̂, against 1% of the days scored.",
    read: "This is where a normal assumption fails on daily returns: fat tails "
        + "produce more 99% breaches than a Gaussian VaR expects, even when the "
        + "95% level is well calibrated.",
  },
  clustering: {
    title: "Christoffersen independence test",
    what: "Whether the VaR breaches arrived independently or in bursts.",
    how: "A likelihood-ratio test of whether a breach today is independent of a "
       + "breach yesterday, on the 95% exceedance sequence.",
    read: "The more dangerous failure, and one the breach count cannot see: a "
        + "model can have exactly the right number of breaches and take all of "
        + "them in one week. Applying a 21-day-horizon forecast to daily returns "
        + "clusters them by construction, so a rejection here is expected and is "
        + "not evidence about the factor panel.",
  },

  // --- the covariance matrix ---------------------------------------------
  shrinkage: {
    title: "Shrinkage intensity",
    what: "How far the covariance estimate was pulled from the sample matrix "
        + "toward a structured target.",
    how: "The Ledoit-Wolf optimal intensity δ, chosen to minimise expected squared "
       + "error against a constant-correlation target: Σ̂ = δ·target + (1−δ)·sample.",
    read: "Zero is the raw sample matrix, one is the target. High values mean the "
        + "sample is short relative to the number of factors and the sample "
        + "covariance on its own would be badly conditioned.",
  },
  /**
   * Not the same quantity as `conditionNumber`, despite the shared name. That one
   * measures a security's regression design and is gated at 200; this one measures
   * the factor covariance matrix, where four figures is ordinary for 39 factors and
   * nothing is excluded on account of it. Sharing an entry would have put "excluded
   * from risk scoring" under a number that is not.
   */
  covConditionNumber: {
    title: "Covariance condition number",
    what: "How close the estimated factor covariance matrix is to singular.",
    how: "Ratio of its largest eigenvalue to its smallest, after shrinkage and any "
       + "PSD repair.",
    read: "Four figures is ordinary for 39 correlated factors and is not a fault — "
        + "this is the covariance matrix, not a regression design, and nothing is "
        + "excluded on account of it. What it does say is how much an optimiser "
        + "inverting this matrix would amplify estimation error in the smallest "
        + "direction, which is the argument for the shrinkage beside it. The raw "
        + "panel runs near 9,000 against about 1,000 orthogonalised.",
  },
  psd: {
    title: "Positive semi-definite",
    what: "Whether every portfolio this matrix prices has non-negative variance.",
    how: "True when the smallest eigenvalue is at or above zero. When it is not, "
       + "the matrix is repaired by clipping negative eigenvalues to zero and "
       + "rescaling so the diagonal variances are preserved.",
    read: "A sample covariance on fewer observations than factors is singular, and "
        + "numerical error can push a shrunk one slightly indefinite. Repaired is "
        + "fine and is recorded; unrepaired and indefinite would let an optimiser "
        + "build a portfolio out of the negative eigenvalue and claim risk-free "
        + "return.",
  },
  minEigenvalue: {
    title: "Smallest eigenvalue",
    what: "Whether the covariance matrix is positive semi-definite — whether every "
        + "portfolio it prices has non-negative variance.",
    how: "The smallest eigenvalue of the estimated matrix, before any repair.",
    read: "Negative means the matrix is indefinite, and an optimiser handed it "
        + "will happily build a portfolio out of the negative eigenvalue and claim "
        + "risk-free return. Repaired by clipping eigenvalues at zero and "
        + "rescaling, which is recorded when it happens.",
  },
  pc1Share: {
    title: "PC1 variance share",
    what: "How much of the factor panel's common variation lies in a single "
        + "direction.",
    how: "The largest eigenvalue of the correlation matrix divided by the sum of "
       + "all of them. Correlation rather than covariance, so a high-volatility "
       + "factor cannot dominate through scale alone.",
    read: "A high share means the factor set is less independent than its count "
        + "suggests. On the raw panel this runs near 50% and on the orthogonalised "
        + "panel near 35% — the difference is what the block hierarchy removes.",
  },
};
