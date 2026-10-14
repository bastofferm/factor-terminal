/**
 * What every number the portfolio view shows means and how it is computed.
 *
 * Same shape and the same rule as metrics.ts and agentMetrics.ts: one definition
 * per quantity. Transcribed from backend/core/portfolio.py (decompose, score) and
 * backend/pipeline/portfolio_risk.py.
 */

import { AGENT_METRICS } from "@/lib/agentMetrics";
import { Metric } from "@/lib/metrics";

export const PORTFOLIO_METRICS: Record<string, Metric> = {
  predictedVol: {
    title: "Predicted volatility",
    what: "The portfolio's annualised volatility as the model forecasts it for the "
        + "next day, from the latest data.",
    how: "Each holding's betas from its latest regression, the factor covariance "
       + "through today, each holding's specific variance, combined with the "
       + "weights. The part of systematic risk that is only beta estimation noise "
       + "is removed.",
    formula: String.raw`\hat\sigma_p^2 = x^{\prime}\hat\Sigma_F\,x - \sum_i w_i^2 \operatorname{tr}\!\left(\hat\Sigma_F V_i\right) + \sum_i w_i^2 \hat\sigma^2_{\varepsilon,i}, \qquad x = B^{\prime} w

\hat\sigma_p^{\text{ann}} = \sqrt{252\, \hat\sigma_p^2}`,
    read: "Compare it with the backtest below: a forecast is only as good as its "
        + "record on the portfolio's own history.",
  },
  systematicShare: {
    title: "Systematic share",
    what: "How much of the portfolio's variance the factors account for; the rest "
        + "is the holdings' specific risk.",
    how: "Systematic variance divided by total variance on the latest day.",
    formula: String.raw`\frac{x^{\prime}\hat\Sigma_F x - \sum_i w_i^2 \operatorname{tr}(\hat\Sigma_F V_i)}{\hat\sigma_p^2}`,
    read: "Diversification drives specific risk down roughly as 1/n, so a broad "
        + "portfolio is mostly systematic. A share far below that for many "
        + "holdings means they share risk the factors miss.",
  },
  specificVol: {
    title: "Specific volatility",
    what: "The part of the portfolio's risk that comes from its holdings' "
        + "idiosyncratic moves, treated as independent across holdings.",
    how: "Square root of the weighted specific variances, annualised.",
    formula: String.raw`\hat\sigma_{\text{spec}} = \sqrt{252 \sum_i w_i^2\, \hat\sigma^2_{\varepsilon,i}}`,
    read: "Independence across holdings is an assumption. For stocks of one sector "
        + "it held only once the sector factor was added: without it their "
        + "residuals correlated 0.22 in the portfolio validation.",
  },
  var99: {
    title: "1-day VaR 99%",
    what: "The one-day loss the portfolio should exceed on one day in a hundred.",
    how: "The forecast daily volatility times the 99% quantile of a Student t whose "
       + "degrees of freedom are fitted to the portfolio's last 756 standardised "
       + "returns (normal in brackets).",
    formula: String.raw`\mathrm{VaR}^{99\%} = \frac{t^{-1}_{\hat\nu}(0.99)}{\sqrt{\hat\nu/(\hat\nu-2)}}\; \frac{\hat\sigma_p^{\text{ann}}}{\sqrt{252}}`,
    read: "A fraction of the portfolio's value. The backtest's 99% breach count says "
        + "whether it was right in the past.",
  },
  trackingError: {
    title: "Tracking error",
    what: "The forecast volatility of the portfolio's return relative to the "
        + "benchmark fund - the active risk.",
    how: "The same forecast applied to the active weights: the portfolio's weights "
       + "less one unit of the benchmark, which is estimated like any holding.",
    formula: String.raw`a = w_p - w_b, \qquad \mathrm{TE} = \sqrt{252\; a^{\prime}\hat\Sigma_r\, a}, \qquad \hat\Sigma_r = B\hat\Sigma_F B^{\prime} + \operatorname{diag}\!\left(\hat\sigma^2_{\varepsilon} - n\right)`,
    read: "A benchmark in the same market leaves mostly specific and style risk; one "
        + "in another market leaves the difference between the two markets.",
  },
  contribution: {
    title: "Contribution to risk",
    what: "Each holding's share of the portfolio's variance - what removing a small "
        + "amount of it would change.",
    how: "The Euler decomposition: weight times the holding's covariance with the "
       + "portfolio, over the portfolio variance. The shares sum to 100%.",
    formula: String.raw`c_i = \frac{w_i\, (\hat\Sigma_r w)_i}{w^{\prime}\hat\Sigma_r w}, \qquad \sum_i c_i = 1`,
    read: "A share far above the weight marks a holding that dominates the risk; a "
        + "negative share, one that hedges the rest.",
  },
  betaToPortfolio: {
    title: "Beta to the portfolio",
    what: "How much a holding moves when the portfolio moves by one.",
    how: "The holding's covariance with the portfolio over the portfolio variance.",
    formula: String.raw`\beta_{i,p} = \frac{(\hat\Sigma_r w)_i}{w^{\prime}\hat\Sigma_r w}, \qquad \sum_i w_i \beta_{i,p} = 1`,
    read: "Above 1 the holding amplifies the portfolio's moves, below 1 it damps "
        + "them.",
  },
  standaloneVol: {
    title: "Standalone volatility",
    what: "The holding's own forecast volatility, as if held alone.",
    how: "Its own systematic and specific variance, annualised.",
    formula: String.raw`\hat\sigma_i = \sqrt{252\left(b_i^{\prime}\hat\Sigma_F b_i - \operatorname{tr}(\hat\Sigma_F V_i) + \hat\sigma^2_{\varepsilon,i}\right)}`,
    read: "Compare with the contribution: a volatile holding can contribute little "
        + "if it is small or moves against the rest.",
  },
  blockShare: {
    title: "Risk by factor block",
    what: "Which factor blocks the systematic risk comes from.",
    how: "Each factor's Euler contribution, summed over its block, as a share of "
       + "total variance; the beta-noise correction is spread over factors in "
       + "proportion.",
    formula: String.raw`c_k = \frac{x_k\, (\hat\Sigma_F x)_k}{\hat\sigma_p^2} \cdot \frac{\hat\sigma^2_{\text{sys}}}{x^{\prime}\hat\Sigma_F x}, \qquad c_{\text{block}} = \sum_{k \in \text{block}} c_k`,
    read: "Shares sum to the systematic share. Negative is a block the portfolio is "
        + "hedged against.",
  },
  exposure: {
    title: "Factor exposure",
    what: "The portfolio's beta to a factor: the weighted sum of its holdings' "
        + "betas.",
    how: "x = B′w on the latest day.",
    formula: String.raw`x_k = \sum_i w_i\, b_{i,k}`,
    read: "Betas to orthogonalised factors are incremental: the exposure to US "
        + "equity is over and above global equity.",
  },
  portfolioVsEwma: {
    title: "Against the portfolio's own EWMA",
    what: "Whether the factor model forecasts the portfolio's daily volatility "
        + "better than the portfolio's own EWMA volatility (half-life 21 days).",
    how: "Daily QLIKE of each forecast against the portfolio's next-day return, "
       + "compared with a Diebold-Mariano test.",
    formula: String.raw`d_t = L_t^{\text{model}} - L_t^{\text{EWMA}}, \qquad L_t = \frac{r_{p,t}^2}{\hat\sigma_{p,t-1}^2} + \ln \hat\sigma_{p,t-1}^2`,
    h0: "Equal expected loss, E[d_t] = 0.",
    read: "Negative means the factor model wins. Over 54 large caps it did so on "
        + "average only with sector factors and a 21-day covariance.",
  },
  biasStatistic: AGENT_METRICS.biasStatistic,
  zStd: AGENT_METRICS.zStd,
  kupiec99: AGENT_METRICS.kupiec99,
};
