/**
 * What every number the model agent shows means, how it is computed, and - for
 * the statistical tests - what the null hypothesis is.
 *
 * Same shape as `METRICS` in metrics.ts and the same rule: one definition per
 * quantity, written once, so the leaderboard header, the candidate card and the
 * verdict's gate list cannot describe the same number three ways. The formulas
 * are transcribed from backend/core/risk.py, backend/core/selection.py and
 * backend/pipeline/run_risk.py; where those change, these must.
 *
 * Unless a definition says otherwise, a candidate's figures are on the
 * validation span. The held-out test span is opened once, for the chosen model
 * only, and appears only in the verdict.
 */

import { Metric } from "@/lib/metrics";

export const AGENT_METRICS: Record<string, Metric> = {
  // --- how the choice is made -------------------------------------------------
  selectionLoss: {
    title: "Selection loss (daily QLIKE)",
    formula: String.raw`L_t = \frac{r_t^2}{\hat\sigma_{t-1}^2} + \ln \hat\sigma_{t-1}^2

\bar L = \frac{1}{\lvert T_c \rvert} \sum_{t \in T_c} L_t, \qquad T_c = \text{validation days on which every candidate has a forecast}`,
    what: "How well the candidate's daily volatility forecast matched what happened, "
        + "averaged over every validation day. This is the number the model is "
        + "chosen on. Lower is better.",
    how: "For each day t: L = r_t² / σ²_(t−1) + ln σ²_(t−1), where σ_(t−1) is the "
       + "daily volatility the candidate forecast the evening before (betas from its "
       + "latest regression, factor covariance and specific risk as of that day) and "
       + "r_t the security's excess return. Averaged over the validation days on "
       + "which every candidate has a forecast, so all are compared on the same days. "
       + "The textbook QLIKE also subtracts ln r_t² + 1; that term depends only on "
       + "the returns, is infinite on a zero-return day and cancels in any "
       + "comparison, so it is dropped.",
    read: "Only differences between candidates mean anything; the level (around −7) "
        + "is the log of a daily variance. As a scale: a forecast whose variance is "
        + "15% too high or too low throughout loses about 0.01. QLIKE is used because "
        + "it ranks variance forecasts correctly even though the squared daily return "
        + "is a very noisy measure of the true variance, it penalises "
        + "under-forecasting more than over-forecasting, and it is not dominated by "
        + "a few crisis days as squared error is.",
    ref: "Patton (2011), Volatility forecast comparison using imperfect volatility "
       + "proxies, Journal of Econometrics 160.",
  },
  mcs: {
    title: "Model confidence set (MCS)",
    formula: String.raw`T_{\max} = \max_{i \in \mathcal{M}} \frac{\bar L_i - \bar L_{\mathcal{M}}}{\widehat{\operatorname{se}}\left(\bar L_i - \bar L_{\mathcal{M}}\right)}

H_0: \ \mathrm{E}[L_{i,t}] = \mathrm{E}[L_{j,t}] \quad \text{for all } i, j \in \mathcal{M}`,
    what: "The candidates the data cannot distinguish from the best on the selection "
        + "loss, at 90% confidence. ✓ means the candidate is in it.",
    how: "Hansen-Lunde-Nason T_max procedure on the daily loss series of all scored "
       + "candidates. For the candidates still in the set, the statistic is the "
       + "largest studentised excess of a candidate's mean loss over the set's "
       + "average. Its distribution under the null comes from a moving-block "
       + "bootstrap of the daily losses (2,000 resamples, blocks of T^(1/3) days, "
       + "which keeps their autocorrelation). While the null is rejected, the worst "
       + "candidate is removed and the test repeated; what survives is the set. A "
       + "candidate's MCS p-value is the largest elimination p-value up to its own.",
    h0: "All candidates still in the set have the same expected loss. Rejected at "
      + "p < 0.10.",
    read: "Outside the set: significantly worse than the best. Inside: not "
        + "distinguishable from it - which on a single stock is most candidates, "
        + "because its daily losses are dominated by noise. The set says the data "
        + "cannot rank these models, not that they are equally good.",
    ref: "Hansen, Lunde & Nason (2011), The model confidence set, Econometrica 79.",
  },
  objective: {
    title: "Optimise for",
    what: "Which forecast the confidence set judges the candidates on: the next "
        + "day's, the one over the horizon, or both.",
    how: "Next day: the daily loss of the forecast in force (selection loss). "
       + "Horizon: the QLIKE of the h-day forecast against the h-day return, on the "
       + "monthly validation grid. Both: a candidate must be in the confidence set "
       + "on each; if none is, the daily set is used. The calibration screen and the "
       + "simplicity rule follow either way; ties are broken on the h-day loss when "
       + "it is part of the objective.",
    read: "Next day rewards speed: a short covariance half-life follows volatility "
        + "from one day to the next, and its noise costs a single stock little. "
        + "Horizon rewards a forecast whose monthly risk is right, which a noisy "
        + "covariance gets wrong more often - over 54 large caps the slower "
        + "covariance calibrated 21-day risk in 90% of random baskets against 70%. "
        + "The horizon loss has about one observation a month rather than one a "
        + "day, so its confidence set is wider: it separates less.",
  },
  calibrated: {
    title: "Calibrated on validation",
    formula: String.raw`\lvert B - 1 \rvert \le \sqrt{2/n} \quad\wedge\quad p^{95\%}_{\text{Kupiec}} \ge 0.05 \quad\wedge\quad p^{99\%}_{\text{Kupiec}} \ge 0.05`,
    what: "Whether the candidate's forecasts got the level of risk and the frequency "
        + "of large losses right on the validation span. Only calibrated members of "
        + "the confidence set may be chosen, when any are.",
    how: "All three must hold: the bias statistic within 1 ± √(2/n); the Kupiec test "
       + "not rejected for the 95% VaR; the Kupiec test not rejected for the 99% "
       + "VaR (each at p ≥ 0.05). A statistic that could not be computed does not "
       + "count against the candidate.",
    read: "This screen exists because the loss cannot see a systematic "
        + "under-forecast on one stock, and the simplicity rule that follows "
        + "prefers heavily shrunk ridge fits - which understate risk. Without it the "
        + "agent chose such a fit for AAPL and it failed its test span.",
  },
  params: {
    title: "Effective parameters",
    formula: String.raw`k_{\text{eff}} = k \quad \text{(OLS, Huber)}

k_{\text{eff}} = \operatorname{tr}\left[(X^{\prime}WX + \lambda n D)^{-1} X^{\prime}WX\right] - 1 \quad \text{(ridge)}, \qquad D = \operatorname{diag}(0, \sigma_1^2, \dots, \sigma_k^2)`,
    what: "How many slope coefficients the candidate effectively estimates per "
        + "regression, averaged over its windows. The simplicity rule chooses the "
        + "eligible candidate with the fewest.",
    how: "Least squares and Huber: the number of factor slopes (Dimson lags add one "
       + "per lagged factor). Ridge: the trace of the hat matrix, "
       + "tr[(X′WX + P)⁻¹X′WX], less the intercept - each slope counts for less than "
       + "one as the penalty shrinks it. In the simplicity comparison the stock's own "
       + "sector factor is not charged, and two candidates within half a parameter "
       + "of each other count as equally simple, the lower loss deciding.",
    read: "Among models the data cannot rank, the simplest is the least likely to "
        + "have fitted noise. The sector factor is free because its value lies in "
        + "what stocks of one sector share - portfolio risk - which one stock's "
        + "loss cannot see.",
  },
  effectiveObs: {
    title: "Effective observations",
    formula: String.raw`n_{\text{eff}} = \frac{\left(\sum_{k=0}^{W-1} w_k\right)^2}{\sum_{k=0}^{W-1} w_k^2}, \qquad w_k = \lambda^k, \quad \lambda = 0.5^{1/h}

n_{\text{eff}} \to \frac{1+\lambda}{1-\lambda} \approx 2.89\,h \qquad (W \gg h)`,
    what: "How many equally weighted days an estimate is worth. Exponential weights "
        + "make old days count for little, so a long window with a short half-life "
        + "rests on far fewer observations than its length.",
    how: "Kish's effective sample size, (Σw)² / Σw², over the weights the estimate "
       + "uses: 1 for each day of an equally weighted window; 0.5^(k/h) for the day "
       + "k days back under a half-life of h days. A half-life of h is worth about "
       + "2.9·h days: 21 days ≈ 61, 63 ≈ 182, 126 ≈ 364.",
    read: "Compare it with the number of parameters it supports. A factor "
        + "covariance of 20 factors has 210 distinct entries; on 61 effective days "
        + "it is noisy, which the shrinkage of the blend and Ledoit-Wolf methods "
        + "exists to contain.",
  },

  // --- accuracy against simpler forecasts ---------------------------------------
  dmDaily: {
    title: "Against own EWMA, day by day (Diebold-Mariano)",
    formula: String.raw`d_t = L_t^{\text{model}} - L_t^{\text{EWMA}}, \qquad DM = \frac{\bar d}{\sqrt{\hat\Omega / T}} \;\overset{H_0}{\sim}\; \mathcal{N}(0, 1)

\hat\Omega = \hat\gamma_0 + 2 \sum_{j=1}^{q} \left(1 - \frac{j}{q+1}\right) \hat\gamma_j, \qquad q = \left\lfloor 4\,(T/100)^{2/9} \right\rfloor`,
    what: "Whether the candidate's daily forecast is more or less accurate than the "
        + "security's own EWMA volatility - the naive forecast that needs no factor "
        + "model (half-life 21 days).",
    how: "d_t = L_t(candidate) − L_t(own EWMA), the daily QLIKE difference. "
       + "DM = mean(d) / √(LRV / T), where LRV is the Newey-West long-run variance of "
       + "d with ⌊4(T/100)^(2/9)⌋ lags, so autocorrelated losses do not overstate "
       + "the evidence. Compared with a standard normal, two-sided.",
    h0: "Equal predictive accuracy: E[d_t] = 0.",
    read: "p < 0.05 with the candidate's loss lower: significantly better (green); "
        + "with it higher: significantly worse (red). Otherwise indistinguishable, "
        + "which is common for a single stock, whose own history forecasts its total "
        + "risk well.",
    ref: "Diebold & Mariano (1995), Journal of Business & Economic Statistics 13.",
  },
  qlikeMonthly: {
    title: "QLIKE (monthly)",
    formula: String.raw`L_k = x_k - \ln x_k - 1, \qquad x_k = \frac{\sigma^2_{\text{real},k}}{\hat\sigma^2_k}, \qquad \sigma^2_{\text{real},k} = 252 \cdot \widehat{\operatorname{var}}\left(r_{k+1}, \dots, r_{k+h}\right)`,
    what: "The loss of the forecast made on each monthly grid date against the "
        + "volatility actually realised over the following h trading days.",
    how: "x = σ²_realised / σ²_forecast, L = x − ln x − 1, where σ_realised is the "
       + "annualised standard deviation of the next h daily returns. Averaged over "
       + "the grid dates (one every 21 trading days).",
    read: "Zero is a perfect forecast; compare with own EWMA's figure on the same "
        + "dates. About one observation a month, so far noisier than the daily "
        + "loss; it is a diagnostic here and a hard gate in the verdict.",
    ref: "Patton (2011).",
  },
  dmMonthly: {
    title: "Against own EWMA, monthly (Diebold-Mariano)",
    formula: String.raw`DM = \frac{\bar d}{\sqrt{\hat\Omega / n}}, \qquad q = \max\left(\left\lfloor 4\,(n/100)^{2/9} \right\rfloor,\ \lceil h/\text{step} \rceil - 1\right)`,
    what: "The same comparison as day by day, on the monthly QLIKE - the forecast's "
        + "stated horizon.",
    how: "Diebold-Mariano statistic on the monthly QLIKE differences, with HAC lags "
       + "max(⌊4(T/100)^(2/9)⌋, ⌈h/step⌉ − 1): outcome windows that overlap share "
       + "returns, and the standard error has to allow for it.",
    h0: "Equal predictive accuracy against own EWMA.",
    read: "A hard gate in the verdict: significantly worse than own EWMA on the test "
        + "span (p < 0.05) is a fail, however well calibrated the model looks.",
    ref: "Diebold & Mariano (1995).",
  },
  qlikeCapm: {
    title: "QLIKE of the CAPM benchmark",
    formula: String.raw`\hat\sigma_{\text{CAPM},t} = \sqrt{\hat\beta_t^2\, \hat\sigma_{m,t}^2 + \hat\sigma_{e,t}^2}, \qquad \hat\beta_t = \frac{\widehat{\operatorname{cov}}_{252}(y, m)}{\widehat{\operatorname{var}}_{252}(m)}`,
    what: "The monthly loss of a one-factor forecast: the security's rolling beta to "
        + "global equity, that factor's EWMA volatility, and an EWMA of the CAPM "
        + "residual.",
    how: "β from a 252-day rolling covariance, both volatilities EWMA with half-life "
       + "60, combined as √(β²σ²_m + σ²_e); scored like the candidate.",
    read: "A second bar: a factor model that cannot beat one factor has not earned "
        + "its other factors.",
  },

  // --- calibration ----------------------------------------------------------------
  meanBias: {
    title: "Bias (mean ratio)",
    formula: String.raw`\bar b = \frac{1}{n} \sum_{k=1}^{n} \frac{\sigma_{\text{real},k}}{\hat\sigma_k}`,
    what: "Whether the forecast gets the level of risk right on average.",
    how: "Mean over the monthly grid dates of σ_realised / σ_forecast, the realised "
       + "volatility over the next h days divided by the forecast.",
    read: "1 is right; above 1 risk was understated, below 1 overstated. Noisy and "
        + "right-skewed (a ratio of volatilities), and blind to timing: right on "
        + "average and wrong in every month scores 1.0. The bias statistic is the "
        + "stricter check.",
  },
  biasStatistic: {
    title: "Bias statistic (h-day)",
    formula: String.raw`z_k = \frac{R_k}{\hat\sigma_{h,k} \sqrt{h/252}}, \qquad R_k = \sum_{j=1}^{h} r_{k+j}, \qquad B = \operatorname{sd}(z_1, \dots, z_n)

\operatorname{se}(B) \approx \sqrt{\tfrac{1}{2n}} \quad\Rightarrow\quad \text{accept if } \lvert B - 1 \rvert \le \sqrt{2/n}`,
    what: "Whether returns over the forecast horizon were as dispersed as forecast, "
        + "period by period.",
    how: "z_k = R_k / (σ_h,k · √(h/252)), where R_k is the sum of the h daily returns "
       + "after grid date k and σ_h,k the h-day forecast (the daily forecast plus any "
       + "lead-lag covariance, the collapsed beta for a Dimson model). The statistic "
       + "is the sample standard deviation of z_k over non-overlapping periods.",
    h0: "A correct forecast gives z unit variance. With n periods the statistic's "
      + "standard error is about √(1/2n), so the acceptance band 1 ± √(2/n) is two "
      + "standard errors.",
    read: "Above the band: risk understated; below: overstated. A gate in the verdict "
        + "and part of the calibration screen.",
    ref: "The Barra bias statistic; Connor, Goldberg & Korajczyk (2010), Portfolio "
       + "Risk Analysis.",
  },
  zStd: {
    title: "Dispersion of standardised daily returns",
    formula: String.raw`\operatorname{sd}\left(\frac{r_t}{\hat\sigma_{t-1} / \sqrt{252}}\right)`,
    what: "The daily counterpart of the bias statistic.",
    how: "Standard deviation of r_t / σ_(t−1) over every validation day, σ the daily "
       + "forecast in force.",
    read: "1 is right. Daily calibration can be right while h-day calibration is not: "
        + "a security that responds to the US market a day late has correct daily "
        + "risk and understated monthly risk.",
  },
  mzSlope: {
    title: "Mincer-Zarnowitz slope",
    formula: String.raw`\sigma^2_{\text{real},k} = a + b\, \hat\sigma^2_k + e_k`,
    what: "Whether the forecast tracks the variation in realised risk from month to "
        + "month, not just its level.",
    how: "Slope b of the regression σ²_realised = a + b·σ²_forecast + e over the "
       + "monthly grid, with HAC standard errors (lags as for the monthly "
       + "Diebold-Mariano test).",
    read: "1 is right. Below 1: high forecasts are followed by lower realised "
        + "volatility - the forecast over-reacts. Partly mechanical when volatility "
        + "moves faster than the horizon: with 21-day regimes a perfect forecast "
        + "scores about 0.5.",
    ref: "Mincer & Zarnowitz (1969).",
  },
  mzJoint: {
    title: "Mincer-Zarnowitz joint test",
    formula: String.raw`W = (\hat\theta - \theta_0)^{\prime}\, \hat V_{\text{HAC}}^{-1}\, (\hat\theta - \theta_0) \;\overset{H_0}{\sim}\; \chi^2_2, \qquad \hat\theta = (\hat a, \hat b), \quad \theta_0 = (0, 1)`,
    what: "Whether level and scale of the forecast are together consistent with an "
        + "unbiased forecast.",
    how: "Wald test of a = 0 and b = 1 jointly in the same regression, using the HAC "
       + "covariance of (a, b); χ² with 2 degrees of freedom.",
    h0: "a = 0 and b = 1: realised variance equals the forecast on average, at every "
      + "level of the forecast.",
    read: "p < 0.05 rejects. It rejects for almost every model here: the level of risk "
        + "is right and the month-to-month timing is not. A caveat in the verdict, "
        + "not a fail.",
    ref: "Mincer & Zarnowitz (1969).",
  },
  mzR2: {
    title: "Mincer-Zarnowitz R²",
    formula: String.raw`R^2 = 1 - \frac{\sum_k \hat e_k^2}{\sum_k \left(\sigma^2_{\text{real},k} - \overline{\sigma^2_{\text{real}}}\right)^2}`,
    what: "How much of the month-to-month variation in realised variance the forecast "
        + "explains - its timing, as opposed to its level.",
    how: "R² of the same regression.",
    read: "Low values are normal: realised variance over 21 days is itself a noisy "
        + "estimate, which caps what any forecast can explain.",
  },
  kupiec95: {
    title: "VaR 95% count (Kupiec)",
    formula: String.raw`\mathrm{VaR}_t = q_{\alpha}\, \frac{\hat\sigma_{t-1}}{\sqrt{252}}, \qquad I_t = \mathbf{1}\{ r_t < -\mathrm{VaR}_t \}, \qquad x = \sum_t I_t

LR_{\text{POF}} = -2 \ln \frac{(1-p)^{T-x}\, p^{x}}{(1-\hat\pi)^{T-x}\, \hat\pi^{x}} \;\overset{H_0}{\sim}\; \chi^2_1, \qquad \hat\pi = x/T, \quad p = 0.05`,
    what: "Whether the number of days the loss exceeded the 95% value-at-risk matches "
        + "the 5% it should.",
    how: "VaR_t = q · σ_(t−1) / √252, q the 95% quantile of the normal or of the "
       + "fitted Student t. A breach is a day with r_t < −VaR_t. With x breaches in T "
       + "days, LR = −2 ln[(1−p)^(T−x) p^x] + 2 ln[(1−x/T)^(T−x) (x/T)^x], p = 0.05, "
       + "compared with χ² with 1 degree of freedom.",
    h0: "The probability of a breach on any day is 5%.",
    read: "p < 0.05 rejects: too many breaches (risk understated in the tail) or too "
        + "few (too wide). Counts only how many, not when.",
    ref: "Kupiec (1995), Journal of Derivatives 3.",
  },
  kupiec99: {
    title: "VaR 99% count (Kupiec)",
    formula: String.raw`LR_{\text{POF}} = -2 \ln \frac{(1-p)^{T-x}\, p^{x}}{(1-\hat\pi)^{T-x}\, \hat\pi^{x}} \;\overset{H_0}{\sim}\; \chi^2_1, \qquad p = 0.01

q_{\alpha} = \frac{t_{\nu}^{-1}(\alpha)}{\sqrt{\nu/(\nu-2)}} \quad \text{(Student t, unit variance)}`,
    what: "The same count at the 99% level, where fat tails show.",
    how: "As at 95%, with p = 0.01 and the 99% quantile.",
    h0: "The probability of a breach on any day is 1%.",
    read: "A normal VaR is typically breached too often here; a fitted Student t "
        + "widens it. Part of the calibration screen and a verdict gate.",
    ref: "Kupiec (1995).",
  },
  christoffersen95: {
    title: "VaR 95% clustering (Christoffersen)",
    formula: String.raw`\hat\pi_{ij} = \frac{n_{ij}}{n_{i0} + n_{i1}}, \qquad n_{ij} = \#\{t : I_{t-1} = i,\ I_t = j\}

LR_{\text{ind}} = -2 \ln \frac{(1-\hat\pi)^{n_{00}+n_{10}}\, \hat\pi^{n_{01}+n_{11}}}{(1-\hat\pi_{01})^{n_{00}}\, \hat\pi_{01}^{n_{01}}\, (1-\hat\pi_{11})^{n_{10}}\, \hat\pi_{11}^{n_{11}}}

LR_{\text{cc}} = LR_{\text{POF}} + LR_{\text{ind}} \;\overset{H_0}{\sim}\; \chi^2_2`,
    what: "Whether the 95% VaR breaches came at the right rate and independently of "
        + "each other, or in bursts.",
    how: "The breach indicators are treated as a two-state Markov chain. LR_ind "
       + "compares the probability of a breach after a breach with the probability "
       + "after a quiet day; the conditional-coverage statistic LR_cc = LR_Kupiec + "
       + "LR_ind is compared with χ² with 2 degrees of freedom.",
    h0: "Breaches occur with probability 5% and independently of whether the "
      + "previous day was a breach.",
    read: "p < 0.05 rejects. The dangerous failure the count cannot see: the right "
        + "number of breaches, all in one week. A forecast held for a month clusters "
        + "them by construction; a daily refresh is the fix.",
    ref: "Christoffersen (1998), International Economic Review 39.",
  },
  horizonLoss: {
    title: "h-day loss",
    formula: String.raw`L_k = \frac{R_k^2}{v_k} + \ln v_k, \qquad v_k = \hat\sigma_{h,k}^2\, \frac{h}{252}`,
    what: "QLIKE of the h-day forecast against the h-day return itself - the only "
        + "outcome that can see covariance across days.",
    how: "L = R²/v + ln v with v = σ_h² · h/252, R the sum of the next h daily "
       + "returns, over non-overlapping periods.",
    read: "Chooses the number of lead-lag lags after the search (the fewest within "
        + "one standard error of the best), and is the loss the model itself is "
        + "chosen on when the run optimises for the horizon or for both. About one "
        + "observation a month, so far noisier than the daily loss.",
  },

  // --- the regressions ------------------------------------------------------------
  oosR2: {
    title: "Out-of-sample R²",
    formula: String.raw`R^2_{\text{oos}} = 1 - \frac{\sum_t \left(y_t - \hat b_{\tau(t)}^{\prime} f_t\right)^2}{\sum_t (y_t - \bar y)^2}, \qquad \tau(t) = \text{latest regression ending before } t`,
    what: "How much of the security's daily return the factors explain with loadings "
        + "fixed in advance - the hedge an analyst could actually have put on.",
    how: "1 − Σ(y_t − b′_(t−1) f_t)² / Σ(y_t − ȳ)² over the validation days, b_(t−1) "
       + "the betas of the latest regression ending before day t.",
    read: "Compare with the in-sample adjusted R²: a large gap means the regression "
        + "fits noise.",
  },
  adjR2: {
    title: "Adjusted R² (in sample)",
    formula: String.raw`\bar R^2 = 1 - (1 - R^2)\, \frac{n - 1}{n - k - 1}`,
    what: "The share of return variance the factors explain inside each regression "
        + "window, penalised for the number of regressors; averaged over windows.",
    how: "1 − (1 − R²)(n − 1)/(n − k − 1) per window.",
    read: "Flatters a model with many factors on a short window; the out-of-sample R² "
        + "is the honest version.",
  },
  residAutocorr: {
    title: "Residual autocorrelation (Ljung-Box)",
    formula: String.raw`Q = n(n+2) \sum_{j=1}^{10} \frac{\hat\rho_j^2}{n - j} \;\overset{H_0}{\sim}\; \chi^2_{10}`,
    what: "In how many regression windows the residuals still carry serial "
        + "correlation.",
    how: "Share of windows whose residuals reject the Ljung-Box test with 10 lags at "
       + "5%.",
    h0: "The residuals are not autocorrelated up to lag 10.",
    read: "A high share on a security that closes before New York is the case for "
        + "Dimson lags; elsewhere it points to a missing factor or stale prices.",
    ref: "Ljung & Box (1978), Biometrika 65.",
  },
  condMax: {
    title: "Condition number (worst window)",
    formula: String.raw`\kappa = \frac{s_{\max}(Z)}{s_{\min}(Z)}, \qquad Z = \text{standardised regressors}`,
    what: "How close the worst regression design came to singular.",
    how: "Largest over windows of the ratio of the largest to the smallest singular "
       + "value of the standardised regressor matrix.",
    read: "Above 200 a window's betas are not trusted and its forecast is excluded.",
  },
  maxVif: {
    title: "Maximum VIF (median window)",
    formula: String.raw`\mathrm{VIF}_j = \frac{1}{1 - R_j^2}`,
    what: "How much the most collinear factor's coefficient variance is inflated by "
        + "its overlap with the others, in a typical window.",
    how: "Per window, the largest of 1/(1 − R²_j) over factors j, R²_j from "
       + "regressing factor j on the rest; the median over windows.",
    read: "Above 100 on the orthogonalised panel is anomalous; on the raw panel it is "
        + "the factor set's nature.",
  },
  betaShift: {
    title: "Loading change per refit",
    formula: String.raw`\left\lVert \hat b_\tau - \hat b_{\tau-1} \right\rVert_1 = \sum_j \left\lvert \hat b_{j,\tau} - \hat b_{j,\tau-1} \right\rvert`,
    what: "How much the loading vector moves from one monthly regression to the next.",
    how: "Median over windows of Σ_j |b_j,new − b_j,old| (L1 norm).",
    read: "Large values with a short, equally weighted window mean unstable betas; a "
        + "longer or decayed window is the remedy the agent can test.",
  },
  nForecasts: {
    title: "Forecasts scored",
    what: "How many monthly grid forecasts the candidate was scored on.",
    how: "Grid dates in the span with a usable forecast and a realised outcome; "
       + "forecasts from ill-conditioned windows are excluded.",
    read: "Sets the width of the bias statistic's band, 1 ± √(2/n).",
  },

  // --- the verdict's other gates ----------------------------------------------------
  forecastsIssued: {
    title: "Forecasts issued",
    what: "Whether the model could produce a forecast on the test dates at all.",
    how: "Share of test grid dates with a finite forecast.",
    read: "Below 95% is a fail: a model that is silent on a twentieth of the days "
        + "cannot be relied on.",
  },
  designConditioning: {
    title: "Design conditioning",
    what: "Whether the regressions behind the test forecasts were numerically sound.",
    how: "Share of test windows whose standardised condition number exceeds 200 (or "
       + "maximum VIF exceeds 100 on the orthogonalised panel).",
    read: "Above 10% is a fail.",
  },
  stationarity: {
    title: "Stationary returns",
    what: "Whether the security's return series passes the stationarity battery.",
    how: "Augmented Dickey-Fuller and KPSS read jointly, with variance-ratio, "
       + "structural-break and stale-price checks.",
    h0: "ADF: a unit root (non-stationary). KPSS: stationarity. A series fails when "
      + "ADF cannot reject and KPSS rejects.",
    read: "A fail here is a fail of the model: nothing estimated on a non-stationary "
        + "series means what it appears to.",
  },
  circularity: {
    title: "Circularity",
    what: "Whether the security is itself an input to a factor the model uses.",
    how: "Checked against every factor's construction inputs.",
    read: "Its fit on such a factor is partly the security regressed on itself, so "
        + "the verdict carries a caveat.",
  },
};

/** The verdict's gate names, mapped to their definitions. */
export const GATE_METRIC: Record<string, string> = {
  "test sample": "nForecasts",
  "forecasts issued": "forecastsIssued",
  "design conditioning": "designConditioning",
  "stationary returns": "stationarity",
  "beats the naive forecast": "dmMonthly",
  "beats the naive forecast day by day": "dmDaily",
  "bias statistic": "biasStatistic",
  "Mincer-Zarnowitz": "mzJoint",
  "VaR 95% count": "kupiec95",
  "VaR 99% count": "kupiec99",
  "VaR 95% clustering": "christoffersen95",
  "circularity": "circularity",
};
