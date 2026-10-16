# Methodology Review

**October 2026.** A review of the model as built, done by reading the statistical
core, the construction, estimation and risk pipelines and the API behind the Risk
Lens, then re-running the stored backtests against the live database to test each
suspicion with data rather than argument. The numbers below are those
measurements; where a figure depends on the default specification it is spec
`88b9b8e7` (orthogonalised, 252-day OLS, monthly step, blended covariance, 21-day
horizon).

The engineering discipline in the project is unusually good - causal
orthogonalisation, release-event macro factors, total-return inputs, a test suite
built on processes whose answer is known. The weaknesses were concentrated in two
places: in **how the risk forecast's quality was measured**, which flattered it, and
in **the forecast itself**, which was too slow and systematically too high for
single stocks. The first had to be fixed before the second could be judged, and
the model agent (see [model-agent.md](model-agent.md)) is built on both fixes.

---

## Findings

### 1. Specific-risk shrinkage looked ahead, at the wrong peers

`run_risk.peer_specific_variance` shrank each security's specific variance 25%
toward "the cross-sectional median". It was the median residual variance over
**every window, past and future,** of **every instrument that happened to share the
spec**. On the default spec that was eighteen names, including AGG and TLT and four
factor constituents - SPY, GLD, HYG, MTUM - whose residuals are near zero by
construction. The median was 6.0% annualised, applied to a stock whose own
specific volatility is about 25%.

| AAPL, default spec | With the shrinkage | Without |
|---|---|---|
| Bias ratio | 0.964 | 0.900 |
| VaR 95% breaches (260 expected) | 248, Kupiec p = 0.43 | 208, Kupiec p = 0.001 |
| VaR 99% breaches (52 expected) | 98 | 82 |

The calibration the documentation reported for AAPL depended on it. It also
confounded the raw-against-orthogonalised comparison of §5.2 in the model
write-up: the two panels' pools had medians of 6.0% and 18.7%, an input unrelated
to the panel choice. And it made one security's forecast depend on which other
securities anyone had happened to estimate. **Removed;** the 2% floor remains.

### 2. There was no benchmark, and in the full-factor era the model lost to one

No forecast was compared with anything simpler. Since mid-2014 - when all forty
factors exist - the 40-factor forecast does no better than the security's own EWMA
volatility, the forecast that needs no factor model at all:

| Since 2014-06, monthly | MZ slope | MZ R² | QLIKE model | own EWMA (21d) | CAPM + EWMA |
|---|---|---|---|---|---|
| AAPL | 0.39 | 0.02 | 0.370 | 0.311 | 0.358 |
| AMZN | 0.55 | 0.07 | 0.339 | 0.339 | 0.340 |
| JPM | 0.42 | 0.02 | 0.477 | 0.367 | 0.429 |
| XOM | 0.51 | 0.10 | 0.363 | 0.229 | 0.299 |
| MSCI | 0.54 | 0.05 | 0.331 | 0.362 | 0.355 |
| SPY | 0.42 | 0.02 | 0.616 | 0.508 | 0.594 |

Own EWMA beats the model for five of six; XOM's model is significantly worse than
own EWMA (60-day) and the CAPM benchmark (Diebold-Mariano p ≈ 0.03). AAPL's
headline Mincer-Zarnowitz slope of 1.02 was earned in 2005-13 (1.23 there), when few
factors existed and AAPL's volatility was falling from a great height. **Fixed:**
every backtest now carries QLIKE for the model and for both benchmarks with a
Diebold-Mariano test, and the agent's verdict fails a model that is significantly
worse than own EWMA.

### 3. The forecast was held for a month, on a covariance too slow to follow volatility

Forecasts were issued at each monthly loading window and held. Every one of the
nine scored instruments failed Christoffersen at 95% (p ≈ 0); 99% breaches ran 1.6
to 3.2 times nominal for eight of nine (SPY 168 against 53 expected); standardised
returns had excess kurtosis of 5 to 45. In an experiment on 2016-2026:

| Excess kurtosis of r / σ̂ | As built | Σ refreshed daily | + 21-day EWMA, OOS specific |
|---|---|---|---|
| SPY | 16.1 | 9.3 | 3.9 |
| HYG | 45.4 | 14.3 | 4.1 |
| TLT | 8.8 | 3.0 | 1.0 |
| AAPL | 6.8 | 5.5 | 3.5 |

With a fitted Student t, Christoffersen passes at 99% rose from two of nine to five.
The forecast was also identical for every horizon: only the realised window changed.
**Fixed** as options the agent searches: daily refresh with the betas held,
out-of-sample specific risk, Student-t VaR, and the covariance half-life and blend.

### 4. Estimation noise in the betas inflated systematic risk

E[β̂′Σβ̂] = β′Σβ + tr(ΣV). With forty factors on 244 observations, k/n = 16%:

| | tr(ΣV) as share of systematic variance | of specific variance |
|---|---|---|
| AAPL | 12% | 20% |
| AMZN | 13% | 22% |
| JPM | 9% | 21% |
| XOM | 11% | 21% |
| MSCI | 16% | 20% |

That is about a third of the 9-13% volatility over-forecast every single stock
showed once the shrinkage was gone. On AAPL, subtracting it moved the bias from
0.90 to 0.94 and the Mincer-Zarnowitz slope from 0.93 to 0.98. **Fixed:** fits keep
the full HAC covariance of the slopes (`fact_loading_cov`), and the forecast can
subtract the term; the out-of-sample specific risk is corrected for the same noise.

### 5. Normal VaR under fat tails

At 99% the normal VaR passed Kupiec for one instrument of nine; a Student t with
five degrees of freedom passed six. `value_at_risk(..., "t")` existed and was never
used. **Fixed:** degrees of freedom fitted to the trailing returns standardised by
their own EWMA volatility, as a searchable option.

### 6. The reliability gate measured units, not collinearity

The condition number was computed on centred but unscaled regressors. The factors
run from 1.4% to 68% annualised volatility, so an orthogonal design already scores
about 50. Across AAPL's windows the gate's number correlated 0.86 with the ratio of
the largest to the smallest factor volatility and 0.3 with the same statistic on
standardised regressors. Rescaling the standardised-change factors - "a units
convention and nothing more" - moved it. Neither version predicted forecast error
(Spearman ≤ 0.16), and the sixteen raw-panel windows it excluded had predicted
volatilities of 0.31 to 0.62, not blow-ups. **Fixed:** standardised regressors.

### 7. Bugs of the kind the project exists to catch

- **`run_risk.write`** upserted the backtest on its sample dates and refreshed only
  the bias and Mincer-Zarnowitz columns, so a re-run kept stale exception counts,
  Kupiec and Christoffersen. Now delete-then-insert.
- **`/api/matrix/blocks`** took the newest loading window of *any* spec - the page
  sends no spec - and paired it with whichever covariance basis was toggled: β′Σβ
  on two different factor sets, the error the model write-up records fixing in
  `run_risk`. Now the spec's panel must match the basis.
- **`run_for`** upserted the spec, then removed diagnostics-blocked factors from it,
  changing its `spec_id`: a foreign-key violation the first time any factor failed
  the battery. Now the exclusion is applied per window and the spec is untouched.
- **`newey_west_cov`** was handed the unweighted design for EWMA- and Huber-weighted
  fits. Now the weighted problem's sandwich (checked against statsmodels WLS-HAC).
- **Mincer-Zarnowitz** used Newey-West's automatic lag whatever the overlap of the
  outcome windows. Now at least the overlap.

### 8. Asynchronous prices: the factor covariance had no lead-lag term

Japanese securities close about six hours before the NY-close ETF factors; the
euro and Japanese curves, `IGLT.L` and Yahoo FX are asynchronous to them too. One
day's common shock then lands in two days' returns, and the daily covariance - the
only one the model had - cannot see the second half. The h-day forecast was the
daily one times √h, so for any security or factor whose co-movement is partly a
day late the h-day risk was understated by construction. Dimson lags corrected
the betas for an early-closing security; nothing corrected the covariance.

**Built.** `covariance.lead_lag_term` adds the cross-day autocovariances of the
factor returns with the Bartlett weights of the horizon itself,
Γ₀ + Σₗ (1 − l/h)(Γₗ + Γₗ′), truncated at L lags and weighted as the covariance
method weights its own sample, then repaired to positive semi-definite. Every
forecast now carries `sigma_horizon_ann` beside the daily `sigma_pred_ann`; for a
Dimson model it uses the collapsed beta, whose cross-lag terms belong to a
multi-day return. The bias statistic is read against it, and a new h-day loss
scores it against the squared h-day return, the only outcome that can see
cross-day covariance at all. The daily forecast, its VaR and its daily loss are
unchanged by construction: a day's return does not contain tomorrow's catch-up.

On a simulated pair where one factor carries 60% of the other's move a day late,
the unadjusted h-day forecast's bias statistic is 1.30 and the adjusted one's
within 0.12 of 1 (`test_horizon.py`). The lag count is chosen by the model agent
after its search - it changes no daily forecast, so the search cannot rank it - by
the one-standard-error rule on the validation span's h-day loss.

### 9. No industry structure

Only `eq_cyclical` spoke for sectors, so whatever two stocks in one sector share
landed in their specific risks, which a diagonal Σ_ε treats as independent.

**Built.** Eleven sector factors, one SPDR sector ETF per GICS sector,
residualised against the market, the US market and the cyclical tilt, with role
`sector`: never in a default factor set, and a single stock may add only its own
(`sector: own` in the agent's search space). Whether that is worth it cannot be
judged on one stock's total risk - the sector factor mostly moves variance from
specific to systematic - so it is judged on portfolios (finding 11).

### 10. Two constructions measured the wrong thing

**`vol_rates`** was the daily change in TLT's 21-day rolling volatility. That
change is (r²ₜ − r²ₜ₋₂₁)/21 to first order, so half of it is the squared return
leaving the window three weeks earlier - an echo no security responds to. It is
now a volatility surprise: |rₜ| over yesterday's EWMA forecast (half-life 21) less
√(2/π), zero on average when volatility is as expected
(`transforms.abs_return_surprise`). Dividing by the forecast also keeps one
crisis day from dominating a covariance estimated on it.

| `vol_rates` | autocorrelation lag 21 | kurtosis | 30 largest TLT days: z on the day | 21 days later |
|---|---|---|---|---|
| change in 21-day volatility | −0.435 | 10.2 | +4.17 | **−3.47** |
| surprise | +0.004 | 3.7 | +3.59 | −0.29 |

**`liq_conditions`** took the NFCI on FRED's dates, which are the Fridays ending
each reference week. The Chicago Fed publishes the reading the following
Wednesday, so the factor placed each change five days before anyone knew it. It
now reads first releases from ALFRED (`ingest_fred.fetch_first_releases`), each on
the day it was published - 2,686 of 2,801 on a Wednesday, the rest on a Thursday
after a holiday - with observations older than the first vintage (May 2011) placed
five days after their reference date. A release on a market holiday rolls to the
next trading day.

| NFCI release factor | correlation with eq_global: same day | 5 days before | 5 days after |
|---|---|---|---|
| on the reference Friday | −0.153 | **−0.332** | −0.296 |
| on the publication day | −0.023 | −0.106 | −0.043 |

On the Friday dating the factor "explained" a third of the week's equity move
before the index existed. A security regressed on it was being fitted, in part, to
information published after the returns it explained.

### 11. Nothing tested whether the risk aggregates

Every test above is of one security's total risk. Whether the factor covariance and
the specific risks combine into the right risk for a portfolio - the part of a
factor model a single-security forecast cannot test - was untested.

**Built.** `backend/pipeline/run_portfolio.py` forecasts, every day, the risk of 54
portfolios of 54 US large caps (five per GICS sector, four in Communication
Services): the market, each sector, a within-sector pair per sector, cross-sector
spreads and twenty random baskets, from the stocks' betas, the factor covariance
and their out-of-sample specific risks, and scores them. Results in
[§ Results](#portfolio-level-validation).

### 12. The agent's choice on a single stock was unstable

Two AAPL runs chose different members of a wide confidence set. **Built:** each
run is seeded with what earlier runs chose for the security and for securities of
the same kind, re-scored on its own validation span (`store.load_pool`). That
exposed the reason: the parsimony rule took the most heavily shrunk ridge fit in a
confidence set of twenty-odd, and heavily shrunk fits understate risk. **Built:**
a calibration screen before parsimony (`selection.calibrated`), and the stock's own
sector factor is not charged as a parameter, because its value is in portfolio
risk, which one stock's loss cannot see. Results in [§ Results](#what-pooling-and-the-screen-did-to-the-choice).

### 13. The forecast lagged volatility, and the chart made it look worse (16 October)

The Risk Lens drew the *next-day* forecast against realised volatility over the
next h days. That pairs a point forecast with an h-day average: a shock in March
lifts the realised line from December, and no causal forecast can follow it
there. On the global multi-asset portfolio the 42-day forecast lines up exactly
with *trailing* 63-day realised volatility, and even a 5-day EWMA trails the
forward line by 55 days. The charts now offer two like-for-like pairs behind a
toggle: the h-day forecast against the realised volatility of the next h days,
both on the forecast date (the pairing the bias statistic scores), and the
next-day forecast against the five days centred on the day it is for
(`risk.centred_realized`), close to a point estimate of that day's volatility.
On the daily pair the multi-speed forecast peaks on 17 March 2020, a day after
realised.

The part that was the model's: one 42-day half-life. In February 2020 it took
twelve trading days to double, peaked three weeks after the market and needed 94
trading days to come halfway back; long-only portfolios lost to their own EWMA
day by day. An equal-weighted sample is worse, not better - a 252-day window has
an average age of 126 days and holds a shock at full weight for a year, then
drops it in a day. The portfolio forecast now uses a mix of 5-, 63- and 252-day
EWMAs (0.8 / 0.1 / 0.1), and its h-day forecast reverts toward the 252-day level
with a 63-day shock half-life (`portfolio.mixed_cov_path`,
`covariance.reversion_weight`). The agent can choose the single-security
equivalents: half-lives of 5 and 10 days in the blend, and the h-day reversion
in the stage after the search ([model agent](model-agent.md#23-mean-reversion-of-the-h-day-forecast)).

### Still open

- **`vol_variance_premium`** is r²-based and will dominate the covariance after a
  crash day, as `vol_rates` did; the same surprise construction would fit it.
- **The agent's per-security choices are not aggregated.** The portfolio test uses
  one configuration for every stock.
- **The 99% tail at the multi-speed covariance.** It beats the portfolios' own
  EWMA day by day, but the fitted-t 99% VaR passes Kupiec for fewer sector
  baskets (73% against 82%) and cross-sector spreads (91% against 100%) than the
  42-day covariance did, and for the all-weather fund portfolio it breaches 60
  times against about 40 expected (§ Results).
- **Ridge and the noise correction.** The tr(ΣV) correction assumes unbiased
  betas; ridge betas are shrunk, so the two together understate risk. The
  calibration screen keeps such fits from being chosen; it does not correct them.
- **JP:1904's monthly risk is understated by about two fifths** while its daily
  risk is right. The factor lead-lag term does not reach the security's own lag to
  the US factors, and no Dimson candidate has yet won its confidence set.
- **Mincer-Zarnowitz rejects for every chosen model**: the level of risk is right,
  the month-to-month timing is not.

---

## What changed

| | Before | After |
|---|---|---|
| Specific-risk shrinkage | 25% toward a look-ahead, spec-wide median | none; 2% floor |
| Benchmarks | none | own EWMA and CAPM, QLIKE and Diebold-Mariano in every backtest |
| Forecast refresh | at each monthly window | window or daily (searchable) |
| Beta noise | ignored | HAC covariance stored; tr(ΣV) correction (searchable) |
| Specific risk | in-sample residual | in-sample, or noise-corrected out-of-sample EWMA (searchable) |
| VaR | normal | normal or fitted Student t (searchable) |
| Condition number | unscaled | standardised regressors |
| Dimson | every factor | every factor, or equity and FX only; forecast and scored lag by lag |
| Specification | chosen by hand | chosen by the model agent, verdict on a held-out span |
| Multi-day risk | daily forecast × √h | daily plus lead-lag factor covariance; collapsed Dimson beta; scored on h-day returns |
| Industry structure | `eq_cyclical` only | eleven sector factors, a stock's own one searchable |
| `vol_rates` | change in 21-day rolling volatility | volatility surprise against yesterday's EWMA |
| NFCI | on its reference Friday | first release, on its publication day |
| Portfolio risk | untested | 54 portfolios of 54 large caps, five configurations |
| Agent selection | confidence set, then the fewest effective parameters | earlier choices pooled in; calibrated members only; own sector factor not charged |

The re-measured numbers - every cached specification re-estimated and every stored
backtest re-scored without the look-ahead - and the agent's verdicts on the
reviewed securities are in [§ Results](#results) below.

---

## Results

### The original model, scored honestly

Every cached specification was re-estimated (`python -m scripts.reestimate_cached`)
and every stored backtest re-scored without the look-ahead shrinkage - a second
time after the `vol_rates` and NFCI constructions were corrected, which moved no
QLIKE below by more than 0.006 and no verdict of significance. On the default specification, over each
security's full history, against the security's own EWMA volatility:

| | QLIKE model | own EWMA | Diebold-Mariano p | Bias | VaR 95% breaches |
|---|---|---|---|---|---|
| AAPL | 0.323 | 0.290 | 0.142 | 0.90 | 209 / 260 |
| AMZN | 0.359 | 0.398 | 0.293 | 0.90 | 179 / 260 |
| JPM | 0.459 | 0.330 | 0.087 | 0.88 | 176 / 219 |
| XOM | 0.343 | 0.233 | **0.034** | 0.91 | 198 / 219 |
| MSCI | 0.422 | 0.455 | 0.624 | 0.87 | 161 / 219 |
| SPY | 0.645 | 0.445 | **0.016** | 0.99 | 304 / 264 |
| HYG | 0.985 | 0.823 | **0.034** | 0.84 | 173 / 232 |
| TLT | 0.237 | 0.199 | **0.024** | 0.91 | 211 / 264 |
| GLD | 0.341 | 0.248 | 0.119 | 1.01 | 288 / 261 |

Significantly worse than the naive forecast for four of nine, better for none;
over-forecasting for eight of nine.

### Portfolio-level validation

`python -m backend.pipeline.run_portfolio`, stored in `fact_portfolio_backtest`.
Fifty-four US large caps, five per GICS sector (four in Communication Services);
every stock on the equity template, 252-day OLS refitted monthly on the
orthogonalised panel; every day a forecast for the next of 54 portfolios - the
market, eleven sector baskets, eleven within-sector pairs, eleven cross-sector
spreads, twenty random ten-name baskets - from the stocks' held betas, a causal
EWMA factor covariance, their out-of-sample specific risks and the beta-noise
correction; scored from January 2017 to October 2026, 2,334 days and 111
non-overlapping 21-day periods.

**The residuals.** Without sector factors the out-of-sample residuals of two stocks
in the same sector correlate 0.22 on average - risk the model books as specific and
diversifiable - against 0.006 across sectors. With each stock's own sector factor
the within-sector figure is −0.010. The first principal component of the
standardised residuals falls from 8.2% of their variance to 5.0% (1/54 = 1.9% if
they were independent).

**Calibration**, factor covariance half-life 42 days. Share of portfolios whose
21-day bias statistic is inside 1 ± √(2/111) = 1 ± 0.13, median statistic in
brackets; and the share whose 99% VaR, with a Student t fitted to the trailing
standardised returns, passes Kupiec:

| | No sector: in band | Own sector: in band | No sector: t-VaR 99% | Own sector: t-VaR 99% |
|---|---|---|---|---|
| Market | 0% (1.15) | 0% (1.14) | 0% | 100% |
| Sector baskets (11) | 64% (1.09) | 73% (1.02) | 27% | 82% |
| Within-sector pairs (11) | 73% (0.94) | **100%** (1.02) | 73% | 100% |
| Cross-sector spreads (11) | 27% (1.17) | **82%** (1.02) | 27% | **100%** |
| Random baskets (20) | 70% (1.11) | 90% (1.10) | 60% | 80% |

The pattern is the one a missing sector structure predicts. A sector basket's
shared residual is booked as independent, so its risk is understated; a pair's is
counted twice, so it is overstated (0.94); a cross-sector spread holds two
baskets' worth of the first error. The sector factor removes all three. The
daily standardised-return dispersion of sector baskets moves from 1.09 to 1.02,
of cross-sector spreads from 1.13 to 1.00. The market portfolio is under-forecast
at the 21-day horizon either way (1.14), while its daily dispersion is 1.03: the
market's month-to-month risk is not the sum of its days', which no specific-risk
or sector change addresses.

**Speed against calibration.** With a 21-day factor covariance half-life - the
agent's choice for every single stock - the factor model, with sector factors,
beats each portfolio's own EWMA volatility day by day on average (daily QLIKE
−0.003 against it; the market portfolio, 9 of 11 cross-sector spreads and 12 of 20
random baskets do better), which no configuration without sector factors does
(+0.027 at 21 days, +0.071 at 42). The price is calibration: random baskets in band
fall from 90% to 70%, and the t-VaR passes Kupiec for 30% of them rather than 80%. A
fast covariance follows volatility; a slow one sizes the tail. The two half-lives
answer different questions, and a single setting cannot serve both.

**Lead-lag on US large caps.** Two lags of cross-day covariance changed the 21-day
calibration little (sector baskets 73% → 82% in band, pairs 100% → 91%). The
factors these stocks load on close in New York, as the stocks do; the adjustment
is for asynchronous ones, and the agent kept it at zero for every US stock.

**What remains.** Long-only portfolios do not beat their own EWMA at the slower
covariance, and the market portfolio's 21-day returns are 1.14 to 1.19 times as
dispersed as their forecast in every configuration, against 1.03 to 1.06 for its
daily returns.

### Fund portfolios

The Risk Lens's Portfolio view (`backend/pipeline/portfolio_risk.py`) runs the
same forecast on portfolios of funds. Each fund sits on its asset-class template,
and a sector fund also carries its sector factor. The scores are on the
portfolio's own returns since the first day every holding has betas, at a 21-day
horizon. ΔQLIKE is the model's daily loss minus the own-EWMA's (negative wins);
"ex worst" is the bias statistic without its single most extreme block:

| Portfolio (from) | Half-life | Bias (ex worst) | Band | ΔQLIKE (DM p) | t-VaR 99% breaches (Kupiec p) |
|---|---|---|---|---|---|
| 60/40 SPY/AGG (2004) | 42 | 0.99 (0.95) | ±0.09 | +0.093 (<0.001) | 75 / 52 (0.002) |
| | 21 | 1.01 (0.97) | | +0.029 (0.027) | 87 / 52 (<0.001) |
| Global multi-asset (2014) | 42 | 1.69 (0.91) | ±0.12 | +0.070 (0.005) | 40 / 28 (0.029) |
| | 21 | 1.77 (0.93) | | −0.004 (0.13) | 49 / 28 (<0.001) |
| All-weather (2009) | 42 | 0.98 (0.95) | ±0.10 | +0.025 (0.001) | 55 / 40 (0.026) |
| | 21 | 1.00 (0.97) | | −0.000 (0.92) | 62 / 40 (0.001) |
| US sector funds (2019) | 42 | 1.82 (0.81) | ±0.15 | +0.098 (0.004) | 12 / 15 (0.40) |
| | 21 | 1.91 (0.85) | | −0.007 (0.06) | 17 / 15 (0.64) |
| Global equity by region (2009) | 42 | 0.94 (0.86) | ±0.10 | +0.066 (<0.001) | 56 / 41 (0.023) |
| | 21 | 0.96 (0.91) | | +0.001 (0.66) | 59 / 41 (0.007) |

Two of the five are far outside the band only because of one block, the one
that began on 21 February 2020 (z = −17 for the multi-asset portfolio). Without
it, both are at or below 1. The view therefore reports the worst block beside the
formal statistic. The trade between speed and tails is the same one the stock
portfolios showed. A 21-day covariance ties with the portfolio's own EWMA where a
42-day one loses to it, and it breaches the 99% VaR more often. The 60/40 loses
at either speed.

### Multi-speed covariance and h-day reversion (16 October)

The 54-stock universe again, 21-day horizon, own sector factors, scored from
2017. Mean daily loss minus each portfolio's own EWMA's (negative is better than
own EWMA), share of portfolios beating own EWMA by the daily loss, share with
the bias statistic in band (median in brackets), share whose fitted-t 99% VaR
passes Kupiec:

| | Daily loss − own EWMA | Beats own EWMA | In band | t-VaR 99% |
|---|---|---|---|---|
| 42-day | +0.042 | market 0, sectors 0, random 0, cross 36% | market 0% (1.14), sectors 73%, random 90% (1.10), cross 82% | sectors 82%, cross 100%, random 80% |
| 21-day | −0.003 | market 1, sectors 55%, random 60%, cross 82% | market 0% (1.18), sectors 73%, random 70% (1.12), cross 82% | sectors 55%, cross 100%, random 30% |
| Multi-speed, reverting | **−0.046** | market 1, sectors 100%, random 100%, cross 91% | market 100% (1.05), sectors 73%, random **100%** (1.03), cross 82% | sectors 73%, cross 91%, random 80% |

The reversion changes no daily number; it moves the 21-day bias statistics
toward 1 (random baskets 1.06 to 1.03, the market 1.09 to 1.05). On the fund
portfolios at a 63-day horizon the daily Diebold-Mariano statistic against own
EWMA goes from +2.8, +4.7 and +3.3 (global multi-asset, 60/40, all-weather; the
42-day covariance losing) to −3.3, −1.8 and −1.0. The global multi-asset bias
statistic rises from 1.45 to 1.55, all of it in the block from 19 December 2019:
without it, 0.82. The weights were chosen on fund portfolios up to June 2020 and
held out after; the stock universe was not used to choose them.

### What the agent chose

The agent was run on seven securities - five US stocks, one Tokyo-close stock and
one fixed-income ETF that is itself a factor input - with DeepSeek driving and a
budget of four refinement rounds. Metrics are on the held-out test span, which the
search never saw; "original" is the original model scored on the same span
afterwards, which cannot have influenced the choice.

**Second pass (14 October, after the changes above).** Thirty-six candidates at
most, of which up to eight carried in from earlier runs; sector factors, the
lead-lag stage, the calibration screen, and the sector factor free in the
parsimony rule.

| Security | Verdict | Candidates (LLM, pool) / in MCS (calibrated) | Chosen | Test QLIKE: agent / original / own EWMA | DM p vs own EWMA: agent / original | Mean bias: agent / original | VaR99: agent / original (exp) | Lead-lag | Caveats |
|---|---|---|---|---|---|---|---|---|---|
| HYG | pass, caveats | 30 (17, 7) / 21 (0) | `orth 7 factors 504d ridge l=0.1 \| ewma/21/504 spec-ewma21 debias daily t-VaR` (pool) | 0.465 / 0.580 / 0.639 | 0.234 / 0.772 | 0.98 / 0.71 | 9 / 8 (8) | 1 | Mincer-Zarnowitz, circularity |
| JP:1904 | **fail** | 35 (19, 8) / 27 (7) | `orth template 756d ridge l=0.1 \| blend/21/756 a=0.4 spec-ewma21 debias daily t-VaR` | 1.077 / 1.353 / 1.048 | 0.418 / 0.355 | 1.04 / 0.94 | 7 / 8 (8) | 2 | bias statistic, Mincer-Zarnowitz, VaR 95% count, VaR 95% clustering |
| US:AAPL | pass, caveats | 34 (19, 8) / 25 (16) | `orth template 252d ridge l=0.1 hl126 +sector \| ewma/21/504 spec-ewma63 debias daily t-VaR` | 0.375 / 0.358 / 0.364 | 0.703 / 0.907 | 1.02 / 0.85 | 12 / 11 (8) | 0 | Mincer-Zarnowitz, VaR 95% clustering |
| US:AMZN | pass, caveats | 28 (13, 8) / 27 (11) | `orth template 378d ridge l=1 hl126 \| ewma/21/504 spec-ewma126 debias daily t-VaR` (pool) | 0.218 / 0.237 / 0.254 | **0.023** / 0.646 | 1.04 / 0.87 | 9 / 8 (8) | 0 | Mincer-Zarnowitz |
| US:JPM | pass, caveats | 35 (20, 8) / 31 (17) | `orth template 756d huber hl126 +sector \| ewma/21/756 spec-ewma21 debias daily t-VaR` | 0.322 / 0.338 / 0.310 | 0.492 / 0.638 | 1.00 / 0.89 | 16 / 16 (8) | 0 | Mincer-Zarnowitz, VaR 99% count |
| US:MSCI | pass, caveats | 34 (19, 8) / 33 (20) | `orth template 756d huber hl126 +sector \| blend/60/756 a=0.6 spec-ewma21 debias daily t-VaR` | 0.420 / 0.319 / 0.485 | 0.167 / 0.073 | 0.92 / 0.83 | 9 / 10 (8) | 1 | Mincer-Zarnowitz |
| US:XOM | **fail** | 34 (19, 8) / 29 (15) | `orth template 756d ridge l=0.1 hl126 +sector \| ewma/21/504 spec-ewma63 debias daily t-VaR` | 0.170 / 0.173 / 0.145 | 0.119 / 0.312 | 1.06 / 0.91 | 14 / 11 (8) | 0 | Mincer-Zarnowitz, VaR 95% count, VaR 99% count |

**First pass (13 October, morning; before the changes above).** Thirty candidates,
no pool, no sector factors, no screen.

| Security | Verdict | Candidates (LLM) / in MCS | Chosen | Test QLIKE: agent / original / own EWMA | DM p vs own EWMA: agent / original | Mean bias: agent / original | VaR99: agent / original (exp) | Caveats |
|---|---|---|---|---|---|---|---|---|
| HYG | pass, caveats | 29 (23) / 26 | `orth 7 factors 252d ols \| ewma/21/504 spec-ewma21 debias daily t-VaR` | 0.450 / 0.563 / 0.639 | 0.226 / 0.734 | 0.95 / 0.72 | 9 / 9 (8) | Mincer-Zarnowitz, circularity |
| JP:1904 | **fail** | 30 (23) / 22 | `orth template 378d ridge l=0.1 \| ewma/21/504 spec-ewma21 debias daily t-VaR` | 1.129 / 1.348 / 1.048 | 0.265 / 0.356 | 1.00 / 0.94 | 7 / 8 (8) | beats the naive forecast day by day, bias statistic, Mincer-Zarnowitz, VaR 95% count, VaR 95% clustering |
| US:AAPL | pass, caveats | 29 (23) / 23 | `orth template 252d ridge l=0.1 \| ewma/21/504 spec-ewma63 debias daily t-VaR` | 0.357 / 0.358 / 0.364 | 0.794 / 0.902 | 1.01 / 0.85 | 13 / 11 (8) | Mincer-Zarnowitz, VaR 95% clustering |
| US:AMZN | pass, caveats | 29 (23) / 28 | `orth template 252d ridge l=1 \| ewma/21/504 spec-ewma126 debias daily t-VaR` | 0.214 / 0.238 / 0.254 | 0.034 / 0.668 | 1.03 / 0.87 | 10 / 8 (8) | Mincer-Zarnowitz |
| US:JPM | pass, caveats | 29 (23) / 23 | `orth template 252d ridge l=1 \| ewma/21/504 spec-ewma21 debias daily t-VaR` | 0.359 / 0.338 / 0.310 | 0.138 / 0.649 | 1.09 / 0.89 | 16 / 16 (8) | Mincer-Zarnowitz, VaR 99% count |
| US:MSCI | **fail** | 29 (23) / 28 | `raw template 252d ridge l=0.3 \| ewma/21/504 spec-ewma21 debias daily t-VaR` | 0.506 / 0.319 / 0.485 | 0.016 / 0.073 | 1.02 / 0.83 | 12 / 10 (8) | beats the naive forecast, beats the naive forecast day by day, Mincer-Zarnowitz |
| US:XOM | pass, caveats | 26 (20) / 20 | `orth template 252d ridge l=0.3 \| ewma/21/504 spec-ewma21 debias daily t-VaR` | 0.156 / 0.173 / 0.145 | 0.638 / 0.325 | 1.05 / 0.91 | 18 / 11 (8) | VaR 95% count, VaR 99% count |

**The risk half converged, both times.** Every chosen model refreshes the forecast
daily, removes the beta noise term, tracks specific risk out of sample and uses a
fitted Student-t VaR; all but one use a 21-day EWMA covariance. That is findings
3, 4 and 5 rediscovered, fourteen times, by a search that was not told them.

**The sector factor is chosen where the stock has one.** Four of the five US
stocks took their own sector factor in the second pass; AMZN kept the
specification earlier runs had chosen. On AAPL the candidates with it explained
about 40% of the next month's daily returns out of sample against 32% without.

**The lead-lag stage found the asynchronous securities.** It took two lags for
the Tokyo-close JP:1904 and one each for HYG and MSCI, and none for the other
three US stocks, whose factors close when they do. It did not rescue JP:1904: the
verdict's bias statistic - the dispersion of 21-day returns over their forecast -
is 1.66, against 1.67 in the first pass and 1.80 for the original model, while
the mean bias of its daily forecasts is 1.04. Its daily risk is right and its
monthly risk is understated by about two fifths. What the factor covariance's
lead-lag term cannot supply is the security's own lag to the US factors; that is
the Dimson betas' job, and no Dimson candidate entered JP:1904's confidence set.

**The loss.** The chosen model's test QLIKE is lower than the original's for five
of seven, and significantly better than own EWMA for AMZN. MSCI, failed in the
first pass for a raw-panel ridge fit, passes with caveats; XOM fails on a third
flag (14 99% breaches against 8 expected, where the original had 11).

### What pooling and the screen did to the choice

Pooling alone did not steady the choice; it exposed why it was unsteady. With
JPM's winner - ridge, λ = 1 - carried into AAPL's search, the confidence set held
26 candidates, the parsimony rule chose the most heavily shrunk, and it failed its
test span with three calibration flags. Every ridge λ = 1 member of three
confidence sets had already breached its 99% VaR 1.2 to 2.5 times as often as
expected on validation: shrunk betas understate β′Σβ, and the beta-noise
correction, which assumes unbiased betas, subtracts on top. The daily loss cannot
see it on one stock. Hence the calibration screen (model-agent.md §1).

With pooling and the screen together, consecutive runs choose the same model:

| AAPL run | Pool | Screen | Chosen | Test QLIKE | Verdict |
|---|---|---|---|---|---|
| 1 | - | - | 378d ridge λ 0.3, specific EWMA 21 | 0.314 | pass, caveats |
| 2 | - | - | 252d ridge λ 0.1, specific EWMA 63 | 0.357 | pass, caveats |
| 3 | yes | - | 252d ridge λ 1 (JPM's) | 0.365 | **fail** |
| 4 (13 Oct) | yes | yes | 252d ridge λ 0.1 hl126, specific EWMA 63 | 0.371 | pass, caveats |
| 5 (14 Oct) | yes | yes, sector free | the same with its sector factor | 0.375 | pass, caveats |

AMZN chose the same specification in all three pooled runs (378-day ridge λ = 1,
weights half-life 126, test QLIKE 0.218, significantly better than own EWMA), and
JPM a 756-day Huber with its sector factor in both screened runs, differing only in
the covariance window. On AAPL
the test QLIKE of the steadier choice is not better than run 1's luckier one; what
pooling buys is that the choice no longer depends on what the language model
happened to propose in one run.

**What still fails everywhere.** The Mincer-Zarnowitz joint test rejects for all
seven: the forecasts get the level right and the month-to-month timing wrong. The
99% VaR is still breached too often for JPM and XOM even with fitted tails, and
JP:1904's multi-day risk is understated.
