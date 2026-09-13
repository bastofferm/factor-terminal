# Return-Based Multi-Asset Factor Model

**As built.** A revision of the original concept note, rewritten against the
implementation. Where the two differ, this document follows the implementation and
says why the design changed.

Written in English to match the rest of the repository; the original note is in
German.

Panel as of 2026-09-11 · 40 factors · 198 instruments · 50 level series ·
200,072 factor observations · 221 tests.

---

## 1. What the model is

A factor model that uses no holdings and no look-through. Every exposure is
estimated from observed returns. The question it answers is not *what does this
fund own* but *what does its return actually move with* — which is the only
question answerable for a fund that reports holdings quarterly, or for a strategy
whose positions change faster than its disclosures.

The consequence is a standing obligation the original note states plainly and this
implementation takes seriously: a return-based model measures realised
sensitivities, so it must be defended against spurious correlation, stale pricing,
non-stationarity and short histories. Most of the engineering below is that defence.

```
r_i,t − r_f,t  =  α_i  +  β_i′ F_t  +  ε_i,t
Var(r_p)       =  β_p′ Σ_F β_p  +  w′ Σ_ε w
```

All factors are daily log excess returns over cash (`FRED:DFF`) in USD.

---

## 2. Input data

### 2.1 What the concept note asked for, and what the data supported

Section 2.2 of the note specifies nine blocks of daily factor proxies. Each was
audited against what was actually obtainable before anything was built. The audit
is the most consequential part of the project, because a factor built on the wrong
series is worse than a missing factor: it looks fine and is wrong.

| Block | Verdict | Action |
|---|---|---|
| Rates | Excellent — complete US, EA and JP curves daily from FRED, ECB, BOJ, MOF | Used as specified |
| Credit | Excellent — full ICE BofA OAS ladder (AAA→CCC) plus IG/HY/loan/EM ETFs | Used as specified |
| Equity Style | Good — style ETFs plus Fama-French and AQR for validation | Used, with the caveat in §2.3 |
| Volatility | Good — VIX futures ETFs, not index levels | Used as specified |
| Commodities | Good — total-return ETFs | Used, **not** the futures series; see §2.2 |
| Alt. Risk Premia | Good — constructible in-house from the return panel | Built rather than sourced |
| **Equity Market** | **Insufficient** — only price-return index levels | 12 total-return ETFs added |
| **Rates (Gilt)** | **Missing** — no daily UK curve anywhere in the warehouse | `IGLT.L` added as a total-return proxy |
| **Liquidity** | **Thin** — TED and LIBOR-OIS both discontinued | SOFR, EFFR, OBFR, TGCR, NFCI, ANFCI, STLFSI4 added |

Two gaps could not be closed and are carried as documented approximations rather
than quietly patched:

- **FX carry** has no forward points in the warehouse, so it is built from
  policy-rate differentials across the three non-USD policy rates available. It is
  narrower and noisier than a real G10 carry basket.
- **EA and JP rates** come from the ECB, BOJ and MOF via the warehouse, which this
  project does not refresh. They lag by design and the Data Health page says so.

### 2.2 Four decisions that are easy to get wrong

These are recorded here because each one produces a plausible-looking factor when
done carelessly.

**Total return, not price return.** Yahoo's `^`-prefixed index levels carry no
dividends. Used as equity factors they bias every equity beta down by the dividend
yield. They are flagged `is_total_return = false` and excluded from construction.

**Commodity futures are not a return series.** Yahoo's `=F` series are stitched
front-month prices with an artificial jump at every roll. The concept note is
explicit that roll and collateral return matter, so the commodity block uses
total-return funds, which include both.

**Low-frequency macro is never forward-filled.** Filling a weekly series into a
daily regressor manufactures information and understates standard errors — §3 of
the note, *Grundregel*. Such series become sparse release-event factors: the
standardised change lands on the publication day and the factor is exactly zero in
between.

**The trading calendar excludes crypto.** Crypto prices seven days a week. Included,
it added 1,229 weekend rows on which every other factor is missing, which made
252-row estimation windows span fewer than 252 trading days and pushed equity
factors below the coverage threshold in the covariance matrix. This was found only
because the matrix endpoint started returning 8 factors instead of 39.

### 2.3 The weakest inputs, stated plainly

The style block uses long-only ETFs residualised against market and sector, not
true long-short portfolios. §4 below quantifies how close that gets to the academic
factors. Quality and low-volatility are the weakest proxies in the set and should
not be leaned on.

---

## 3. Stationarity

Every model input must be a stationary return series, and the note treats this as a
precondition rather than a diagnostic. A single ADF test is useless as a gate — on
daily returns it rejects almost always — so the battery is built around what
actually goes wrong with financial series.

| Test | What it catches | Gating |
|---|---|---|
| ADF × KPSS, read **jointly** | A level that should have been differenced | Blocks |
| Zivot-Andrews | A structural break masquerading as a unit root | Flags |
| Lo-MacKinlay variance ratio | Stale or smoothed pricing (§6.3 of the note) | Flags |
| Ljung-Box | Autocorrelation — the second stale-pricing signal | Flags |
| ARCH-LM | Volatility clustering | **Records, never gates** |
| Zero-return share | Illiquidity | Flags |

Two points of method:

**ADF and KPSS have opposite nulls, and reading them together gives four verdicts
rather than one**: stationary (ADF rejects, KPSS does not), unit root (the reverse),
break-or-heteroskedasticity (both reject), inconclusive (neither). Only the second
blocks a series. Reading either test alone discards this.

**ARCH effects are recorded and never gated.** A GARCH process is strictly
stationary. Gating on ARCH-LM would exclude essentially every financial series, and
the reason to know about clustering is that it argues for HAC standard errors — not
against the factor.

Current state: 32 pass, 8 warn, 0 fail on the trailing 252-day window.

---

## 4. The factor universe

Forty factors across the nine blocks, arranged in a hierarchy. Level 0 factors are
raw; higher levels are orthogonalised against the levels above them, which is what
keeps a loading interpretable — see §5.

| Block | Factors | Ann. vol range |
|---|---|---|
| Equity Market | eq_global, eq_us, eq_europe, eq_japan, eq_em, eq_size, eq_cyclical | 10.6 – 27.3% |
| Equity Style | sty_value, sty_momentum, sty_quality, sty_lowvol | 13.4 – 20.2% |
| Rates | rt_us_level, rt_us_slope, rt_us_curve, rt_ea_level, rt_jp_level, rt_uk, rt_breakeven | 1.4 – 12.6% |
| Credit | cr_ig, cr_hy, cr_loans, cr_em_hard, cr_em_local | 5.8 – 10.8% |
| FX | fx_usd, fx_jpy, fx_em, fx_carry | 7.5 – 11.5% |
| Commodities | cm_broad, cm_energy, cm_gold, cm_agri, cm_industrial | 16.7 – 37.6% |
| Volatility | vol_equity, vol_rates, vol_variance_premium | 10.0 – 67.7% |
| Liquidity & Stress | liq_conditions, liq_funding, liq_risk_off | 10.0 – 20.8% |
| Alternative Risk Premia | arp_trend, arp_xs_momentum | 5.5 – 19.5% |

Volatilities are of the **raw** series, before orthogonalisation.

### Validation against published factors

Fama-French and AQR series lag by one to two months and can never drive a daily
model. They are kept for one purpose: to confirm that a construction measures what
its name claims. The table below is the strongest evidence in the project, and it
is also the clearest demonstration of why the hierarchy exists.

| Factor | Reference | Raw | Orthogonalised | Overlap |
|---|---|---|---|---|
| `eq_global` | Mkt-RF | **0.963** | 0.963 | 4,551 |
| `eq_size` | SMB | 0.914 | **0.874** | 4,287 |
| `sty_momentum` | Mom | 0.207 | **0.638** | 3,006 |
| `sty_value` | HML | 0.194 | **0.499** | 3,006 |
| `sty_lowvol` | BAB (AQR) | −0.002 | **0.199** | 3,363 |
| `sty_quality` | RMW | −0.145 | **0.143** | 2,943 |

Read the style rows carefully. As raw long-only ETFs they correlate with their
academic counterparts at 0.19 or below — `sty_lowvol` at zero, `sty_quality`
negatively — because market beta swamps the style. Removing market and sector
raises momentum to 0.64 and value to 0.50. That is the orthogonalisation doing
exactly the job it exists for, and it is measurable rather than asserted.

It also shows the honest limit: quality reaches 0.14 and low-volatility 0.20. These
two are weak proxies. They are kept because the block would otherwise be
incomplete, not because they are good.

Value against momentum comes out at −0.11 after orthogonalisation, reproducing the
well-known negative relationship in sign though not in the −0.3 to −0.5 magnitude
usually reported for true long-short portfolios.

---

## 5. Orthogonalisation, and why it is causal

Regressing a security on both `eq_global` and raw `eq_japan` is a poor idea: the
two are strongly correlated, the design matrix is ill-conditioned, and the betas
come out as large offsetting numbers that mean nothing individually. The block
hierarchy of §4 of the note removes the overlap, so each block's loading answers a
specific question.

The implementation choice that matters is **when** the residualisation is fitted.

Full-sample residualisation is exactly orthogonal and wrong: it puts information
from the end of the sample into factor values at the beginning, which flatters the
model precisely where it is being judged. This implementation fits on a trailing
504-day window, refitted monthly. The cost is measurable and was measured:

| Method | Residual correlation | Lookahead |
|---|---|---|
| Rolling 504d (used) | −0.05 | none |
| Expanding | −0.34 | none |
| Full sample | 0.00 | yes |

A residual correlation of −0.05 instead of a clean zero is the price of not
cheating, and it is the right trade.

**Consequence for reading the model.** The orthogonalised series is what the
regression, the covariance and the risk forecast consume — but it is not what the
factor earned. `eq_us` returns 8.6% a year at 18.5% volatility; its residual after
removing `eq_global` returns −0.2% at 4.2%. The application keeps these on separate
pages for that reason: the Factor Explorer describes the model, the Raw Explorer
describes the data.

---

## 6. Exposure estimation

Rolling regression of an instrument's excess return on the factor panel, with the
controls the note asks for exposed as user choices rather than hard-coded:

- **Estimation window** 63 – 756 days
- **Roll-forward step** 1d / 1w / 1m / 3m / 6m
- **Estimator** OLS, Huber (robust), or ridge
- **Weighting** equal or exponentially decayed
- **HAC lags** automatic or explicit
- **Dimson lead-lag** 0–2, for instruments pricing after the factor close
- **Winsorisation** optional at 1% / 99%

Every parameter combination hashes to a `spec_id`, so re-requesting a view an
analyst has already seen is a table read rather than several hundred regressions.

Diagnostics stored per window: R², adjusted R², F and its p-value, RMSE, annualised
residual volatility, Durbin-Watson, condition number, maximum VIF, Ljung-Box and
ARCH-LM on the residuals, and the change in the loading vector since the previous
window.

### Beta stability

§11 of the note treats a jumping loading vector as a drift alert. Two measures are
kept: the L1 change in the whole vector and its correlation with the previous
window.

The comparison is made on the factors **common to both windows**, with the size of
that common set recorded. An earlier version refused to compare whenever the usable
factor set changed, which put gaps in the chart at exactly the moments the exposure
set moved — 169 of 178 missing points — which is when drift is most worth seeing.
Since the L1 sum falls mechanically as the basis narrows, the overlap count is
reported alongside it.

---

## 7. Covariance and risk

```
Σ_F  =  a · Σ_EWMA  +  (1 − a) · Σ_shrunk
Σ_r  =  B Σ_F B′  +  Σ_ε
```

Four estimators: sample, EWMA, Ledoit-Wolf shrinkage to a constant-correlation
target, and a blend. Positive semi-definiteness is checked and, if needed, repaired
by eigenvalue clipping followed by a rescale that puts the factor volatilities back
where they were — clipping alone inflates the diagonal and shifts every risk number.

Specific risk is shrunk 25% toward the cross-sectional median and floored at 2%
annualised, per §6.4 of the note: a single short or unusually quiet history should
not produce a falsely precise idiosyncratic risk.

**The covariance page is computed on the orthogonalised factors and must be.** The
loadings are loadings on that set, and portfolio risk is β′Σβ, so Σ has to be the
covariance of the same series the βs refer to.

### Forecasts are scored, not asserted

For a forecast dated *t*, the loadings come from a window ending at or before *t*
and the covariance from returns up to *t*; the realised volatility it is compared
against covers *t+1 … t+h*. The lag is enforced structurally, not by convention.

Scoring follows §11:

- **Bias statistic** — standard deviation of returns divided by the forecast in
  force. Target 1.
- **Mincer-Zarnowitz** — realised variance regressed on predicted variance; joint
  test of α = 0, β = 1.
- **Kupiec** — is the VaR exception *count* right?
- **Christoffersen** — are the exceptions *independent*, or do they cluster?

Kupiec and Christoffersen are both reported because they fail differently, and the
second failure is the dangerous one: a model can have exactly the right number of
breaches and have them all in one week.

Forecasts built on ill-conditioned windows are stored, flagged and excluded from
scoring. With forty factors on a 252-day window, a period where few factors yet
existed leaves the design near-singular; AAPL produced a 360% predicted volatility
that way, which then dominated the Mincer-Zarnowitz regression as a single leverage
point. The gate is condition number > 200 or max VIF > 100, which excludes about 3%
of windows, almost all in 2002–03.

### A caveat on reading Mincer-Zarnowitz

A slope below 1 is **partly mechanical** when volatility moves faster than the
measurement horizon, because realised volatility over a forward window blends
regimes while the forecast refers to one. Measured on simulated data with a
*perfect* point-in-time forecast and a 21-day horizon:

| True regime length | MZ slope |
|---|---|
| 21 days | 0.50 |
| 126 days | 0.93 |
| 504 days | 1.02 |

So a slope under 1 on real data is not by itself evidence that the model
over-reacts. The application says so where the statistic is displayed.

---

## 8. What is not built

The concept note also specifies portfolio optimisation (§8), performance
attribution (§10) and a mixed-frequency state-space layer (§3.3). None of these are
implemented. The covariance matrix is optimiser-ready — positive semi-definite, with
interpretable exposures — but no optimiser sits on top of it.

Macro release-surprise factors (§3.4) are implemented only as the sparse
release-event transform described in §2.2 above; there is no announcement-window
lead-lag structure and no Kalman-filtered latent macro state.

---

## 9. Validation approach

The test suite does not check that functions return numbers. Each statistical
routine is fed a process whose truth is known, and the assertion is that it reaches
the right conclusion:

- a random walk must fail the stationarity gate, an AR(1) with φ = 0.5 must pass
- a GARCH series must flag ARCH effects **without** failing
- a smoothed series must show a variance ratio above 1
- a forecast at half the true volatility must produce a bias ratio of 2
- clustered VaR breaches must be caught by Christoffersen even when Kupiec sees
  nothing wrong with the count
- shrinkage must beat the sample covariance when factors outnumber observations

This approach earned its keep. Five defects were found by tests or by the
diagnostics rather than by inspection, and each would have silently corrupted
results:

1. `arch`'s `VarianceRatio` consumes a **price level** and differences internally.
   Passing returns differenced twice, giving VR ≈ 1/q and flagging every clean
   factor as mean-reverting.
2. The ridge penalty was scaled by σ⁻² instead of σ², so λ = 1 shrank every loading
   to zero.
3. Crypto's weekend prints polluted the trading calendar, cutting the covariance
   matrix from 39 factors to 8.
4. `fx_carry` was the yen short wearing another name — correlation −0.87 with
   `fx_jpy` before it was residualised against it.
5. Stability columns held IEEE NaN rather than NULL, because pandas converts None
   to NaN in a float column and `double precision` accepts it. `IS NULL` never
   matched, `count()` counted them, and one NaN turned an average over the whole
   column into NaN.

---

## 10. Limits

- **Style proxies are long-only ETFs**, not long-short portfolios. Validated
  correlations after orthogonalisation: momentum 0.64, value 0.50, low-vol 0.20,
  quality 0.14.
- **FX carry is an approximation** from three policy-rate differentials.
- **EA and JP rates lag**, being warehouse-sourced with no refresh here.
- **Short histories.** Style factors begin 2011–2013, `liq_funding` in 2018. Any
  window longer than the shortest factor's history silently drops it.
- **No optimiser, no attribution, no macro state-space.**
- **The model can only measure risks that have appeared in returns.** Derivatives,
  concentration and tail trades remain invisible, which is why residual risk has a
  floor and why model uncertainty is surfaced rather than hidden.

---

## 11. Where each part lives

| Concern | Location |
|---|---|
| Stationarity battery | `backend/core/stationarity.py` |
| Return transforms | `backend/core/transforms.py` |
| Orthogonalisation | `backend/core/orthogonalize.py` |
| Rolling regression | `backend/core/regression.py` |
| Covariance | `backend/core/covariance.py` |
| Forecast validation | `backend/core/risk.py` |
| Density estimation | `backend/core/distribution.py` |
| Factor definitions | `backend/pipeline/factor_defs.py` |
| Schema | `sql/` (14 migrations) |
| API | `backend/app/` |
| Interface | `frontend/` |

`backend/core` holds pure functions over arrays with no database access. That
boundary is what makes the accuracy claims checkable, and it is the main
architectural decision in the project.
