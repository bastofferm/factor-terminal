# Return-Based Multi-Asset Factor Model

**As built.** This document describes the model that is running, in the form it
runs. Every number in it was measured on the panel below rather than assumed, and
where a design decision went one way rather than another the reason is given.

A typeset version of this material — with a flowchart of how the terminal works,
the statistical methodology stated formally as an appendix, and the factor registry
in full — is [factor-model-paper.pdf](factor-model-paper.pdf); rebuild it with
`python -m scripts.build_paper`.
[amzn-case-study.pdf](amzn-case-study.pdf) works a single security end to end on
both factor panels, with every chart and every quoted number generated from the
database by `python -m scripts.build_case_study --pdf`.

Panel as of 2026-09-11 · 40 factors in 9 blocks · 203 instruments · 53 level
series · 200,072 factor observations from 2003-12-02 · 340 tests.

---

## 1. What the model is

A factor model that uses no holdings and no look-through. Every exposure is
estimated from observed returns. The question it answers is not *what does this
fund own* but *what does its return actually move with* — which is the only
question answerable for a fund that reports holdings quarterly, or for a strategy
whose positions change faster than its disclosures.

The consequence is a standing obligation: a return-based model measures realised
sensitivities, so it must be defended against spurious correlation, stale pricing,
non-stationarity and short histories. Most of the engineering below is that defence.

```
r_i,t − r_f,t  =  α_i  +  β_i′ F_t  +  ε_i,t
Var(r_p)       =  β_p′ Σ_F β_p  +  w′ Σ_ε w
```

All factors are daily log excess returns over cash (`FRED:DFF`) in USD.

---

## 2. Input data

### 2.1 What each block needs, and what the data supported

The model spans nine blocks of daily factor proxies. Each was audited against what
was actually obtainable before anything was built. The audit is the most
consequential part of the project, because a factor built on the wrong series is
worse than a missing factor: it looks fine and is wrong.

| Block | Verdict | Action |
|---|---|---|
| Rates | Excellent — complete US, EA and JP curves daily from FRED, ECB, BOJ, MOF | Used in full |
| Credit | Excellent — full ICE BofA OAS ladder (AAA→CCC) plus IG/HY/loan/EM ETFs | Used in full |
| Equity Style | Good — style ETFs plus Fama-French and AQR for validation | Used, with the caveat in §2.3 below |
| Volatility | Good — VIX futures ETFs, not index levels | Used in full |
| Commodities | Good — total-return ETFs | Used, **not** the futures series; see §2.2 below |
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
front-month prices with an artificial jump at every roll. A commodity return is
the spot move plus the roll plus the collateral yield, so the commodity block uses
total-return funds, which include all three.

**Low-frequency macro is never forward-filled.** Filling a weekly series into a
daily regressor manufactures information and understates standard errors: the
filled days carry no news, yet the regression counts them as independent
observations. Such series become sparse release-event factors: the
standardised change lands on the publication day and the factor is exactly zero in
between.

**The trading calendar excludes crypto.** Crypto prices seven days a week. Included,
it added 1,229 weekend rows on which every other factor is missing, which made
252-row estimation windows span fewer than 252 trading days and pushed equity
factors below the coverage threshold in the covariance matrix. This was found only
because the matrix endpoint started returning 8 factors instead of 39.

### 2.3 The weakest inputs, stated plainly

The style block uses long-only ETFs residualised against market and sector, not
true long-short portfolios. §4 quantifies how close that gets to the academic
factors. Quality and low-volatility are the weakest proxies in the set and should
not be leaned on.

---

## 3. Stationarity

Every model input must be a stationary return series, and this is treated as a
precondition rather than a diagnostic. A single ADF test is useless as a gate — on
daily returns it rejects almost always — so the battery is built around what
actually goes wrong with financial series.

| Test | What it catches | Gating |
|---|---|---|
| ADF × KPSS, read **jointly** | A level that should have been differenced | Blocks |
| Zivot-Andrews | A structural break masquerading as a unit root | Flags |
| Lo-MacKinlay variance ratio | Stale or smoothed pricing | Flags |
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

Current state: 30 pass, 10 warn, 0 fail on the trailing 252-day window.

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

### Every construction, written out

**[factor-formulas.md](factor-formulas.md)** gives each of the forty factors as a
formula over named instruments, plus the exact equation that orthogonalises it. It
is generated from the registry by `python -m scripts.render_formula_appendix`, and
the transcription is checked rather than trusted: `backend/tests/test_formula.py`
evaluates each rendered formula independently and asserts it reproduces the
builder's own output.

A sample of what that looks like — the 10s-2s steepener, which is the least obvious
of the forty:

$$
f_t = 5 \cdot \bigl(u^{(10y)}_t - u^{(2y)}_t\bigr), \qquad
u^{(T)}_t = \frac{b^{(T)}_t}{D^{(T)}_{t-1}}
$$

$$
b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t
+ \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2
+ \frac{y^{(T)}_{t-1} - c_{t-1}}{252}
$$

Yield changes at 2y and 10y become bond returns with duration and convexity taken
at *yesterday's* yield; each leg is divided by its own duration so the two are
comparable; the difference is scaled back to five years. Nothing in that is
recoverable from the string `curve`, which is the reason the appendix exists.

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
hierarchy of §4 removes the overlap, so each block's loading answers a specific
question.

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
regression, the covariance and the risk forecast consume by default — but it is not
what the factor earned. `eq_us` returns 12.9% a year at 17.3% volatility; its
residual after removing `eq_global` returns −0.2% at 4.2%. The application keeps
these on separate pages for that reason: the Factor Explorer describes the model,
the Raw Explorer describes the data.

### 5.1 Both panels, end to end

Orthogonalisation is a modelling choice, not a fact about the world, so the model
estimates on either panel rather than assuming one. Every factor is stored twice —
`ret_excess` for the raw series, `ret_orth` for the residual — and the choice
travels with the spec:

```bash
python -m backend.pipeline.run_estimation --instrument US:AAPL --window 252 --step 1m
python -m backend.pipeline.run_estimation --instrument US:AAPL --window 252 --step 1m --raw-factors
```

Those hash to different `spec_id`s, so the two live side by side in the cache rather
than overwriting each other. The Loadings Lab exposes the same switch, and so does
Covariance & PCA.

**The constraint that makes this more than a display option.** Portfolio risk is
$\beta'\Sigma\beta$, so $\Sigma$ has to be the covariance of the same series the
$\beta$s refer to. `run_risk` therefore reads the spec's own `orthogonalized` flag
and builds $\Sigma$ from the matching panel. It did not always: until this was
fixed, a spec estimated on raw factors was scored against the orthogonalised
covariance, and every predicted volatility it produced was wrong — silently, with
no error and no implausible number to give it away. The kind of bug this project
exists to make visible.

**What each panel buys.**

| | Raw | Orthogonalised |
|---|---|---|
| Reading a beta | total exposure — 1.02 to global equity means what it says | incremental — what the factor adds beyond the blocks above it |
| Design matrix | blocks overlap by construction; high VIFs by definition | well-conditioned; that is what the hierarchy is for |
| Attribution | double-counts: HY credit carries duration *and* equity | each block's contribution is its own |
| Best used for | explaining an exposure to someone | decomposing risk, and forecasting it |

### 5.2 What the raw panel actually costs

The argument for the hierarchy has until now been made from in-sample statistics —
correlations, condition numbers, residual correlations. Being able to estimate both
panels end to end turns it into an out-of-sample measurement. Identical settings
(252-day window, monthly step, OLS, blend covariance, 21-day horizon) on US:AAPL:
249 monthly windows each way, and 5,208 daily observations behind the VaR coverage
test.

| | Orthogonalised | Raw |
|---|---|---|
| Windows excluded as ill-conditioned | **0** of 249 | **16** of 249 |
| Mean max VIF (worst) | 26 (90) | 822 (69,720) |
| Mean condition number (worst) | 56 (134) | 142 (1,473) |
| Mean adjusted R² in sample | 0.442 | 0.481 |
| Bias ratio (target 1) | **0.964** | 0.884 |
| Mincer-Zarnowitz slope (target 1) | **1.023** (p = 0.95) | **0.350** (p = 0.0003) |
| Mincer-Zarnowitz R² | 0.134 | 0.055 |
| VaR 95% breaches vs 260 expected | **248** (Kupiec p = 0.43) | 201 (Kupiec p = 0.0001) |

The raw panel fits *better* in sample — adjusted R² 0.481 against 0.442 — and
forecasts distinctly worse out of sample. That is the entire case for the
orthogonalisation in two lines. A Mincer-Zarnowitz slope of 0.35 says the raw-panel
forecast barely tracks the variation in realised risk: it produces roughly the right
average level (bias 0.88) while getting individual periods wrong, which is exactly
what unstable betas on a collinear design do. Its VaR is systematically too wide —
201 breaches where 260 were expected — for the same reason.

Both panels fail Christoffersen (p < 0.0001): breaches cluster. That is a property
of applying a 21-day-horizon forecast to daily returns and is the same on both, so
it is not evidence about the panel choice.

The covariance matrix tells the same story from the other end. Same 39 factors, same
sample (2015-01 to 2026-05, 2,794 days), same blend estimator:

| | Orthogonalised | Raw |
|---|---|---|
| Condition number | 1,068 | **9,254** |
| Smallest eigenvalue | 2.19e-4 | 7.50e-5 |
| PC1 share of variance | 34.5% | **50.3%** |

Half the variance of the raw factor set lies in one direction, which is another way
of saying that a forty-factor model built on it is not really a forty-factor model.
Covariance & PCA draws both, because the raw heatmap is the clearest single picture
of what the hierarchy is for.

### 5.3 A gate that had to be made panel-aware

Ill-conditioned windows are excluded from scoring (§7). That gate originally tested
two things — a condition number above 200, or a maximum VIF above 100 — and the VIF
half does not survive contact with the raw panel: `eq_us` regressed on the other
thirty-nine raw factors has an R² near 0.9998, because `eq_global` is one of them.
Applied unchanged, the rule excluded **236 of 249** raw-panel windows and left
thirteen forecasts to score, which is not an estimable model.

The question is whether those 236 windows deserved exclusion. They did not: their
forecasts averaged 0.34 predicted against 0.28 realised, with a maximum of 0.68 —
nowhere near the 3.6 that motivated the gate in the first place. The mean condition
number on them was 147, inside the limit. They were being rejected for a property of
the factor set rather than a defect in the window.

So the condition number now gates both panels unchanged — it is the invariant that
governs how far $\beta'\Sigma\beta$ can blow up — and the VIF limit applies only to
the orthogonalised panel, where a VIF above 100 is genuinely anomalous. VIF is still
recorded on every forecast either way, keeping the role it was introduced for:
saying *which* factor is responsible. This is the same treatment ARCH-LM gets in the
stationarity battery — recorded, never gated, where what it detects is a property of
the design rather than a defect in it.

The raw panel still loses 16 windows to the condition-number check, all of them in
2002-03 where few factors yet existed. Nothing about relaxing the VIF rule makes it
ungated: a window with a condition number of 610 is excluded on either panel.

### 5.4 Per factor, side by side

The Factor Explorer carries a raw-against-orthogonalised panel for whichever factor
is selected: both cumulative paths on one axis, the difference between them, and
what the residualisation achieved — the correlation with each target before and
after, and the average loading that produced it. Measured on the common sample:

| Factor | Variance removed | vs. its largest target, before → after |
|---|---|---|
| `eq_us` | 94.1% | `eq_global` 0.967 → −0.020 |
| `liq_risk_off` | 80.9% | `eq_global` −0.847 → −0.003 |
| `cr_em_local` | 38.2% | `fx_usd` −0.536 → −0.015 |
| `fx_carry` | 38.7% | `fx_jpy` −0.659 → **+0.164** |
| `arp_trend` | −1.9% | `eq_global` 0.125 → 0.036 |
| `cm_broad` | 0.0% | nothing above it; the panels are identical |

Two rows are worth dwelling on. `fx_carry` still correlates +0.16 with `fx_jpy`
after residualisation — the sign has flipped but the magnitude has not gone away,
which is what a three-legged basket dominated by one funding currency looks like
even after the currency is regressed out; the factor's own registry entry already
says to treat its loading sceptically, and this is the measurement behind it. And
`arp_trend` has *negative* variance removed: the residual is fractionally noisier
than the factor it came from. That is not a defect in the arithmetic but a property
of a causal fit — a loading estimated on the trailing two years and applied for the
next month adds variance rather than removing it when the true loading is near zero
and moving. Both are shown rather than smoothed over.

---

## 6. Exposure estimation

Rolling regression of an instrument's excess return on the factor panel. Every
control that changes the answer is a user choice rather than a hard-coded constant,
because each one is a modelling judgement and none has a default that is right for
every instrument:

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

A loading vector that jumps between windows is a drift alert: the exposure the
model reports is no longer the exposure it reported last month, and a risk number
built on it inherits that instability. Two measures are kept — the L1 change in the
whole vector, and its correlation with the previous window.

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
annualised: a single short or unusually quiet history should not produce a falsely
precise idiosyncratic risk, and the floor is the model's admission that it cannot
see every risk a security carries.

**The covariance page is computed on the orthogonalised factors and must be.** The
loadings are loadings on that set, and portfolio risk is β′Σβ, so Σ has to be the
covariance of the same series the βs refer to.

### Forecasts are scored, not asserted

For a forecast dated *t*, the loadings come from a window ending at or before *t*
and the covariance from returns up to *t*; the realised volatility it is compared
against covers *t+1 … t+h*. The lag is enforced structurally, not by convention.

Four tests, each answering a different question:

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
point. The gate is a condition number above 200 on either panel, plus a maximum VIF
above 100 on the orthogonalised panel only — see §5.3 for why that half cannot apply
to the raw one. It excludes about 3% of orthogonalised windows and 6% of raw ones,
almost all in 2002–03.

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

Three things a factor model of this kind is often expected to carry are absent, and
are absent deliberately rather than by oversight.

**Portfolio optimisation and performance attribution.** The covariance matrix is
optimiser-ready — positive semi-definite, with interpretable exposures — but no
optimiser sits on top of it, and no attribution layer decomposes a realised return
into factor contributions. Both are downstream consumers of the model rather than
parts of it, and building either before the risk forecast has been shown to be
right would be building on an unverified base.

**A mixed-frequency state-space layer.** Macro release surprises are implemented
only as the sparse release-event transform described in §2.2: the standardised
change lands on the publication day and the factor is zero in between. There is no
announcement-window lead-lag structure and no Kalman-filtered latent macro state
interpolating the releases onto a daily grid. Interpolation is exactly the
forward-fill this model refuses elsewhere, with a filter in front of it; doing it
properly means committing to a state equation the data here cannot identify.

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
| Schema | `sql/` (15 migrations) |
| API | `backend/app/` |
| Interface | `frontend/` (eight pages) |

`backend/core` holds pure functions over arrays with no database access. That
boundary is what makes the accuracy claims checkable, and it is the main
architectural decision in the project.
