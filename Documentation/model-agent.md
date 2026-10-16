# The Model Agent

**What it replaces.** The Loadings Lab used to hand the analyst nine controls -
factor panel, window, roll-forward step, estimator, ridge penalty, weighting,
Dimson lags, HAC lags, winsorisation - and leave the question of which combination
was right to them. The question has an answer that can be measured, so the agent
measures it. It searches the specification space for one security, takes the
simplest model the data cannot distinguish from the best to a span of history the
search never saw, and gives a verdict.

The search is driven by a language model; every number is computed in code. The
language model proposes which specifications to try next and when to stop. It does
not fit anything, score anything, choose the winner or write the verdict.

```
profile -> seed -> evaluate -> select -> propose --continue--> evaluate
                                                 \--stop-----> finalize
finalize -> horizon -> final_test -> verdict -> persist
```

It is a [LangGraph](https://github.com/langchain-ai/langgraph) state machine
(`backend/agent/graph.py`). Run it from the Loadings Lab, or:

```bash
python -m backend.agent.run --instrument US:AAPL            # LLM drives, results stored
python -m backend.agent.run --instrument US:AAPL --no-llm   # rule-based critic drives
python -m backend.agent.run --instrument US:AAPL --dry-run  # store nothing
python -m backend.agent.run --instrument US:AAPL --no-pool  # ignore earlier runs
```

---

## 1. Why the protocol looks the way it does

A search that tries forty specifications and keeps the one with the lowest backtest
loss has not found the best model; it has found the luckiest. Its loss is biased
down by the search, more so the more it tries. Three things keep that bias out of
the verdict.

**A held-out test span.** The forecast dates are a monthly grid. The last 36 grid
dates - or 30% of the grid when that is shorter, never fewer than 24 - are the
test span. Everything before it, less an embargo, is the validation span. During
the search the evaluator is handed a copy of the data that *ends the day before
the test span begins* (`evaluate.truncated`), so no estimate, covariance,
benchmark or realised volatility can reach into it however the code above is
arranged. The embargo drops the validation dates whose 21-day outcome window would
overlap the test span. The test span is opened once, for the chosen model, after
the search has ended. `test_the_test_span_cannot_change_the_search` runs the whole
agent twice, once with every test-span return multiplied by fifty, and asserts the
ledger and the choice are identical.

**A model confidence set.** Instead of naming the candidate with the lowest loss,
the Hansen-Lunde-Nason procedure (`core/selection.py`) returns every candidate that
cannot be told apart from the best at 90% confidence, using a moving-block
bootstrap of the loss series. Anything outside it is significantly worse.

**A calibration screen.** Within the set, only candidates whose validation
forecasts were calibrated - the bias statistic within 1 ± √(2/n), Kupiec not
rejected at 95% or 99% (`selection.calibrated`) - may be chosen, when any are. On a
single stock the daily loss is dominated by specific-risk noise and cannot separate
a model that systematically under-forecasts from one that does not; the confidence
sets hold twenty-odd members. Without the screen the next rule chose heavily
shrunk ridge fits, whose effective parameter count is lowest because their betas
are shrunk - and shrunk betas understate β′Σβ. In three October 2026 runs every
ridge λ = 1 member of the set breached its 99% VaR 1.2 to 2.5 times as often as
expected on validation, and the one chosen for AAPL failed its test span for it.
Every ledger entry carries its `calibrated` flag, and the trace says how many
members passed.

**Parsimony.** Among the eligible, the candidate with the fewest effective
parameters wins - the design width for least squares and Huber, the trace of the
hat matrix for ridge - with ties broken on loss. Among models the data cannot
rank, the simplest is the one least likely to have fitted noise.

The stock's own sector factor is not charged as a parameter
(`evaluate.parsimony_complexity`). Its value is in what two stocks of one sector
share, which one stock's loss cannot see: on AAPL every candidate with it
explained about 40% of the next month's daily returns out of sample against 32%
without, at the same selection loss, and over 54 large caps it is what makes
sector, pair and cross-sector portfolios calibrated. Charged a parameter it would
lose every tie, and the stored model would book its sector's co-movement as
diversifiable; free, it ties with the same model without it and the loss decides.

### What the search optimises

A run optimises for one of three objectives (`--objective`, or *Optimise for* in
the Loadings Lab):

| Objective | The confidence set is computed on | Favours |
|---|---|---|
| next day (`daily`, the default) | the daily loss of the forecast in force | speed: a short covariance half-life that follows volatility day by day |
| horizon (`horizon`) | QLIKE of the h-day forecast against the h-day return, on the monthly grid | a forecast whose h-day risk is right, which a noisy covariance gets wrong more often |
| both (`both`) | each of the two; a candidate must survive both | a model that is not significantly worse on either; if none is, the daily set is used and the trace says so |

The calibration screen and the simplicity rule follow in every case; ties are
broken on the h-day loss whenever it is part of the objective. The two objectives
pull in different directions for a reason the portfolio validation measured: a
21-day covariance half-life wins day by day, a 42-day one calibrates 21-day risk
and the 99% tail better. The horizon loss has about one observation a month rather
than one a day, so its confidence set is wider and separates less. Both losses are
recorded for every candidate whatever the objective (`common_loss`,
`common_horizon_loss`).

### The loss the search is judged on

The confidence set is computed on the **daily QLIKE of the forecast in force**:
for every day, the forecast the model had issued by the day before, scored against
that day's squared return (the term that depends only on the return is dropped,
so a zero-return day is not infinite). The first version used monthly QLIKE on the
grid, and two things were wrong with it. With about 200 observations the
confidence set could separate nothing: 19 of 20 candidates survived on AAPL and
parsimony alone chose. And a monthly grid cannot see how often a forecast is
refreshed, which the methodology review found to be the largest single defect.
Daily, there are twenty times the observations, and on AAPL the first round's
confidence set excluded every forecast held for a month.

QLIKE is used because it ranks variance forecasts consistently when the realised
variance is a noisy proxy (Patton, 2011), and because it is not dominated by the
few crisis months whose variance dwarfs everything else, as squared error is.

---

## 2. The search space

Every candidate is two halves, and every field takes values from a fixed list
(`backend/agent/space.py`). A reply from the language model is parsed into these
types; anything that does not parse is rejected with the reason and quoted back to
it, never coerced.

| Estimation | Values |
|---|---|
| panel | orthogonalised, raw |
| factor set | all, the asset-class template, or an explicit list of factor ids |
| window | 126, 252, 378, 504, 756 days |
| weighting | equal, or EWMA with half-life 42, 63, 126, 252 |
| estimator | OLS, Huber, ridge with λ in {0.01, 0.03, 0.1, 0.3, 1} |
| Dimson lags | 0, 1, 2 - on equity and FX only, or on all factors |
| HAC lags | automatic, or 0-20 (changes standard errors, never a forecast) |
| winsorisation | on, off |
| sector | none, or the stock's own GICS sector factor |

| Risk | Values |
|---|---|
| covariance | sample, EWMA, Ledoit-Wolf, blend; half-life 5, 10, 21, 42, 60, 126; EWMA weight 0.4-1.0; window 252, 504, 756 |
| specific risk | the window's in-sample residual, or an EWMA of out-of-sample residuals (half-life 21, 63, 126) |
| noise correction | subtract tr(ΣV) from β′Σβ, or not |
| refresh | at each loading window, or daily with the betas held |
| VaR | normal, or Student t with degrees of freedom fitted to the trailing returns standardised by their own EWMA volatility |
| lead-lag | 0, 1, 2, 3 or 5 lags of cross-day factor covariance in the h-day forecast - chosen after the search, not by it (§2.2) |
| h-day reversion | none, or a variance shock decaying with a half-life of 126, 63 or 21 days - chosen after the lead-lag count (§2.3) |

The step is fixed at 21 days: every candidate is scored on the same grid.

The original model is a point in this space - orthogonalised, all factors, 252-day
OLS, the 60/40 blended covariance held for a month - and is always the first
candidate.

### 2.1 Sector factors

Eleven factors - one SPDR sector ETF per GICS sector, residualised against the
market, the US market and the cyclical tilt - sit beside the forty with role
`sector`. 'all' and the templates never include them: eleven more regressors on
every window would cost more in beta noise than they explain. A candidate with
`sector: own` adds the one factor of the stock's own GICS sector
(`factor_defs.SECTOR_FACTORS`), and only a security whose profile names a
`sector_factor` may ask for it; the language model is told so and a request
without one is rejected. For a security closing before New York the sector
factors are in the Dimson `equity_fx` set, being US-hours ETFs.

What the sector factor does to a single stock's *total* risk is small: what it
explains moves from specific to systematic risk. Its point is the off-diagonal
structure - what two stocks in one sector share - which no single-security test
can see. That is what the portfolio validation (`backend/pipeline/run_portfolio.py`,
see the [methodology review](methodology-review.md#portfolio-level-validation))
is for, and where the evidence for it is.

### 2.2 The h-day forecast and the lead-lag stage

Factors that close at different times - the euro and Japanese curves, a London
ETF, Yahoo's FX fixes, the US sector ETFs against a Tokyo stock - put one day's
common shock into two days' returns. A daily covariance cannot see that, and
should not: a day's return does not contain tomorrow's catch-up. The variance of
an h-day return does:

    Var(x_1 + ... + x_h) = h [Gamma_0 + sum_{l=1}^{h-1} (1 - l/h)(Gamma_l + Gamma_l')]

Every forecast now carries two volatilities. `sigma_pred_ann` is a day's, and the
VaR, the daily loss and the selection are read from it. `sigma_horizon_ann` is the
h-day return's, per day: the daily factor covariance plus the lead-lag term
truncated at `nw_lags` lags (`covariance.lead_lag_term`, weighted as the
covariance method weights its own sample, repaired to positive semi-definite),
and for a Dimson model the collapsed beta, whose cross-lag terms belong to a
multi-day return. The bias statistic - a statement about h-day returns - is read
against it, and a new loss scores it against the squared h-day return itself,
with a Diebold-Mariano test against own EWMA.

The lag count cannot be chosen by the search: it changes no daily forecast, so
every variant has the same selection loss. A stage after the search,
`graph.horizon_node`, scores the chosen model with 0, 1, 2, 3 and 5 lags on the
validation span's h-day returns and keeps the fewest whose loss is within one
standard error of the best, compared through the paired difference so the noise
the counts share cancels. It is still the validation span: the test span is
opened after it.

The evidence available to it is thin. A validation span holds 60 to 150
non-overlapping 21-day returns; on simulated data with a late factor that makes
the true h-day volatility 20% higher than the daily one, 60 of them gave bias
statistics from 0.91 to 1.26 across seeds. The one-standard-error rule keeps the
count at zero unless the h-day evidence is clear, which is the right default for
an adjustment that only an asynchronous security needs.

### 2.3 Mean reversion of the h-day forecast

A forecast scaled by √h assumes today's volatility lasts the whole horizon. It
rarely does: after a shock it falls, after a calm month it rises. If a variance
shock decays as φ^j, φ = 0.5^(1/half-life), the h-day average of the expected
daily variances is

    LR + g (spot - LR),   g = (1 - φ^h) / (h (1 - φ))      (covariance.reversion_weight)

one day ahead g = 1, so the daily forecast is untouched. In a candidate with
`revert_halflife`, the h-day factor covariance moves toward the Ledoit-Wolf
matrix of the same trailing window and an out-of-sample specific variance toward
the window's in-sample one, before the lead-lag term is added. Like the lag count,
it changes no daily forecast, so the search cannot rank it and the proposer may
not ask for it. `graph.horizon_node` chooses it after the lags, on the validation
span's h-day loss, by the same one-standard-error rule, weakest first: none, then
126, 63 and 21 days.

The 5- and 10-day covariance half-lives come from the same investigation (October
2026). On fund portfolios a 42-day half-life took twelve trading days to double
in February 2020 and four months to come halfway back. A blend that is 80% a
5-day EWMA and 20% the long-run matrix reacts within days without collapsing in a
quiet week, and the critic proposes it when a daily forecast with a 21-day or
longer half-life still loses to the own EWMA.

### Three corrections the risk half needed

These are in `backend/pipeline/run_risk.py` and available to hand-run forecasts too.

**Beta noise.** With β̂ estimated with covariance V, E[β̂′Σβ̂] = β′Σβ + tr(ΣV).
The second term is sampling error in the loadings, not risk the security carries;
with forty factors on 252 days it is about k/n = 16% of the specific variance, and
measured on five single stocks it was 9-16% of their systematic variance. The fit
now keeps the full HAC covariance of the slopes (with the ridge penalty in the
bread for ridge fits), and `debias` subtracts the term.

**Out-of-sample specific risk.** The one-day-ahead residual of the betas in force
is ε + f′(β − β̂), so E[e²] = σ²ε + f′Vf. Each squared residual is corrected by
f′Vf before it is averaged; without that, specific risk counts the beta noise a
second time.

**Dimson and the daily variance.** A Dimson model's collapsed beta, Σₗbₗ, is right
for the variance of a multi-day return. The variance of a *daily* return is
Σₗ bₗ′Σbₗ, without the cross-lag terms, and predicting a day's return needs each
lag separately. The realised volatility the forecasts are scored against is a
standard deviation of daily returns, so a Dimson candidate is forecast and scored
lag by lag.

---

## 3. Who proposes

**Seed.** Deterministic: the original model; the raw panel; ridge; a longer,
decayed window; a fast, daily-refreshed, debiased forecast with t-VaR; the
asset-class template, debiased; for a security that closes before the US
factors, one Dimson lag on equity and FX; and for a stock with a sector factor,
the template with it.

**The pool.** Then what earlier runs found (`store.load_pool`): the security's own
earlier choices, the choices made for securities of the same template, and the
rest of its earlier confidence sets by loss - at most eight, re-scored here like
any candidate, under origin `pool`. On a single stock the confidence set is wide,
so which near-equals happen to be proposed decides the winner; two AAPL runs
chose different members. Pooling puts the earlier winners in front of the
confidence set every time, so a later run either confirms them or beats them on
its own validation span. Only specifications cross over - never a loss, never a
test-span number - and a pooled candidate that does not fit the security (a
sector factor it does not have) is dropped. `--no-pool` turns it off.

**The language model** (`backend/agent/llm.py`) then drives. Each round it sees the
security's profile, the split, the search space, every candidate so far with its
validation diagnostics, the confidence set, the leader, the critic's hints and the
remaining budget, and answers with JSON: continue or stop, a rationale, and at
most six candidates, each optionally written as changes to a ledger entry ("c04
with a daily refresh"). One invalid reply earns one repair round; a second
failure, a missing key or a network error hands the round to the critic, and the
substitution is in the trace. A stop is honoured only after one refinement round.

**The critic** (`backend/agent/critic.py`) reads the leader's diagnostics and states
what each points to, with the change that would test it: losing to the security's
own EWMA volatility means the forecast is too slow; 99% breaches far above the
count with 95% right is a tail problem; residual autocorrelation on an
early-closing security is the Dimson case; an in-sample fit far above the
out-of-sample one has too many parameters; and so on. Its lines are hints for the
language model, and the critic is the proposer whenever the model is unavailable.
When no diagnostic points anywhere, it proposes the leader's untried neighbours,
so a search never ends on the seed alone.

**Budget.** At most five refinement rounds, forty candidates and thirty minutes.

### What is sent to DeepSeek

The proposer uses the client the Ask panel already uses. It sends the instrument
id, its profile (asset class, history length, stationarity verdict, kurtosis), the
factor names and blocks, the search space, and the validation metrics of every
candidate. It sends no return series, no prices and nothing from the test span.

---

## 4. The verdict

Computed by `core/selection.verdict` on the test span, once.

| Gate | Kind | Rule |
|---|---|---|
| test sample | correctness | at least 24 test forecasts, or the verdict is *insufficient data* |
| forecasts issued | correctness | a finite forecast on at least 95% of test dates |
| design conditioning | correctness | at most 10% of test windows ill-conditioned |
| stationary returns | correctness | the security's returns do not fail the stationarity battery |
| beats the naive forecast | value | not significantly worse than the security's own EWMA volatility on monthly QLIKE (Diebold-Mariano, p < 0.05) |
| ... day by day | value | the same comparison on the daily loss; losing is a caveat |
| bias statistic | calibration | the standard deviation of h-day returns over their h-day forecast within 1 ± √(2/n) |
| Mincer-Zarnowitz | calibration | joint test of a = 0, b = 1 not rejected |
| VaR 95% / 99% count | calibration | Kupiec not rejected |
| VaR 95% clustering | calibration | Christoffersen not rejected |
| circularity | calibration | the security is not an input to a factor in the model |

A failed correctness or monthly value gate is a **fail**. No calibration flag is a
**pass**; one or two are a **pass with caveats**; more is a fail. Every gate carries
the number it was decided on.

Forecasts dated before the test span are stored with `in_selection_sample = true`:
they include the dates the model was chosen on and are not evidence for the choice.
The Risk Lens's full-history backtest is labelled accordingly and the held-out one
is shown beside it.

---

## 5. Where the results go

The chosen specification is stored through the pipeline's own writers - its spec,
loadings and beta covariances, its forecasts and both backtests (`scope` `full` and
`test`) - so the Loadings Lab, the Risk Lens and the covariance page read it like
any other. The risk assembly is stored under the covariance label `agent:<hash>`,
or the plain method name when it is the original model's. The run itself is in
`agent_run`, every candidate in `agent_candidate` and the trace in `agent_step`;
Operations lists runs as `model_agent`.

In the Loadings Lab every leaderboard row opens a card for that candidate: what
it does in plain terms, the effective number of observations behind its betas,
factor covariance and specific risk, every validation number, and where it stood
in the selection - replayed from the stored ledger by `agent/explain.py`, so c20
reads "not calibrated: 99% VaR breach count rejected by Kupiec (p = 0.042)"
rather than a row of numbers. Every column, tile and verdict gate carries its
definition, computation and, for a test, its null hypothesis
(`frontend/src/lib/agentMetrics.ts`).

---

## 6. Limits

- **One security at a time.** The agent chooses a model per security and the
  verdict is per security. Whether a model's risks aggregate is tested separately,
  on a fixed configuration over 54 large caps (`run_portfolio`); the agent's own
  choices, which differ by security, are not yet aggregated.
- **The test span is short.** 36 monthly forecasts is enough for the verdict's
  tests to have some power and not enough for any of them to be decisive alone.
- **The language model is not deterministic** even at temperature 0, so two runs
  can try different candidates. The selection rule and the verdict are; the trace
  records exactly what was proposed and why. On a single stock this matters: the
  confidence set is wide, so which near-equals the language model happened to
  propose decides the winner. Two AAPL runs chose different members and scored
  0.314 and 0.357 test QLIKE - both *pass with caveats*. The pool narrows this by
  carrying earlier winners into every run; see the
  [methodology review's results](methodology-review.md#results).
- **The lead-lag count rests on few h-day returns** (§2.2), so it is kept at zero
  unless the evidence is clear.
- **The confidence set is honest about what it cannot tell apart**, which on a
  single stock is often a good deal. Parsimony then decides, and a verdict of
  *indistinguishable from own EWMA* is common: a single stock's own history
  forecasts its total risk well, and the factor model's case then rests on the
  decomposition rather than the total.
