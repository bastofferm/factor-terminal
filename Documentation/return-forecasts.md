# Return Forecasts: ten-year building blocks

**22 October 2026.** This note covers the terminal's first return-side feature, the
**Return Forecasts** page (`/returns`). It forecasts each of thirteen asset classes
over ten years, in the way published capital market assumptions are built:
income, growth, inflation and a valuation change, each a number with a source.
The forecast is an expected value with a band, not a point. Around it the page
shows:

- the historical average, kept as the rear-view mirror it is;
- the return vectors implied by reference portfolios;
- a perspectives matrix, scoring every portfolio under every vector;
- a track record on the only histories long enough to judge a decade.

The starting point was the Portfolio Selection GmbH white paper *Rückspiegel oder
Frontscheibe?* (19 October 2026). Every number below was measured on the live
database on 22 October 2026. Two commands reproduce them:

- `python -m backend.pipeline.return_forecast --no-persist` gives the forecast;
- `python -m backend.pipeline.return_track_record` gives the track record.

The risk model is unchanged. It still forecasts risk, not returns; this page sits
beside it and shares none of its covariance settings (see section 4).

---

## 1. What was taken from the paper, and what was corrected

**Taken.**
- *Return forward, risk from long history.*
- *The horizon is ten years.* The page shows no one-year forecast: over a year,
  valuation explains next to nothing of the following return.
- *Conservative and transparent.* Every block and every assumption is shown with
  its default and its source, and stored with each run (`return_forecast_run`).
- *An expected value with a band.* The band is a Monte Carlo over the assumptions
  and a decade of return noise. Arithmetic and geometric figures are kept apart.
- *The naive extrapolation stays as a reference.*
- *Two kinds of return vector.* "Conviction" vectors (the building blocks and
  hand-entered published CMAs) are weighted against "concern" vectors (the implied
  returns of the market, 1/N and minimum-variance portfolios), in two stages. The
  consequences of each are shown as a perspectives matrix (Broadie 1993).

**Corrected or flagged.**
- **The minimum-variance implied vector is constant.** Unconstrained, Σ·w_mv is
  proportional to a vector of ones, so λΣw_mv gives every asset the same implied
  return. Long-only, the held assets share one value and the excluded ones get a
  higher one. The paper's "depends on the covariance matrix" holds only through
  which assets are held. A test pins the property down
  (`test_long_only_min_variance_implies_equal_returns_on_the_held_assets`).
- **A ten-year forecast cannot be validated on the terminal's own data.** The funds
  start between 2000 and 2013, which is fewer than two independent decades. The
  measurement discipline of the risk page (a held-out span, a naive benchmark,
  Diebold-Mariano) is therefore applied to Shiller (US, 1871 on) and to the JST
  panel (16 countries, 1870 on); see section 5.
- **The strategic covariance must be long history, not the risk default.** See
  section 4. The factor-model version planned at first was measured and lost.
- **Building blocks are compound returns.** Mean-variance needs
  μ_arith = (1+g)·exp(σ²/2) − 1.
- **Arithmetic.** The paper's Invesco 2023 blocks sum to 6.3%, not the stated 6.4%.
  Brooks's decomposition of the 11.9% does add up.
- **Vendor material.** The paper is partly vendor material (the "Return Estimator"
  of Portfolio Generator GmbH). The ideas are used, the product names are not.

The weights between vectors are the user's, with documented defaults. The model
agent does not choose them, because a ten-year horizon leaves it no held-out loss
to judge by.

---

## 2. Data, and where it comes from

The factors database had no fundamentals. AI_Analyst's warehouse (`xbrl_sec`) was
searched first and cannot supply index-level valuation history:

- US company data starts around FY2010.
- Its stored market capitalisations are wrong historically (split-adjusted price
  times as-reported shares, so AAPL FY2014 comes out at $154B), and so are the
  P/E and yield metrics built on them.
- Japan starts in FY2012-2015, Europe and EM only in 2021, and there is no CAPE.

Its ACM term-premium series are used. Three external files are loaded by hand into
`data/external/`, which is kept out of git, by `python -m
backend.pipeline.ingest_external [shiller|barclays|jst|all]`:

| Source | Content | Rows | Licence |
|---|---|---|---|
| Shiller, `ie_data.xls` (shillerdata.com) | US monthly 1871-01 to 2026-10: price, dividend, earnings, CPI, GS10, real series, CAPE, bond returns | 1,870 months | published without a licence; the disclaimer puts the risk of use on the user |
| Barclays Indices, `cape.csv` (Historic CAPE Ratio by country) | CAPE monthly 1981-12 to 2026-09, 26 markets including Europe and Japan | 10,516 | Barclays IP; no use in any product without written consent. **Internal use only** |
| Jordà-Schularick-Taylor Macrohistory R6, `JSTdatasetR6.xlsx` | annual 1870-2020, 18 countries: equity total return, dividend yield, bonds, bills, CPI | 33,038 | CC BY-NC-SA 4.0, **non-commercial**; cite Jordà et al. (2019) |

`ref_data_licence` holds these terms, and the page prints them beside the numbers.
The assistant gets only the derived forecasts, never the licensed series. Barclays'
US CAPE sits on average 0.66 points above Shiller's over 1981-2026, with a
correlation of 0.983.

Sources considered and rejected:

| Source | Why it was rejected |
|---|---|
| Siblis Research | The free tier is snapshots; full history costs $349 to $768; use in "an application … an AI assistant" needs a further licence |
| Nikkei and JPX P/E statistics | Nikkei forbids reproduction beyond personal use; JPX publishes PDFs only, with a methodology break in 2018 |
| Research Affiliates and Cambria | Current snapshots only |

Already in the database:
- ETF prices, from which each fund's distribution yield is derived as the
  close-versus-adjusted-close gap;
- the US, euro-area AAA and JGB curves, T10YIE, DFF and the ECB/BOJ rates;
- the BAML IG OAS, from 2023 only.

---

## 3. The method, and its defaults

Thirteen classes, each represented by a fund: SPY, VGK, EWJ, EEM, VNQ, IEF, TLT,
AGG, BNDX, EMB, DBC, GLD, plus cash. VT and ACWI are mapped onto the regional
classes by a returns-based style regression (Sharpe 1992) on 36 months. On
22 October that gave VT: US 64%, Europe 17%, Japan 9%, EM 10%. Every fund preset
of the Risk Lens can therefore be scored.

| Block | Default | Source |
|---|---|---|
| Equity income | distribution yield, median of the trailing 12-month values over 3 years | the fund's close vs adjusted close |
| Net buybacks | 0 | Brightman (2012); Invesco uses +1.2% for the US |
| Real growth per share | 1.89% ± 0.58% | Shiller real EPS, decade-average ends, 1871-2026 |
| Inflation | 2.35% | T10YIE; abroad by relative PPP, because the forecast is in dollars |
| US valuation | CAPE 40.7 toward 16.3 (geometric mean since 1881), over 15 years | Shiller |
| Europe/Japan valuation | CAPE toward its median since 1982 (Europe 22.3 → 19.6, Japan 29.9 → 33.8) | Barclays |
| EM/REIT valuation | dividend yield toward its mean since the fund started | fund distributions |
| Treasuries | yield at the fund's maturity (IEF 8.5y, TLT 25y), unchanged at the end, plus roll-down | US curve |
| AGG | UST at 8y + 30% × IG OAS − 30% × 0.10% loss | curve, BAML |
| BNDX | 70% euro-area AAA + 30% JGB at 8.5y, hedged: + (DFF − ECB/BOJ rate) | curves |
| EMB | distribution yield − 1.0% default loss | the fund |
| Cash | DGS10 − ACM 10y term premium: the expected average short rate | FRED, NY Fed |
| Commodities, gold | cash + 0; inflation + 0 | assumption |

The band uses 20,000 paths with a fixed seed. Each path draws:
- the valuation norm's sample (US 1881/1950/1990; Europe and Japan 1982/1995/2005;
  EM and REITs since start or the last 10 years);
- a reversion period from 10 to 20 years;
- growth within its standard error;
- a shift in end-of-horizon yields, from the 3.0-point spread of ten-year GS10
  changes.

It then adds ten years of correlated return noise.

**The forecast on 22 October 2026** (compound annual, USD):

| Class | Income | Growth | Inflation | Valuation | Other | Expected | 5% | 95% | Fund history |
|---|---|---|---|---|---|---|---|---|---|
| US equities | 1.27% | 1.89% | 2.35% | −6.11% | | **−0.59%** | −7.6% | 9.5% | 8.3% |
| European equities | 3.29% | 1.89% | 2.35% | −0.87% | | **6.67%** | −3.1% | 17.8% | 6.2% |
| Japanese equities | 2.31% | 1.89% | 2.35% | +0.81% | | **7.36%** | −2.4% | 15.6% | 3.1% |
| EM equities | 2.59% | 1.89% | 2.35% | +2.19% | | **9.03%** | −2.6% | 21.0% | 9.8% |
| US REITs | 4.14% | 0 | 2.35% | −0.25% | | **6.24%** | −4.9% | 19.4% | 7.2% |
| UST 7-10y | 5.16% | | | 0 | +0.22% | **5.38%** | 1.8% | 9.2% | 3.4% |
| UST 20y+ | 5.58% | | | 0 | −0.05% | **5.53%** | −2.5% | 14.3% | 3.4% |
| US aggregate | 5.39% | | | 0 | +0.18% | **5.57%** | 3.1% | 8.1% | 2.9% |
| Intl bonds, hedged | 5.03% | | | 0 | +0.62% | **5.65%** | 3.2% | 8.2% | 2.2% |
| EM sovereigns | 5.23% | | | 0 | −1.00% | **4.23%** | −2.0% | 10.9% | 4.4% |
| Commodities | 4.13% | | | | | **4.13%** | −5.7% | 15.0% | 2.7% |
| Gold | | | 2.35% | | | **2.35%** | −6.3% | 12.0% | 10.2% |
| Cash | 4.13% | | | | | **4.13%** | | | |

The US figure is the paper's "pure CAPE regression" reading: a full reversion to the
1881 mean. With the 1950 norm the US block rises by about 1.1 points, and with the
1990 norm by about 3.2. The band draws among all three, which is why its median
(+0.7%) sits above the central value. Read it as what a reversion to the long-run
mean would cost, not as a point forecast (section 5 shows this assumption was too
pessimistic after 1983).

---

## 4. The strategic covariance: measured, and changed from the plan

The plan was the risk model's own covariance, Σ = BΣ_FB′ + D, using the latest
loadings and the whole daily factor history since 2003 plus a 63-day lead-lag term.
Measured against the funds' own monthly returns, it overstated decade-horizon
volatility:

| Fund | Factor model, daily | Monthly sample | Quarterly sample |
|---|---|---|---|
| SPY | 20.2% | 15.2% | 16.6% |
| EWJ | 22.4% | 16.2% | 17.1% |
| EMB | 24.7% | 11.6% | 10.2% |
| AGG | 5.8% | 4.4% | 4.6% |
| IEF | 6.7% | 6.6% | 7.1% |

Daily returns carry crisis noise that partly reverses within the month. The
strategic matrix is therefore the equal-weighted pairwise sample of monthly log
total returns since 2000, crises included, repaired to positive semi-definite;
BNDX contributes 160 months, SPY 321. The quarterly figures confirm that monthly
sampling is not driving the result. The factor-model figure stays on the page as
a check column.

Neither covariance should be tuned toward the other. The risk page's fast blend
(0.8 × hl5) answers "how risky is this tomorrow"; this matrix answers "how do these
move together over a decade".

---

## 5. Track record

**US, Shiller, real ten-year returns.** At every month from 1901 the forecast is
rebuilt from data known at the time:
- the dividend yield;
- a CAPE with earnings lagged three months;
- expanding-mean real growth;
- an expanding-mean CAPE norm.

Its benchmark is the expanding historical mean.

| Sample | n | Corr. | Mean error | RMSE | RMSE of mean | OOS R² | DM t | p |
|---|---|---|---|---|---|---|---|---|
| Monthly from 1901 | 1,387 | 0.39 | −1.5% | 5.8% | 6.0% | **+0.06** | −0.27 | 0.79 |
| Monthly from 1950 | 802 | 0.52 | −3.8% | 6.0% | 5.7% | −0.13 | 0.32 | 0.75 |
| Monthly from 1983 | 406 | 0.77 | −6.7% | 7.4% | 5.1% | −1.11 | 1.80 | 0.07 |
| Non-overlapping decades | 11 | 0.41 | −1.2% | 6.5% | 7.1% | +0.16 | −0.51 | 0.61 |
| Rear-view mirror instead | 1,390 | −0.37 | −0.2% | 9.0% | 6.0% | −1.30 | **4.04** | **<0.001** |

What the table shows:
- **The paper's predictor correlations reproduce.** The CAPE yield against the next
  decade's return correlates 0.50; Ilmanen (2025) reports +0.52. The last decade's
  return against the next correlates −0.37, the same as Ilmanen's −0.37. The
  rear-view mirror is significantly *worse* than the historical mean.
- **The building blocks beat the mean only slightly, and not significantly.** That
  is Goyal and Welch in long-horizon form, and about a dozen independent decades
  cannot settle it.
- **Since 1983 the forecast tracks the realised path closely but sits 6.7 points
  too low.** The correlation is 0.77, but valuations stayed above their expanding
  norm for forty years. This is the paper's warning that a CAPE norm implicitly
  assumes rates return to the level that shaped it.

**Bonds.**
- Shiller's GS10 against the following decade's bond return, 1871-2016:
  correlation 0.95, mean error −0.11%. On non-overlapping decades the correlation
  is 0.95 with a mean error of −0.06%.
- IEF: correlation 0.89, mean error −0.52%. TLT: correlation 0.88, mean error
  −0.67%. Both cover 1.4 decades and are descriptive only.
- The starting yield is the forecast.

**Sixteen countries, JST, the dividend-yield form.** Non-overlapping decades
1890-2010. Pooled losses are averaged across countries within each decade before
the t-test.

| | n | OOS R² | t | Mean error |
|---|---|---|---|---|
| Pooled ex US, with yield reversion | 165 | **−0.19** | 0.96 | −2.2% |
| Pooled ex US, without valuation | 165 | +0.07 | −0.92 | −1.2% |

Australia (+0.63), Spain (+0.32) and France (+0.24) are positive. Denmark (−3.35),
Japan (−0.87) and Germany (−0.61) are negative. Japan's mean error is −9.5% a
year: the post-war decades beat a yield-based forecast by far. **On this evidence
the dividend-yield reversion used for EM and REITs does worse than no valuation
view**, though neither difference is significant. `dy_valuation_weight` (default
1, as chosen) sets how much of it counts. 0 is Brightman's form.

**Country CAPE, Barclays 1982 on, against JST's following real returns.**
Descriptive only, at 1.2 to 3 independent decades per market. The CAPE yield
correlates positively everywhere:

| Market | Corr. |
|---|---|
| France | 0.92 |
| Netherlands | 0.75 |
| USA | 0.74 |
| UK | 0.72 |
| Germany | 0.71 |
| Sweden | 0.67 |
| Japan | 0.64 |
| Switzerland | 0.55 |
| Italy | 0.42 |
| Spain | 0.42 |

The CAPE form has support where the dividend-yield form does not.

---

## Still open

- **Dividend-yield valuation for EM and REITs.** The JST panel favours dropping it
  (section 5). The decision is left to the user; the default stays as chosen.
- **The US CAPE norm.** The expanding-mean norm was too pessimistic for forty
  years. A norm tied to the cycle-neutral rate, as J.P. Morgan does, would answer
  the paper's criticism. It is not built.
- **Payout consistency.** A low US dividend yield (1.3%, after decades of buybacks)
  combined with growth measured in an era of high payouts understates per-share
  growth. Net buybacks of 0 make this worse. It is a known tension in Brightman's
  form, and it is shown, not fixed.
- **A EUR view.** The forecast is in USD only. An FX block (hedged by the rate
  differential, unhedged by a long-run assumption) is not built.
- **EM CAPE.** Barclays' EM countries start between 2001 and 2011, too short for a
  norm. A composite from country CAPEs is a later step.
- **BNDX's mix.** It covers euro-area AAA and JGB only. The UK, Canada and
  Australia are in the index but not in the database's curves.
- **AI_Analyst's market capitalisations.** Rebuilding them would allow a US
  bottom-up check of payout and buyback yields from FY2010. That work belongs in
  that repository.
- **Shared-database hazard.** `seed_level_series` deactivates every series it does
  not define. Running it from a branch without `seed.RETURN_INPUT_LEVELS`
  switches the ACM series off.

---

## Where each part lives

| Concern | File |
|---|---|
| Pure arithmetic: blocks, reverse optimisation, frontier, style analysis, band, OOS R² | `backend/core/returns.py` |
| Classes, assumptions, inputs, strategic covariance, the run | `backend/pipeline/return_forecast.py` |
| Shiller, JST and Barclays backtests | `backend/pipeline/return_track_record.py` |
| Loading the external files | `backend/pipeline/ingest_external.py` |
| Tables and licence rows | `sql/019_return_forecasts.sql` |
| ACM series registration | `backend/pipeline/seed.py` (`RETURN_INPUT_LEVELS`) |
| API | `backend/app/routers/returns.py` |
| Page and views | `frontend/src/app/returns/page.tsx`, `frontend/src/components/returns/` |
| Definitions behind every "?" | `frontend/src/lib/returnMetrics.ts` |
| Tests | `backend/tests/test_returns.py`, `test_return_forecast.py`, `test_ingest_external.py` |
