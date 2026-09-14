# Factor Formulas

Every factor in the model, written out: the arithmetic that builds it from named
instruments, and the equation that residualises it against the blocks above it.

**This file is generated.** `python -m scripts.render_formula_appendix` rewrites it
from `backend/pipeline/factor_defs.py` through `backend/pipeline/formula.py`, which
is transcribed from the builders in `backend/pipeline/build_factors.py`. The
transcription is not taken on trust: `backend/tests/test_formula.py` evaluates each
rendered formula independently and asserts it reproduces the builder's own output on
simulated inputs.

## Reading these

Each factor is stored twice, and the two are different series.

| | Column | What it is |
|---|---|---|
| **Raw** | `fact_factor_return.ret_excess` | $f_t$, the construction formula's own output |
| **Orthogonalised** | `fact_factor_return.ret_orth` | $\tilde{f}_t$, after the block hierarchy has residualised it |

The model can be estimated on either. The choice is recorded on the spec
(`dim_model_spec.orthogonalized`) and carried through to the covariance matrix,
because $\beta'\Sigma\beta$ requires $\Sigma$ to be the covariance of the same
series the $\beta$s refer to. A factor whose orthogonalisation list is empty has
$\tilde{f}_t = f_t$ identically, and estimating on one panel rather than the other
changes nothing about it.

## Notation

| Symbol | Meaning |
|---|---|
| $r^X_t$ | log total return of instrument $X$: $\ln(P_t / P_{t-1})$, on a price series already adjusted for dividends and splits |
| $c_t$ | daily cash rate: FRED:DFF at $t$, an annualised percent, divided by 100 and by 252 |
| $x^X_t$ | excess return over cash: $x^X_t = r^X_t - c_{t-1}$ |
| $\Delta$ | first difference: $\Delta s_t = s_t - s_{t-1}$ |
| $f_t$ | the raw factor |
| $\tilde{f}_t$ | the orthogonalised factor |
| $\tau$ | the most recent orthogonalisation refit at or before $t$ |

Every lag is deliberate. A quantity dated $t-1$ is there because using its value at
$t$ would put information into a return that was not available when the return was
earned.

---


## 1. Equity Market

### `eq_global` — Global Equity

Hierarchy level 0. Method `single`.

> MSCI ACWI total return less cash. The single global market factor.

**Construction**

$$
f_t = x^{\text{ACWI}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). ACWI is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^ACWI` | `ACWI` | instrument return: log total return of ACWI |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `eq_us` — US Equity (ex-global)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{IVV}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). IVV is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^IVV` | `IVV` | instrument return: log total return of IVV |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `eq_europe` — Europe Equity (ex-global)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{VGK}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). VGK is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^VGK` | `VGK` | instrument return: log total return of VGK |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `eq_japan` — Japan Equity (ex-global)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{EWJ}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). EWJ is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^EWJ` | `EWJ` | instrument return: log total return of EWJ |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `eq_em` — EM Equity (ex-global)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{EEM}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). EEM is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^EEM` | `EEM` | instrument return: log total return of EEM |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `eq_size` — Size (small minus large)

Hierarchy level 1. Method `spread`.

> Russell 2000 minus Russell 1000. Self-financing, so no cash leg.

**Construction**

$$
f_t = r^{\text{IWM}}_t - r^{\text{IWB}}_t
$$

1. Self-financing, so no cash leg: the funding cost cancels between the two legs and subtracting it would understate the spread.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^IWM` | `IWM` | long leg: log total return of IWM |
| `r_t^IWB` | `IWB` | short leg: log total return of IWB |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `eq_cyclical` — Cyclical minus Defensive

Hierarchy level 1. Method `basket`.

> Sector breadth without spending eleven factors on it.

**Construction**

$$
f_t = \frac{r^{\text{XLI}}_t + r^{\text{XLB}}_t + r^{\text{XLE}}_t + r^{\text{XLF}}_t + r^{\text{XLY}}_t}{5} - \frac{r^{\text{XLP}}_t + r^{\text{XLU}}_t + r^{\text{XLV}}_t}{3}
$$

1. Equally weighted long basket of 5: each leg carries 1/5 of the notional.
2. Minus an equally weighted short basket of 3. Both sides carry the market direction, so it cancels and what survives is the spread between them.
3. Self-financing, so no cash leg.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^XLI` | `XLI` | long leg: log total return of XLI |
| `r_t^XLB` | `XLB` | long leg: log total return of XLB |
| `r_t^XLE` | `XLE` | long leg: log total return of XLE |
| `r_t^XLF` | `XLF` | long leg: log total return of XLF |
| `r_t^XLY` | `XLY` | long leg: log total return of XLY |
| `r_t^XLP` | `XLP` | short leg: log total return of XLP |
| `r_t^XLU` | `XLU` | short leg: log total return of XLU |
| `r_t^XLV` | `XLV` | short leg: log total return of XLV |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 2. Equity Style

### `sty_value` — Value

Hierarchy level 2. Method `single`.

> Long-only ETF residualised to market and sector. An approximation to a true long-short HML; validated against F-F HML over the common sample.

**Construction**

$$
f_t = x^{\text{VLUE}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). VLUE is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^VLUE` | `VLUE` | instrument return: log total return of VLUE |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{eq\_us}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{eq\_cyclical}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `eq_global`, `eq_us`, `eq_cyclical`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `sty_momentum` — Momentum

Hierarchy level 2. Method `single`.

**Construction**

$$
f_t = x^{\text{MTUM}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). MTUM is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^MTUM` | `MTUM` | instrument return: log total return of MTUM |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{eq\_us}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{eq\_cyclical}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `eq_global`, `eq_us`, `eq_cyclical`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `sty_quality` — Quality

Hierarchy level 2. Method `single`.

**Construction**

$$
f_t = x^{\text{QUAL}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). QUAL is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^QUAL` | `QUAL` | instrument return: log total return of QUAL |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{eq\_us}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{eq\_cyclical}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `eq_global`, `eq_us`, `eq_cyclical`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `sty_lowvol` — Low Volatility

Hierarchy level 2. Method `single`.

**Construction**

$$
f_t = x^{\text{USMV}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). USMV is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^USMV` | `USMV` | instrument return: log total return of USMV |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{eq\_us}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{eq\_cyclical}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `eq_global`, `eq_us`, `eq_cyclical`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 3. Rates

### `rt_us_level` — US Rates Level

Hierarchy level 0. Method `curve`.

> Equal unit-duration long across the curve, scaled to 5y duration.

**Construction**

$$
f_t = 5 \cdot \frac{1}{4}\sum_{T} u^{(T)}_t, \qquad u^{(T)}_t = \frac{b^{(T)}_t}{D^{(T)}_{t-1}}
$$

$$
b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t + \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2 + \frac{y^{(T)}_{t-1} - c_{t-1}}{252}
$$

1. Convert each tenor's yield change into a bond return, with duration and convexity evaluated at yesterday's yield. Section 2.2 requires every factor to be a return, and a yield change is not one.
2. Divide each leg by its own modified duration. Five basis points on a 30y is not the same trade as five on a 2y, and without this the long end would dominate every shape.
3. Level: an equal unit-duration long at every tenor (2y, 5y, 10y, 30y).
4. Scale back by 5 years of duration.

| Symbol | Source | Meaning |
|---|---|---|
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |
| `FRED:DGS2` | `FRED:DGS2` | US_TSY par yield at 2y |
| `FRED:DGS5` | `FRED:DGS5` | US_TSY par yield at 5y |
| `FRED:DGS10` | `FRED:DGS10` | US_TSY par yield at 10y |
| `FRED:DGS30` | `FRED:DGS30` | US_TSY par yield at 30y |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `rt_us_slope` — US Rates Slope (10s-2s)

Hierarchy level 0. Method `curve`.

> Duration-neutral steepener: long 10y, short 2y, equal duration.

**Construction**

$$
f_t = 5 \cdot \bigl(u^{(10y)}_t - u^{(2y)}_t\bigr), \qquad u^{(T)}_t = \frac{b^{(T)}_t}{D^{(T)}_{t-1}}
$$

$$
b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t + \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2 + \frac{y^{(T)}_{t-1} - c_{t-1}}{252}
$$

1. Convert each tenor's yield change into a bond return, with duration and convexity evaluated at yesterday's yield. Section 2.2 requires every factor to be a return, and a yield change is not one.
2. Divide each leg by its own modified duration. Five basis points on a 30y is not the same trade as five on a 2y, and without this the long end would dominate every shape.
3. Slope: long 10y against short 2y in equal units of duration, so a parallel shift nets to zero and only the steepening survives.
4. Scale back by 5 years of duration.

| Symbol | Source | Meaning |
|---|---|---|
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |
| `FRED:DGS2` | `FRED:DGS2` | US_TSY par yield at 2y |
| `FRED:DGS10` | `FRED:DGS10` | US_TSY par yield at 10y |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `rt_us_curve` — US Rates Curvature (butterfly)

Hierarchy level 0. Method `curve`.

> Long the belly, short the wings, duration-neutral.

**Construction**

$$
f_t = 5 \cdot \bigl(2u^{(5y)}_t - u^{(2y)}_t - u^{(30y)}_t\bigr), \qquad u^{(T)}_t = \frac{b^{(T)}_t}{D^{(T)}_{t-1}}
$$

$$
b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t + \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2 + \frac{y^{(T)}_{t-1} - c_{t-1}}{252}
$$

1. Convert each tenor's yield change into a bond return, with duration and convexity evaluated at yesterday's yield. Section 2.2 requires every factor to be a return, and a yield change is not one.
2. Divide each leg by its own modified duration. Five basis points on a 30y is not the same trade as five on a 2y, and without this the long end would dominate every shape.
3. Curvature: long two units of the 5y belly against one unit each of the 2y and 30y wings. Neutral to both a parallel shift and a steepening.
4. Scale back by 5 years of duration.

| Symbol | Source | Meaning |
|---|---|---|
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |
| `FRED:DGS2` | `FRED:DGS2` | US_TSY par yield at 2y |
| `FRED:DGS5` | `FRED:DGS5` | US_TSY par yield at 5y |
| `FRED:DGS30` | `FRED:DGS30` | US_TSY par yield at 30y |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `rt_ea_level` — EA Rates Level

Hierarchy level 0. Method `curve`.

> ECB AAA curve. Warehouse-sourced, so it lags the US block.

**Construction**

$$
f_t = 5 \cdot \frac{1}{4}\sum_{T} u^{(T)}_t, \qquad u^{(T)}_t = \frac{b^{(T)}_t}{D^{(T)}_{t-1}}
$$

$$
b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t + \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2 + \frac{y^{(T)}_{t-1} - c_{t-1}}{252}
$$

1. Convert each tenor's yield change into a bond return, with duration and convexity evaluated at yesterday's yield. Section 2.2 requires every factor to be a return, and a yield change is not one.
2. Divide each leg by its own modified duration. Five basis points on a 30y is not the same trade as five on a 2y, and without this the long end would dominate every shape.
3. Level: an equal unit-duration long at every tenor (2y, 5y, 10y, 30y).
4. Scale back by 5 years of duration.

| Symbol | Source | Meaning |
|---|---|---|
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |
| `ECB:BUND_2Y` | `ECB:BUND_2Y` | EA_AAA par yield at 2y |
| `ECB:BUND_5Y` | `ECB:BUND_5Y` | EA_AAA par yield at 5y |
| `ECB:BUND_10Y` | `ECB:BUND_10Y` | EA_AAA par yield at 10y |
| `ECB:BUND_30Y` | `ECB:BUND_30Y` | EA_AAA par yield at 30y |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `rt_us_level`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `rt_jp_level` — JP Rates Level

Hierarchy level 0. Method `curve`.

> MOF/BOJ curve. Warehouse-sourced, so it lags the US block.

**Construction**

$$
f_t = 5 \cdot \frac{1}{4}\sum_{T} u^{(T)}_t, \qquad u^{(T)}_t = \frac{b^{(T)}_t}{D^{(T)}_{t-1}}
$$

$$
b^{(T)}_t = -D^{(T)}_{t-1}\,\Delta y^{(T)}_t + \tfrac{1}{2} C^{(T)}_{t-1}\,\bigl(\Delta y^{(T)}_t\bigr)^2 + \frac{y^{(T)}_{t-1} - c_{t-1}}{252}
$$

1. Convert each tenor's yield change into a bond return, with duration and convexity evaluated at yesterday's yield. Section 2.2 requires every factor to be a return, and a yield change is not one.
2. Divide each leg by its own modified duration. Five basis points on a 30y is not the same trade as five on a 2y, and without this the long end would dominate every shape.
3. Level: an equal unit-duration long at every tenor (2y, 5y, 10y, 30y).
4. Scale back by 5 years of duration.

| Symbol | Source | Meaning |
|---|---|---|
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |
| `MOF_JP:JGB_2Y` | `MOF_JP:JGB_2Y` | JP_JGB par yield at 2y |
| `MOF_JP:JGB_5Y` | `MOF_JP:JGB_5Y` | JP_JGB par yield at 5y |
| `MOF_JP:JGB_10Y` | `MOF_JP:JGB_10Y` | JP_JGB par yield at 10y |
| `MOF_JP:JGB_30Y` | `MOF_JP:JGB_30Y` | JP_JGB par yield at 30y |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `rt_us_level`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `rt_uk` — UK Gilts

Hierarchy level 0. Method `single`.

> No daily gilt curve exists anywhere in the warehouse, so this is an ETF total return converted to base currency rather than a curve factor.

**Construction**

$$
f_t = \bigl(x^{\text{IGLT.L}}_t + \Delta \ln e^{\text{GBP}}_t\bigr)
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). IGLT.L is a funded long position, so its return has to be measured over cash.
2. Convert to the base currency by adding the log change in GBP/USD. In logs the conversion is a sum, with no cross term.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^IGLT.L` | `IGLT.L` | instrument return: log total return of IGLT.L |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |
| `e_t^GBP` | `GBP` | USD per unit of GBP; its log change converts the local return into the base currency, which is exact in logs |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `rt_us_level`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `rt_breakeven` — Inflation Breakeven

Hierarchy level 0. Method `synthetic_bond`.

> 10y breakeven change, duration-scaled. Carry excluded: a breakeven is a spread between two yields and has no coupon of its own.

**Construction**

$$
f_t = -D_{t-1}\,\Delta y_t + \tfrac{1}{2} C_{t-1}\,(\Delta y_t)^2
$$

1. Treat the series as the yield of a 10y par bond and price a day's move on it, with duration and convexity taken at t-1 so nothing about today's move leaks into its own pricing.
2. No carry leg. A breakeven is the difference between two yields and earns no coupon of its own, so the builder adds the carry term and subtracts it again; the two cancel exactly and the factor is a pure duration-scaled change.

| Symbol | Source | Meaning |
|---|---|---|
| `FRED:T10YIE` | `FRED:T10YIE` | quoted yield series, in percent; y_t is that divided by 100 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `rt_us_level`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 4. Credit

### `cr_ig` — IG Credit

Hierarchy level 1. Method `single`.

> Rates-hedged by regression, not by analytic duration match.

**Construction**

$$
f_t = x^{\text{LQD}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). LQD is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^LQD` | `LQD` | instrument return: log total return of LQD |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{rt\_us\_slope}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, g_2$ = `rt_us_level`, `rt_us_slope`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cr_hy` — High Yield

Hierarchy level 1. Method `single`.

> Also residualised against equity: HY beta to equity is large and leaving it in would double-count equity risk.

**Construction**

$$
f_t = x^{\text{HYG}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). HYG is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^HYG` | `HYG` | instrument return: log total return of HYG |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{rt\_us\_slope}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `rt_us_level`, `rt_us_slope`, `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cr_loans` — Leveraged Loans

Hierarchy level 1. Method `single`.

> Floating rate, so little duration; the interesting part is what it adds over HY.

**Construction**

$$
f_t = x^{\text{BKLN}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). BKLN is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^BKLN` | `BKLN` | instrument return: log total return of BKLN |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{cr\_hy}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, g_2$ = `rt_us_level`, `cr_hy`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cr_em_hard` — EM Hard Currency

Hierarchy level 1. Method `single`.

> Residualised against equity as well: EM sovereign spreads carry real equity beta (0.68 before this was removed), which would otherwise be double-counted against the equity block.

**Construction**

$$
f_t = x^{\text{EMB}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). EMB is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^EMB` | `EMB` | instrument return: log total return of EMB |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{rt\_us\_slope}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{cr\_hy}}_t - \hat{b}_{4,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{4}$ = `rt_us_level`, `rt_us_slope`, `cr_hy`, `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cr_em_local` — EM Local Currency

Hierarchy level 2. Method `single`.

> Residualised against USD as well: the local-currency leg is mostly FX.

**Construction**

$$
f_t = x^{\text{EMLC}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). EMLC is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^EMLC` | `EMLC` | instrument return: log total return of EMLC |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{cr\_em\_hard}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{fx\_usd}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `rt_us_level`, `cr_em_hard`, `fx_usd`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 5. FX

### `fx_usd` — USD Broad

Hierarchy level 0. Method `single`.

> Dollar index return. Not a funded position, so no cash leg.

**Construction**

$$
f_t = r^{\text{DX-Y.NYB}}_t
$$

1. No cash leg: this is not a funded position, so there is nothing to fund.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^DX-Y.NYB` | `DX-Y.NYB` | instrument return: log total return of DX-Y.NYB |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `fx_jpy` — JPY (ex-USD)

Hierarchy level 1. Method `single`.

> Sign flipped so a positive value means a stronger yen, the usual risk-off direction.

**Construction**

$$
f_t = -1\,r^{\text{USDJPY=X}}_t
$$

1. No cash leg: this is not a funded position, so there is nothing to fund.
2. Multiply by -1 so the factor points in the stated direction.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^USDJPY=X` | `USDJPY=X` | instrument return: log total return of USDJPY=X |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{fx\_usd}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `fx_usd`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `fx_em` — EM FX

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{CEW}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). CEW is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^CEW` | `CEW` | instrument return: log total return of CEW |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{fx\_usd}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `fx_usd`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `fx_carry` — FX Carry (G4 approximation)

Hierarchy level 1. Method `fx_carry`.

> APPROXIMATION. Built from policy-rate differentials because the warehouse has no forward points, and limited to the three non-USD policy rates available. Narrower and noisier than a real G10 carry basket; treat its loading with corresponding scepticism. Also residualised against JPY: with only three legs and the yen as the perennial funding currency, the raw basket correlated -0.87 with fx_jpy, i.e. it was the yen short wearing a different name.

**Construction**

$$
\kappa_{j,t} = \frac{i_{j,t-1} - i_{\text{USD},t-1}}{100 \cdot 252}, \qquad p_{j,t} = \operatorname{sign}(\kappa_{j,t})
$$

$$
f_t = \frac{1}{3}\sum_j p_{j,t}\bigl(s_j\,r^{X_j}_t + \kappa_{j,t}\bigr)
$$

1. Carry is approximated from policy-rate differentials (CHF, EUR, JPY against USD) because the warehouse holds no forward points. Covered interest parity says the forward discount equals the rate differential, so this stands in for it.
2. The differential is lagged one day: the carry earned over day t is fixed by the rates set at t-1.
3. Each leg is held long when it yields more than the dollar and short when it yields less, and the basket is the equally weighted average of the legs.
4. APPROXIMATION. Three non-USD legs is narrower and noisier than a real G10 carry basket; treat its loading with corresponding scepticism.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^USDCHF=X` | `USDCHF=X` | CHF spot leg, sign flipped so the quoted pair reads as long the foreign currency against the dollar: log total return of USDCHF=X |
| `SNB:POLICY_RATE` | `SNB:POLICY_RATE` | CHF policy rate i_CHF, annualised percent |
| `r_t^EURUSD=X` | `EURUSD=X` | EUR spot leg: log total return of EURUSD=X |
| `ECB:DFR` | `ECB:DFR` | EUR policy rate i_EUR, annualised percent |
| `r_t^USDJPY=X` | `USDJPY=X` | JPY spot leg, sign flipped so the quoted pair reads as long the foreign currency against the dollar: log total return of USDJPY=X |
| `BOJ:IR01_OCRT` | `BOJ:IR01_OCRT` | JPY policy rate i_JPY, annualised percent |
| `FRED:DFF` | `FRED:DFF` | USD policy rate, the funding leg |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{fx\_usd}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{fx\_jpy}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, g_2$ = `fx_usd`, `fx_jpy`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 6. Commodities

### `cm_broad` — Broad Commodity

Hierarchy level 0. Method `single`.

> DBC is a total-return futures fund: roll and collateral included.

**Construction**

$$
f_t = x^{\text{DBC}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). DBC is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^DBC` | `DBC` | instrument return: log total return of DBC |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `cm_energy` — Energy (ex-broad)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{USO}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). USO is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^USO` | `USO` | instrument return: log total return of USO |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{cm\_broad}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `cm_broad`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cm_gold` — Gold (ex-broad)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{GLD}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). GLD is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^GLD` | `GLD` | instrument return: log total return of GLD |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{cm\_broad}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `cm_broad`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cm_industrial` — Industrial Metals (ex-broad)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{CPER}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). CPER is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^CPER` | `CPER` | instrument return: log total return of CPER |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{cm\_broad}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `cm_broad`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `cm_agri` — Agriculture (ex-broad)

Hierarchy level 1. Method `single`.

**Construction**

$$
f_t = x^{\text{DBA}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). DBA is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^DBA` | `DBA` | instrument return: log total return of DBA |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{cm\_broad}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `cm_broad`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 7. Volatility

### `vol_equity` — Equity Volatility (VIX futures)

Hierarchy level 1. Method `single`.

> Long VIX-futures strategy return, residualised against equity.

**Construction**

$$
f_t = x^{\text{VIXY}}_t
$$

1. Subtract the overnight rate: x_t = r_t - c_(t-1). VIXY is a funded long position, so its return has to be measured over cash.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^VIXY` | `VIXY` | instrument return: log total return of VIXY |
| `c_t` | `FRED:DFF` | daily cash rate: FRED:DFF_t / 100 / 252, lagged one day because the overnight rate earned on day t is set at t-1 |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `vol_variance_premium` — Variance Risk Premium

Hierarchy level 1. Method `variance_premium`.

> Daily payoff of a short variance position: yesterday's implied variance minus today's realised squared return.

**Construction**

$$
f_t = \frac{1}{252}\left(\frac{\text{VIX}_{t-1}}{100}\right)^2 - \bigl(r^{\text{SPY}}_t\bigr)^2
$$

$$
f_t \leftarrow k\,f_t, \qquad k = \frac{0.1 / \sqrt{252}}{\operatorname{sd}(f)}
$$

1. Yesterday's implied variance is what a variance swap struck at t-1 pays against, so the lag is the contract and not a modelling choice.
2. Today's realised variance is the squared SPY return, which is what the daily leg of such a swap settles on.
3. The difference is the daily payoff of a short variance position, positive on average because implied sits above realised.
4. The payoff is convex in the underlying, so a linear beta on it is a first approximation and nothing more (section 2.2).
5. Rescale to 10% annualised volatility. The transform above yields a z-score, and section 2.2 requires every factor to be a return.

| Symbol | Source | Meaning |
|---|---|---|
| `FRED:VIXCLS` | `FRED:VIXCLS` | VIX close in index points; divided by 100 it is an annualised volatility in decimal |
| `r_t^SPY` | `SPY` | underlying whose realised variance is delivered: log total return of SPY |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `eq_global`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `vol_rates` — Rates Volatility (realised)

Hierarchy level 1. Method `realized_vol_change`.

> Substitute for ^TYVIX, which CBOE discontinued in 2020.

**Construction**

$$
v_t = \sqrt{252}\;\operatorname{sd}\bigl(r^{\text{TLT}}_{t-20},\dots,r^{\text{TLT}}_t\bigr)
$$

$$
f_t = \frac{\Delta v_t}{\operatorname{sd}(\Delta v_{t-251},\dots,\Delta v_t)}
$$

$$
f_t \leftarrow k\,f_t, \qquad k = \frac{0.1 / \sqrt{252}}{\operatorname{sd}(f)}
$$

1. Realised volatility over a trailing 21 days, annualised.
2. Differenced, because a volatility level is not stationary.
3. Standardised on a trailing window, because a change in volatility is not measured in return units.
4. This is a change in a risk measure, not a tradable return. It stands in for a rates-volatility series because CBOE discontinued ^TYVIX in 2020.
5. Rescale to 10% annualised volatility. The transform above yields a z-score, and section 2.2 requires every factor to be a return.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^TLT` | `TLT` | instrument whose realised volatility is tracked: log total return of TLT |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1$ = `rt_us_level`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 8. Liquidity & Stress

### `liq_funding` — Funding Spread (SOFR-EFFR)

Hierarchy level 0. Method `spread_level`.

> TED and LIBOR-OIS are discontinued; SOFR over the effective funds rate is the live equivalent. Standardised because basis points are not comparable to the return-scaled factors.

**Construction**

$$
s_t = \text{FRED:SOFR}_t - \text{FRED:EFFR}_t
$$

$$
f_t = \frac{\Delta s_t}{\operatorname{sd}\bigl(\Delta s_{t-251},\dots,\Delta s_t\bigr)}
$$

$$
f_t \leftarrow k\,f_t, \qquad k = \frac{0.1 / \sqrt{252}}{\operatorname{sd}(f)}
$$

1. Form the spread between the two quoted levels.
2. Divide the daily change by its own trailing volatility. Basis points are not comparable with the return-scaled blocks, and the scaling window is strictly trailing: a full-sample standard deviation would leak future volatility into every historical observation.
3. Rescale to 10% annualised volatility. The transform above yields a z-score, and section 2.2 requires every factor to be a return.

| Symbol | Source | Meaning |
|---|---|---|
| `FRED:SOFR` | `FRED:SOFR` | first leg of the spread, in percent |
| `FRED:EFFR` | `FRED:EFFR` | second leg of the spread, in percent |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `liq_conditions` — Financial Conditions (NFCI)

Hierarchy level 0. Method `level_transform`.

> Weekly series; the daily factor is zero between releases rather than forward-filled, since a forward fill would manufacture information (PDF section 3, Grundregel).

**Construction**

$$
f_t = \begin{cases}\dfrac{\Delta s_k}{\operatorname{sd}(\Delta s_{k-51},\dots,\Delta s_k)} & t = \tau_k,\ \text{a release day} \\[2ex]0 & \text{otherwise}\end{cases}
$$

$$
f_t \leftarrow k\,f_t, \qquad k = \frac{0.1 / \sqrt{252}}{\operatorname{sd}(f)}
$$

1. The series prints weekly, not daily.
2. It is NOT forward filled. A filled series carries no new information between releases, which manufactures autocorrelation, understates standard errors and can smuggle in lookahead (section 3, Grundregel).
3. Instead the standardised change lands on the publication day and the factor is exactly zero in between: a release-event factor, sparse daily (section 3.2).
4. The scaling window counts releases rather than days, so 52 of them is about a year of the series' own history.
5. Rescale to 10% annualised volatility. The transform above yields a z-score, and section 2.2 requires every factor to be a return.

| Symbol | Source | Meaning |
|---|---|---|
| `FRED:NFCI` | `FRED:NFCI` | the published level series |

**Orthogonalisation**

$$
\tilde{f}_t = f_t
$$

Nothing sits above this factor in the hierarchy, so the raw and orthogonalised panels hold the same series.

---

### `liq_risk_off` — Cross-Asset Risk-Off

Hierarchy level 2. Method `basket`.

> The residual co-movement of classic haven and risk assets after each leg's own block exposure is removed.

**Construction**

$$
f_t = \frac{r^{\text{TLT}}_t + r^{\text{UUP}}_t + r^{\text{GLD}}_t}{3} - \frac{r^{\text{HYG}}_t + r^{\text{EEM}}_t}{2}
$$

1. Equally weighted long basket of 3: each leg carries 1/3 of the notional.
2. Minus an equally weighted short basket of 2. Both sides carry the market direction, so it cancels and what survives is the spread between them.
3. Self-financing, so no cash leg.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^TLT` | `TLT` | long leg: log total return of TLT |
| `r_t^UUP` | `UUP` | long leg: log total return of UUP |
| `r_t^GLD` | `GLD` | long leg: log total return of GLD |
| `r_t^HYG` | `HYG` | short leg: log total return of HYG |
| `r_t^EEM` | `EEM` | short leg: log total return of EEM |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{cr\_hy}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{3}$ = `eq_global`, `rt_us_level`, `cr_hy`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## 9. Alternative Risk Premia

### `arp_trend` — Time-Series Momentum

Hierarchy level 3. Method `tsmom`.

> 12-month-minus-1 signal, inverse-volatility sized, equally weighted across assets. Validated against DBMF and KMLM over their short lives.

**Construction**

$$
S_{i,t} = \operatorname{sign}\Bigl(\sum_{u=t-252}^{t-22} r_{i,u}\Bigr), \qquad v_{i,t} = \operatorname{sd}\bigl(r_{i,t-63},\dots,r_{i,t-1}\bigr)
$$

$$
w_{i,t} = \frac{S_{i,t}}{v_{i,t}}, \qquad f_t = \frac{\sum_i w_{i,t}\,r_{i,t}}{\sum_i \lvert w_{i,t}\rvert}
$$

1. The signal at t reads returns through t-22 only, and the position it implies is applied to the return at t. Nothing about day t enters its own weight.
2. Skipping the most recent 21 days is the standard 12-minus-1 construction, which keeps short-term reversal out of the trend signal.
3. Inverse-volatility sizing puts each of the 13 instruments on a comparable risk footing; without it the equity legs would drown out the bond legs.
4. Dividing by gross exposure makes the factor a return on one unit of notional rather than a sum that grows with the size of the universe.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^SPY` | `SPY` | instrument return: log total return of SPY |
| `r_t^EFA` | `EFA` | instrument return: log total return of EFA |
| `r_t^EEM` | `EEM` | instrument return: log total return of EEM |
| `r_t^TLT` | `TLT` | instrument return: log total return of TLT |
| `r_t^IEF` | `IEF` | instrument return: log total return of IEF |
| `r_t^DBC` | `DBC` | instrument return: log total return of DBC |
| `r_t^GLD` | `GLD` | instrument return: log total return of GLD |
| `r_t^USO` | `USO` | instrument return: log total return of USO |
| `r_t^HYG` | `HYG` | instrument return: log total return of HYG |
| `r_t^LQD` | `LQD` | instrument return: log total return of LQD |
| `r_t^UUP` | `UUP` | instrument return: log total return of UUP |
| `r_t^FXE` | `FXE` | instrument return: log total return of FXE |
| `r_t^FXY` | `FXY` | instrument return: log total return of FXY |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{rt\_us\_level}}_t - \hat{b}_{3,\tau}\,\tilde{f}^{\text{cm\_broad}}_t - \hat{b}_{4,\tau}\,\tilde{f}^{\text{fx\_usd}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, \dots, g_{4}$ = `eq_global`, `rt_us_level`, `cm_broad`, `fx_usd`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

### `arp_xs_momentum` — Cross-Sectional Momentum

Hierarchy level 3. Method `xs_momentum`.

> Long the top third, short the bottom third by trailing return; dollar-neutral by construction.

**Construction**

$$
M_{i,t} = \sum_{u=t-252}^{t-22} r_{i,u}, \qquad q_{i,t} = \operatorname{rank}_\text{pct}\bigl(M_{\cdot,t}\bigr)_i
$$

$$
w_{i,t} = \frac{\mathbf{1}\{q_{i,t} > 2/3\}}{n^{\text{long}}_t} - \frac{\mathbf{1}\{q_{i,t} < 1/3\}}{n^{\text{short}}_t}, \qquad f_t = \sum_i w_{i,t}\,r_{i,t}
$$

1. Rank the 13 instruments by trailing return, again skipping the most recent 21 days.
2. Long the top third, short the bottom third, equally weighted within each leg.
3. Dollar-neutral by construction: the long and the short weights each sum to one, so the common market direction cancels and what is left is the spread between winners and losers.

| Symbol | Source | Meaning |
|---|---|---|
| `r_t^SPY` | `SPY` | instrument return: log total return of SPY |
| `r_t^EFA` | `EFA` | instrument return: log total return of EFA |
| `r_t^EEM` | `EEM` | instrument return: log total return of EEM |
| `r_t^TLT` | `TLT` | instrument return: log total return of TLT |
| `r_t^IEF` | `IEF` | instrument return: log total return of IEF |
| `r_t^DBC` | `DBC` | instrument return: log total return of DBC |
| `r_t^GLD` | `GLD` | instrument return: log total return of GLD |
| `r_t^USO` | `USO` | instrument return: log total return of USO |
| `r_t^HYG` | `HYG` | instrument return: log total return of HYG |
| `r_t^LQD` | `LQD` | instrument return: log total return of LQD |
| `r_t^UUP` | `UUP` | instrument return: log total return of UUP |
| `r_t^FXE` | `FXE` | instrument return: log total return of FXE |
| `r_t^FXY` | `FXY` | instrument return: log total return of FXY |

**Orthogonalisation**

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \hat{b}_{1,\tau}\,\tilde{f}^{\text{eq\_global}}_t - \hat{b}_{2,\tau}\,\tilde{f}^{\text{arp\_trend}}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

with $g_1, g_2$ = `eq_global`, `arp_trend`, each taken as its own *orthogonalised* series, and $\tau$ the most recent refit at or before $t$.

---

## The orthogonalisation, once

The equation above is the same for every factor that has targets; only the regressor list changes. In general form, for $k$ targets $g_1, \dots, g_k$:

$$
\tilde{f}_t = f_t - \hat{a}_\tau - \sum_{j=1}^{k} \hat{b}_{j,\tau}\,\tilde{f}^{g_j}_t
$$

$$
(\hat{a}_\tau, \hat{b}_\tau) = \arg\min_{a,b} \sum_{s=\tau-504}^{\tau-1} \Bigl(f_s - a - \sum_j b_j\,\tilde{f}^{g_j}_s\Bigr)^2
$$

1. Coefficients come from ordinary least squares on the trailing 504 observations (2 years), refitted every 21 observations and applied forward until the next refit.
2. The fitting window ends at tau-1, so day t is never part of the regression that residualises it. That is the whole reason for the rolling refit: a full-sample residual would put future information into historical factor values, and this project's headline output is a predicted-versus-realised risk comparison, which such a leak would flatter exactly where the model is being judged.
3. The regressors are the targets' own orthogonalised series, so the hierarchy compounds: by the time a level-3 factor is residualised, the level-0 factors it sees have already had everything above them removed.
4. Before the first refit, with fewer than 252 usable observations or a missing regressor on the day, the orthogonalised value is NULL rather than a silent fallback to the raw value. A factor that quietly stops being orthogonal on some dates is worse than one with a documented gap.

Two alternative modes exist and are not used. `expanding` fits on all history to date: causal, but it never forgets, so a loading that was high in one crisis stays high for a decade and the residual acquires a systematic negative loading on its own regressor. `full_sample` fits once on everything: exactly orthogonal, and what commercial risk models do, but every historical factor value then contains information from its own future. Measured residual correlation of `eq_em` against `eq_global` over 2009-2026: rolling -0.05, expanding -0.34, full sample 0.00.
