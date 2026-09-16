/**
 * What counts as a good outcome, in one place.
 *
 * Green and red are a claim, not decoration: they say the number in front of you
 * is or is not what the model wants. Two pages run the same battery on different
 * series — the Factor Explorer on the orthogonalised factors, the Raw Explorer on
 * the raw ones and their underlying instruments — and if they coloured the same
 * p-value differently, one of them would be lying. So the rules live here, the
 * same way `backend/core/summary.py` exists so the two cannot disagree about a
 * Sharpe ratio.
 *
 * Three tones, and the third is the one that matters most:
 *
 *   good     the test says what we want
 *   warn     the test flags something worth knowing, and nothing is gated on it
 *   neutral  there is no good or bad here
 *
 * Nothing gets `bad` from a diagnostic. A red light should mean "this series is
 * not usable", and only the joint ADF/KPSS verdict makes that claim — the battery
 * is read jointly for exactly that reason. A single rejected ARCH-LM test painted
 * red would say a GARCH process is broken, which is false: it is strictly
 * stationary, and volatility clustering is the normal condition of a daily return
 * series. Same for Jarque-Bera: returns are fat-tailed, everyone knows it, and the
 * model does not assume otherwise.
 */

export type Tone = "good" | "warn" | "bad" | "neutral";

export interface Assessment {
  tone: Tone;
  /** Why, in a few words. Rendered under the figure or as a tooltip. */
  reason: string;
}

const SIG = 0.05;

/** Below this, a p-value is a rejection. Exported so captions can state it. */
export const SIGNIFICANCE = SIG;

function p(value: number | null | undefined): number | null {
  return value === null || value === undefined || !isFinite(value) ? null : value;
}

// ---------------------------------------------------------------------------
// the stationarity battery
// ---------------------------------------------------------------------------

/**
 * H0 is a unit root, so rejecting is what we want. Shared by ADF and
 * Phillips-Perron, which differ only in how they handle serial correlation.
 */
function rejectsUnitRoot(pv: number | null, name: string): Assessment {
  if (pv === null) return { tone: "neutral", reason: "not computed" };
  return pv < SIG
    ? { tone: "good", reason: `${name} rejects a unit root` }
    : { tone: "warn", reason: `${name} cannot reject a unit root` };
}

export const assess = {
  adf: (pv: number | null | undefined) => rejectsUnitRoot(p(pv), "ADF"),

  pp: (pv: number | null | undefined) => rejectsUnitRoot(p(pv), "Phillips-Perron"),

  /** H0 is stationarity — the opposite null, so here we want a *high* p-value. */
  kpss: (pv: number | null | undefined): Assessment => {
    const v = p(pv);
    if (v === null) return { tone: "neutral", reason: "not computed" };
    return v > SIG
      ? { tone: "good", reason: "KPSS does not reject stationarity" }
      : { tone: "warn", reason: "KPSS rejects stationarity" };
  },

  /**
   * H0 is no autocorrelation. Rejection is the second stale-pricing signal and is
   * worth seeing, but it does not make a series unusable on its own — a
   * mildly autocorrelated return series is still stationary.
   */
  ljungBox: (pv: number | null | undefined): Assessment => {
    const v = p(pv);
    if (v === null) return { tone: "neutral", reason: "not computed" };
    return v > SIG
      ? { tone: "good", reason: "no autocorrelation" }
      : { tone: "warn", reason: "autocorrelated — check for stale pricing" };
  },

  /**
   * Recorded, never gated. A GARCH process is strictly stationary, so rejection
   * here is the expected state of a daily return series and says nothing about
   * whether the series may be used.
   */
  archLm: (pv: number | null | undefined): Assessment => {
    const v = p(pv);
    if (v === null) return { tone: "neutral", reason: "not computed" };
    return {
      tone: "neutral",
      reason: v < SIG
        ? "volatility clusters — expected, and not gated"
        : "no detectable clustering",
    };
  },

  /** Likewise informational: daily returns are fat-tailed and the model knows it. */
  jarqueBera: (pv: number | null | undefined): Assessment => {
    const v = p(pv);
    if (v === null) return { tone: "neutral", reason: "not computed" };
    return {
      tone: "neutral",
      reason: v < SIG ? "non-normal — expected, drives the t-fit" : "normal-looking",
    };
  },

  /**
   * Lo-MacKinlay variance ratio. One is a random walk; above one is stale or
   * smoothed pricing, below one is bid-ask bounce. The band is deliberately wide:
   * a daily VR wanders on sampling noise alone, and a 0.95 does not mean anything
   * is wrong.
   */
  varianceRatio: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (Math.abs(x - 1) <= 0.15) return { tone: "good", reason: "close to a random walk" };
    return x > 1
      ? { tone: "warn", reason: "above 1 — smoothed or stale pricing" }
      : { tone: "warn", reason: "below 1 — mean reversion or bid-ask bounce" };
  },

  /** First-order autocorrelation. Near zero is what a tradable return looks like. */
  autocorrelation: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (Math.abs(x) <= 0.05) return { tone: "good", reason: "negligible" };
    return {
      tone: "warn",
      reason: x > 0 ? "positive — smoothing or stale prices" : "negative — bounce",
    };
  },

  /**
   * Share of days with an exactly zero return. The cheapest illiquidity check and
   * often the most telling: a series that does not move on 20% of days is not
   * being priced on those days.
   */
  zeroReturns: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (x <= 0.02) return { tone: "good", reason: "prices every day" };
    if (x <= 0.10) return { tone: "warn", reason: "some non-trading days" };
    return { tone: "bad", reason: "illiquid — a fifth of days do not move" };
  },

  /**
   * The joint ADF x KPSS verdict as the diagnostics job stored it. This is the one
   * place a red light is earned, because it is the only judgement the pipeline
   * actually gates on.
   */
  verdict: (v: string | null | undefined): Assessment => {
    if (v === "pass") return { tone: "good", reason: "usable as a model input" };
    if (v === "warn") return { tone: "warn", reason: "usable, with a caveat" };
    if (v === "fail") return { tone: "bad", reason: "not usable as a model input" };
    return { tone: "neutral", reason: "not assessed" };
  },
};

// ---------------------------------------------------------------------------
// the regression design
// ---------------------------------------------------------------------------

/** Thresholds shared with the risk pipeline's reliability gate. */
export const MAX_CONDITION_NUMBER = 200;
export const MAX_VIF = 100;

export const design = {
  /**
   * Durbin-Watson on the regression residuals. Two means no autocorrelation;
   * either side of it says the residual carries structure the factors missed.
   */
  durbinWatson: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    const off = Math.abs(x - 2);
    if (off < 0.3) return { tone: "good", reason: "residuals are independent" };
    if (off < 0.6) return { tone: "warn", reason: "mild residual autocorrelation" };
    return { tone: "bad", reason: "residuals are autocorrelated" };
  },

  /**
   * Condition number of the design. The invariant that governs how far beta'
   * Sigma beta can blow up, so it means the same thing on either panel.
   */
  conditionNumber: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (x <= MAX_CONDITION_NUMBER) return { tone: "good", reason: "well conditioned" };
    return { tone: "bad",
             reason: `above ${MAX_CONDITION_NUMBER} — excluded from risk scoring` };
  },

  /**
   * Maximum variance inflation factor, and the one figure whose meaning depends
   * on which panel produced it.
   *
   * On the orthogonalised panel a VIF above 100 is anomalous and the risk
   * pipeline gates on it. On the raw panel it is the definition of the factor
   * set: eq_us regressed on the other thirty-nine raw factors has an R-squared
   * near 0.9998 because eq_global is one of them. Painting that red every time
   * would say the panel is broken when it is merely collinear, which is exactly
   * the judgement `run_risk.reliability` was changed to stop making.
   */
  maxVif: (v: number | null | undefined,
           orthogonalized = true): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (!orthogonalized) {
      return x > 1000
        ? { tone: "warn", reason: "near-collinear even for a raw panel" }
        : { tone: "neutral",
            reason: "collinear by construction on the raw panel — not gated" };
    }
    if (x <= MAX_VIF) return { tone: "good", reason: "no factor is redundant" };
    return { tone: "bad", reason: `above ${MAX_VIF} — excluded from risk scoring` };
  },
};

// ---------------------------------------------------------------------------
// the risk backtest
// ---------------------------------------------------------------------------

export const backtest = {
  /** Mean realised / predicted. One is right; the bands are the usual tolerance. */
  bias: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    const off = Math.abs(x - 1);
    if (off < 0.15) return { tone: "good", reason: "forecast level is right" };
    if (off < 0.35) {
      return { tone: "warn",
               reason: x > 1 ? "forecast runs low" : "forecast runs high" };
    }
    return { tone: "bad",
             reason: x > 1 ? "badly under-forecasts" : "badly over-forecasts" };
  },

  /** Standard deviation of r_t / sigma_hat. One means the scaling is right. */
  zStd: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    const off = Math.abs(x - 1);
    if (off < 0.10) return { tone: "good", reason: "standardised returns have unit sd" };
    if (off < 0.25) return { tone: "warn", reason: "scaling is slightly off" };
    return { tone: "bad", reason: "standardised returns are not unit variance" };
  },

  /**
   * Mincer-Zarnowitz slope. One is unbiased, and a slope *below* one is partly
   * mechanical whenever volatility moves faster than the measurement horizon:
   * realised volatility over a forward window blends regimes the forecast refers
   * to singly. On simulated data with a perfect forecast the slope comes out at
   * 0.50 with 21-day regimes and 1.02 with 504-day ones. So a low slope is amber,
   * not red: it is evidence worth reading, not proof the model over-reacts.
   */
  mzSlope: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (Math.abs(x - 1) < 0.25) return { tone: "good", reason: "unbiased" };
    if (x <= 0) return { tone: "bad", reason: "forecast moves the wrong way" };
    return x < 1
      ? { tone: "warn", reason: "below 1 — partly mechanical at this horizon" }
      : { tone: "warn", reason: "above 1 — forecast under-reacts" };
  },

  /** H0: a = 0 and b = 1 jointly. Not rejecting is what an unbiased forecast does. */
  mzJoint: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    return x > SIG
      ? { tone: "good", reason: "cannot reject an unbiased forecast" }
      : { tone: "warn", reason: "rejects unbiasedness" };
  },

  /** How much of the variation in realised risk the forecast explains. */
  mzR2: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    if (x >= 0.10) return { tone: "good", reason: "tracks the variation, not just the level" };
    if (x >= 0.03) return { tone: "warn", reason: "weak timing ability" };
    return { tone: "warn", reason: "level only — little timing ability" };
  },

  /** Kupiec: is the *number* of VaR breaches right? */
  kupiec: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    return x > SIG
      ? { tone: "good", reason: "breach count is right" }
      : { tone: "bad", reason: "wrong number of breaches" };
  },

  /**
   * Christoffersen: are the breaches independent? The more dangerous failure, and
   * one Kupiec cannot see — a model can have exactly the right breach count and
   * take all of them in one week.
   */
  christoffersen: (v: number | null | undefined): Assessment => {
    const x = p(v);
    if (x === null) return { tone: "neutral", reason: "not computed" };
    return x > SIG
      ? { tone: "good", reason: "breaches are independent" }
      : { tone: "bad", reason: "breaches cluster — the dangerous failure" };
  },
};

// ---------------------------------------------------------------------------
// performance figures
// ---------------------------------------------------------------------------

/**
 * Sign colouring, and only for the figures where a sign means something.
 *
 * A return, a Sharpe ratio and a cumulative total have a direction an investor
 * cares about. Volatility, skew, kurtosis, drawdown, VaR and expected shortfall
 * do not: there is no good or bad volatility for a factor, and a drawdown is
 * negative by definition, so painting it red every time is noise rather than
 * information. Those stay in the plain colour.
 */
export function bySign(v: number | null | undefined): Tone {
  if (v === null || v === undefined || !isFinite(v)) return "neutral";
  if (v > 0) return "good";
  if (v < 0) return "bad";
  return "neutral";
}
