"use client";

/**
 * A candidate's estimation and forecast written out as mathematics, with its own
 * numbers in place - the window, the half-life, the penalty, the effective
 * sample - so the formula on the card is this model's, not a generic one.
 *
 * The same notation as the definitions in lib/agentMetrics.ts and the methodology
 * appendix: tau a regression date, t a forecast date, W a window, h a half-life,
 * K the number of factors.
 */

import { Equation } from "@/components/Math";

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "?");
const lam = (h: number) => Math.pow(0.5, 1 / h);

/** Kish's effective sample size of weights lambda^k over W days. */
export function nEff(h: number, W?: number): number {
  const l = lam(h);
  if (!W) return (1 + l) / (1 - l);
  let s = 0, s2 = 0;
  for (let k = 0; k < W; k++) { const w = Math.pow(l, k); s += w; s2 += w * w; }
  return (s * s) / s2;
}

export function Formulas({ lines, caption }: { lines: string[]; caption?: string }) {
  const latex = lines.join("\n\n");
  return (
    <div className="mt-2">
      <div className="mb-1 text-2xs uppercase tracking-label text-muted">
        {caption ?? "In formulas"}
      </div>
      <div className="text-[12.5px]"><Equation latex={latex} plain={latex} /></div>
    </div>
  );
}

/** The regression behind the betas. */
export function estimationLines(e: Record<string, any>, K: number, nRegression: number): string[] {
  const W = Number(e.window_days ?? 252);
  const out: string[] = [];
  if (e.dimson_lags) {
    out.push(String.raw`y_t = \alpha + \sum_{j=1}^{${K}} \beta_{j,0}\, f_{j,t} + \sum_{l=1}^{${e.dimson_lags}} \sum_{j \in D} \beta_{j,l}\, f_{j,t-l} + \varepsilon_t, \qquad t = \tau - ${W} + 1, \dots, \tau`);
  } else {
    out.push(String.raw`y_t = \alpha + \sum_{j=1}^{${K}} \beta_j\, f_{j,t} + \varepsilon_t, \qquad t = \tau - ${W} + 1, \dots, \tau`);
  }
  if (e.estimator === "ridge") {
    out.push(String.raw`(\hat\alpha, \hat\beta) = \arg\min \sum_t w_t\, \varepsilon_t^2 + \lambda n \sum_j \sigma_j^2 \beta_j^2, \qquad \lambda = ${e.ridge_lambda}`);
  } else if (e.estimator === "huber") {
    out.push(String.raw`(\hat\alpha, \hat\beta) = \arg\min \sum_t w_t\, \rho_c\!\left(\frac{\varepsilon_t}{\hat s}\right), \qquad \rho_c(u) = \begin{cases} u^2/2 & \lvert u \rvert \le c \\ c \lvert u \rvert - c^2/2 & \lvert u \rvert > c \end{cases}, \quad c = 1.345`);
  } else {
    out.push(String.raw`(\hat\alpha, \hat\beta) = \arg\min \sum_t w_t\, \varepsilon_t^2`);
  }
  if (e.weighting === "ewma") {
    const h = Number(e.ewma_halflife);
    out.push(String.raw`w_t = 0.5^{(\tau - t)/${h}} \quad\Rightarrow\quad n_{\text{eff}} = \frac{\left(\sum_t w_t\right)^2}{\sum_t w_t^2} = ${fmt(nRegression, 0)} \text{ of } ${W} \text{ days}`);
  } else {
    out.push(String.raw`w_t = 1 \quad\Rightarrow\quad n_{\text{eff}} = ${W}, \qquad \frac{n_{\text{eff}}}{K} = ${fmt(W / Math.max(K, 1), 0)} \text{ days per factor}`);
  }
  out.push(String.raw`\tau \in \{\tau_0,\ \tau_0 + 21,\ \tau_0 + 42,\ \dots\} \qquad \text{(a new regression every 21 trading days)}`);
  return out;
}

/** The forecast assembled from the betas. */
export function riskLines(e: Record<string, any>, r: Record<string, any>, K: number,
                          horizon: number): string[] {
  const out: string[] = [];
  const debias = r.debias ? String.raw` - \operatorname{tr}\!\left(\hat\Sigma_t \hat V_\tau\right)` : "";
  if (e.dimson_lags) {
    out.push(String.raw`\hat\sigma^2_{t+1} = \sum_{l=0}^{${e.dimson_lags}} \hat\beta_{\tau,l}^{\prime}\, \hat\Sigma_t\, \hat\beta_{\tau,l}${debias} + \hat\sigma^2_{\varepsilon,t}`);
  } else {
    out.push(String.raw`\hat\sigma^2_{t+1} = \underbrace{\hat\beta_\tau^{\prime}\, \hat\Sigma_t\, \hat\beta_\tau${debias}}_{\text{systematic}} + \underbrace{\hat\sigma^2_{\varepsilon,t}}_{\text{specific}}`);
  }
  const W = Number(r.cov_window ?? 504);
  const h = Number(r.halflife ?? 60);
  const l = lam(h);
  const ewma = String.raw`\hat\Sigma^{\text{EWMA}}_t = \frac{1}{1 - \sum_k \tilde w_k^2} \sum_{k=0}^{${W - 1}} \tilde w_k \left(f_{t-k} - \bar f\right)\left(f_{t-k} - \bar f\right)^{\prime}, \qquad \tilde w_k = \frac{\lambda^k}{\sum_j \lambda^j}, \quad \lambda = 0.5^{1/${h}} = ${fmt(l, 4)}`;
  if (r.cov_method === "ewma") {
    out.push(ewma.replace(String.raw`\hat\Sigma^{\text{EWMA}}_t`, String.raw`\hat\Sigma_t`));
  } else if (r.cov_method === "blend") {
    out.push(String.raw`\hat\Sigma_t = ${fmt(Number(r.ewma_weight ?? 0.6), 1)}\, \hat\Sigma^{\text{EWMA}}_t + ${fmt(1 - Number(r.ewma_weight ?? 0.6), 1)}\, \hat\Sigma^{\text{LW}}_t`);
    out.push(ewma);
    out.push(String.raw`\hat\Sigma^{\text{LW}}_t = \hat\delta\, F + (1 - \hat\delta)\, S, \qquad S = \text{sample covariance over } ${W} \text{ days}, \quad F = \text{constant-correlation target}`);
  } else if (r.cov_method === "ledoit_wolf") {
    out.push(String.raw`\hat\Sigma_t = \hat\delta\, F + (1 - \hat\delta)\, S, \qquad S = \text{sample covariance over } ${W} \text{ days}, \quad F = \text{constant-correlation target}`);
  } else {
    out.push(String.raw`\hat\Sigma_t = \frac{1}{${W - 1}} \sum_{k=0}^{${W - 1}} \left(f_{t-k} - \bar f\right)\left(f_{t-k} - \bar f\right)^{\prime}`);
  }
  if (r.specific === "ewma") {
    const hs = Number(r.specific_halflife ?? 63);
    out.push(String.raw`\hat\sigma^2_{\varepsilon,t} = \max\!\left(0.02^2,\ 252 \cdot \mathrm{EWMA}_{${hs}}\!\left[e_s^2 - f_s^{\prime} \hat V f_s\right]_{s \le t}\right), \qquad e_s = y_s - \hat\beta_{\tau(s)}^{\prime} f_s`);
  } else {
    out.push(String.raw`\hat\sigma^2_{\varepsilon} = \max\!\left(0.02^2,\ \frac{252}{n - k - 1} \sum_t \hat\varepsilon_t^2\right) \qquad \text{(residuals of the latest regression)}`);
  }
  if (r.var_dist === "t") {
    out.push(String.raw`\mathrm{VaR}^{\alpha}_{t+1} = \frac{t^{-1}_{\hat\nu}(\alpha)}{\sqrt{\hat\nu / (\hat\nu - 2)}}\, \frac{\hat\sigma_{t+1}}{\sqrt{252}}, \qquad \hat\nu = \text{fitted to the trailing 756 standardised returns}`);
  } else {
    out.push(String.raw`\mathrm{VaR}^{\alpha}_{t+1} = z_\alpha\, \frac{\hat\sigma_{t+1}}{\sqrt{252}}`);
  }
  const L = Number(r.nw_lags ?? 0);
  if (L > 0) {
    out.push(String.raw`\hat\sigma^2_{${horizon}\text{d}} = \hat\beta^{\prime}\left[\hat\Sigma_t + \sum_{l=1}^{${L}} \left(1 - \frac{l}{${horizon}}\right)\left(\hat\Gamma_l + \hat\Gamma_l^{\prime}\right)\right]\hat\beta + \hat\sigma^2_{\varepsilon,t}, \qquad \hat\Gamma_l = \widehat{\operatorname{cov}}\left(f_t, f_{t-l}\right)`);
  }
  return out;
}

/**
 * What the covariance's effective sample means for a K-factor matrix: the noise
 * in one correlation, in one variance, and in the matrix as a whole.
 */
export function NoiseBlock({ n, K, longWindow }: { n: number; K: number; longWindow?: number }) {
  if (!(n > 0) || K < 2) return null;
  const entries = (K * (K + 1)) / 2;
  const q = K / n;
  const seRho = 1 / Math.sqrt(n);
  const seVar = Math.sqrt(2 / n);
  const lines = [
    String.raw`\frac{K(K+1)}{2} = ${entries} \text{ entries to estimate}, \qquad \frac{n_{\text{eff}}}{K} = \frac{${fmt(n, 0)}}{${K}} = ${fmt(n / K, 1)} \text{ observations per factor}`,
    String.raw`\operatorname{se}(\hat\rho_{ij}) \approx \frac{1 - \rho_{ij}^2}{\sqrt{n_{\text{eff}}}} \le ${fmt(seRho, 2)}, \qquad \frac{\operatorname{se}(\hat\sigma_j^2)}{\sigma_j^2} \approx \sqrt{\frac{2}{n_{\text{eff}}}} = ${fmt(seVar, 2)}`,
  ];
  if (q < 1) {
    const lo = Math.pow(1 - Math.sqrt(q), 2), hi = Math.pow(1 + Math.sqrt(q), 2);
    lines.push(String.raw`q = \frac{K}{n_{\text{eff}}} = ${fmt(q, 2)} \quad\Rightarrow\quad \text{eigenvalues of } \hat R \in \left[(1 - \sqrt q)^2,\ (1 + \sqrt q)^2\right] = \left[${fmt(lo, 2)},\ ${fmt(hi, 2)}\right] \text{ when } R = I`);
  }
  return (
    <div className="mt-3 rounded border border-lineSoft bg-white px-3 py-2">
      <div className="text-[11.5px] font-semibold text-navy">
        What {fmt(n, 0)} effective days buy for a {K}-factor covariance
      </div>
      <Formulas lines={lines} caption=" " />
      <p className="mt-1.5 text-[11.5px] leading-snug text-muted">
        A correlation that is truly zero is typically estimated anywhere within about
        ±{fmt(2 * seRho, 2)}, and a factor&apos;s variance is off by more than{" "}
        {fmt(100 * seVar, 0)}% about one time in three.
        {q < 1
          ? <> Even if all {K} factors were truly uncorrelated with unit variance, the
              estimated matrix would show directions with up to{" "}
              {fmt(Math.pow(1 + Math.sqrt(q), 2), 1)} times and as little as{" "}
              {fmt(Math.pow(1 - Math.sqrt(q), 2), 2)} times the true variance - pure
              noise (Marchenko-Pastur).</>
          : <> With fewer effective days than factors the matrix is singular on its
              own.</>}
        {" "}A single stock&apos;s β′Σβ averages much of this out, which is why its
        daily loss tolerates a short half-life; a hedged portfolio, which loads on the
        small directions, does not.
        {longWindow ? <> Here the exponential part is blended with a shrunk
          long-run matrix over {longWindow} days, which damps the noise.</> : null}
      </p>
    </div>
  );
}
