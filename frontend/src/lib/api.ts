/**
 * Typed client for the factor-model API.
 *
 * Requests go to a same-origin /api path, which next.config.js rewrites to the
 * FastAPI host, so the browser never deals with CORS in development.
 */

const BASE = "";

async function get<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const qs = params
    ? "?" +
      Object.entries(params)
        .filter(([, v]) => v !== undefined && v !== null && v !== "")
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join("&")
    : "";
  const res = await fetch(`${BASE}${path}${qs}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await describeError(res, path));
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await describeError(res, path));
  return res.json() as Promise<T>;
}

/** Surface the API's own explanation rather than a bare status code. */
async function describeError(res: Response, path: string): Promise<string> {
  try {
    const body = await res.json();
    if (body?.detail) return String(body.detail);
  } catch {
    /* fall through to the status line */
  }
  return `${res.status} ${res.statusText} — ${path}`;
}

// --- types ---------------------------------------------------------------

export type Verdict = "pass" | "warn" | "fail";

export interface FactorMeta {
  factor_id: string;
  block_id: string;
  name: string;
  hierarchy_level: number;
  version: number;
  orthogonalize_against: string[];
  construction: { method: string; inputs: Record<string, unknown>; note?: string };
  first_date: string | null;
  last_date: string | null;
  n_obs: number | null;
  vol_pct: number | null;
  verdict: Verdict | null;
  verdict_reason: string | null;
  flags: string[] | null;
}

export interface Block {
  block_id: string;
  name: string;
  sort_order: number;
  description: string;
  n_factors: number;
}

export interface Instrument {
  instrument_id: string;
  source_ticker: string;
  asset_class: string | null;
  currency: string;
  role: string;
  is_total_return: boolean;
  is_live: boolean;
  first_obs: string | null;
  last_obs: string | null;
  n_obs: number | null;
  notes: string | null;
}

export interface Spec {
  spec_id: string;
  name: string;
  window_days: number;
  step_days: number;
  estimator: string;
  weighting: string;
  n_instruments: number;
  created_at: string;
}

export interface MatrixResult {
  names: string[];
  blocks: (string | null)[];
  method: string;
  n_obs: number;
  start: string;
  end: string;
  covariance: number[][];
  correlation: number[][];
  volatilities: number[];
  diagnostics: {
    shrink_intensity: number | null;
    blend_weight: number | null;
    min_eigenvalue: number;
    max_eigenvalue: number;
    condition_number: number;
    is_psd: boolean;
    psd_repaired: boolean;
    pc1_share: number;
    pc3_share: number;
  };
}

export interface LoadingResult {
  spec_id: string;
  spec: Record<string, unknown> | null;
  instrument_id: string;
  from_cache: boolean;
  window_ends: string[];
  factors: string[];
  betas: Record<string, (number | null)[]>;
  standard_errors: Record<string, (number | null)[]>;
  t_stats: Record<string, (number | null)[]>;
  diagnostics: Array<Record<string, number | string | null>>;
  latest: {
    window_end: string;
    loadings: Array<{
      factor_id: string;
      beta: number;
      se: number | null;
      t_stat: number | null;
      p_value: number | null;
      vif: number | null;
    }>;
  };
}

export interface RiskResult {
  instrument_id: string;
  spec_id: string;
  horizon_days: number;
  cov_method: string;
  n_forecasts: number;
  n_unreliable: number;
  interpretation: string[];
  forecasts: Array<{
    as_of_date: string;
    sigma_pred_ann: number;
    sigma_factor_ann: number;
    sigma_specific_ann: number;
    factor_risk_share: number | null;
    sigma_realized_ann: number | null;
    bias_ratio: number | null;
    var95_pred: number;
    var99_pred: number;
    condition_number: number | null;
    max_vif: number | null;
    is_reliable: boolean;
    unreliable_reason: string | null;
  }>;
  backtest: Record<string, number | string | null> | null;
}

// --- endpoints -----------------------------------------------------------

export const api = {
  health: () => get<{ status: string; as_of: string; active_factors: number }>("/api/health"),

  blocks: () => get<Block[]>("/api/meta/blocks"),
  factors: () => get<FactorMeta[]>("/api/meta/factors"),
  instruments: (role?: string, liveOnly = true) =>
    get<Instrument[]>("/api/meta/instruments", { role, live_only: liveOnly }),
  specs: () => get<Spec[]>("/api/meta/specs"),
  dataHealth: () => get<any>("/api/meta/data-health"),
  factorSparklines: () =>
    get<{ points: number; series: Record<string, number[]> }>(
      "/api/meta/factor-sparklines"),

  factorSeries: (id: string, p?: { start?: string; end?: string }) =>
    get<{ dates: string[]; returns: number[]; cumulative: number[] }>(
      `/api/factors/${id}/series`, p),
  factorStats: (id: string, p?: { start?: string; end?: string }) =>
    get<any>(`/api/factors/${id}/stats`, p),
  factorRollingRisk: (id: string, windows = "21,63,252", p?: { start?: string }) =>
    get<{ dates: string[]; series: Record<string, (number | null)[]> }>(
      `/api/factors/${id}/rolling-risk`, { windows, ...p }),
  // bins is deliberately optional: omitting it lets the server pick the bin count
  // by the Freedman-Diaconis rule, which is the whole point of computing it there.
  factorHistogram: (id: string, bins?: number) =>
    get<any>(`/api/factors/${id}/histogram`, { bins }),
  factorQQ: (id: string) => get<any>(`/api/factors/${id}/qq`),
  factorACF: (id: string, lags = 30) => get<any>(`/api/factors/${id}/acf`, { lags }),
  factorDiagnostics: (id: string) => get<any>(`/api/factors/${id}/diagnostics`),
  factorReference: (id: string) => get<any>(`/api/factors/${id}/reference-comparison`),

  matrix: (body: Record<string, unknown>) => post<MatrixResult>("/api/matrix", body),
  pca: (body: Record<string, unknown>) => post<any>("/api/matrix/pca", body),
  rollingCorrelation: (body: Record<string, unknown>) =>
    post<any>("/api/matrix/rolling-correlation", body),

  estimate: (body: Record<string, unknown>) =>
    post<LoadingResult>("/api/loadings/estimate", body),
  stability: (specId: string, instrumentId: string) =>
    get<any>(`/api/loadings/${specId}/${encodeURIComponent(instrumentId)}/stability`),

  securityRisk: (body: Record<string, unknown>) => post<RiskResult>("/api/risk/security", body),
  standardized: (specId: string, instrumentId: string, p?: Record<string, unknown>) =>
    get<any>(`/api/risk/standardized/${specId}/${encodeURIComponent(instrumentId)}`, p),
  decomposition: (specId: string, instrumentId: string) =>
    get<any>(`/api/risk/decomposition/${specId}/${encodeURIComponent(instrumentId)}`),
};

// --- formatting ----------------------------------------------------------

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || !isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined || !isFinite(v) ? "—" : v.toFixed(digits);

/** p-values below 0.001 read better as a bound than as 0.00. */
export const pval = (v: number | null | undefined) =>
  v === null || v === undefined || !isFinite(v)
    ? "—"
    : v < 0.001
    ? "<0.001"
    : v.toFixed(3);
