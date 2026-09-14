/**
 * Typed client for the factor-model API.
 *
 * Requests go to a same-origin /api path, which next.config.js rewrites to the
 * FastAPI host, so the browser never deals with CORS in development.
 */

const BASE = "";

// `object` rather than Record<string, unknown>: the typed query interfaces below
// have no index signature, and giving them one to satisfy this would throw away
// the checking they exist for.
async function get<T>(path: string, params?: object): Promise<T> {
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

/**
 * Surface the API's own explanation rather than a bare status code.
 *
 * The API answers errors as JSON with a `detail`, so anything that is not JSON did
 * not come from the API at all — it came from the proxy in front of it. That is
 * worth saying, because the two failures need opposite responses: a `detail` is
 * something the request got wrong, whereas a bare `Internal Server Error` after a
 * long wait means the computation was abandoned in flight and is probably still
 * running. Reporting the second as "500 Internal Server Error" sent a real
 * debugging session looking for a bug in an endpoint that had worked fine.
 */
async function describeError(res: Response, path: string): Promise<string> {
  let raw = "";
  try {
    raw = await res.text();
    const body = JSON.parse(raw);
    if (body?.detail) return String(body.detail);
  } catch {
    /* not JSON: it did not come from the API */
  }

  if (res.status === 504 || /^internal server error\s*$/i.test(raw)) {
    return (
      `The request to ${path} was cut off before it finished. Long estimations ` +
      `can outlast the proxy timeout — a shorter history, a longer roll-forward ` +
      `step or a smaller factor set will get there, and the run itself may well ` +
      `have completed, so trying again often returns it from the cache.`
    );
  }
  return `${res.status} ${res.statusText} — ${path}`;
}

// --- types ---------------------------------------------------------------

export type Verdict = "pass" | "warn" | "fail";

/**
 * Which stored series a factor request is about.
 *
 * "excess" is the factor itself — a log excess return over cash. "orth" is that
 * series after block-hierarchy orthogonalisation, which is what the regression
 * consumes. They are far apart: eq_us returns 12.9% a year at 17.3% volatility,
 * its residual -0.2% at 4.2%.
 */
export type Basis = "excess" | "orth";

export interface FactorQuery {
  start?: string;
  end?: string;
  basis?: Basis;
}

export interface SeriesStats {
  n_obs: number;
  mean_ann?: number;
  vol_ann?: number;
  sharpe?: number | null;
  skew?: number;
  excess_kurtosis?: number;
  max_drawdown?: number;
  total_compounded?: number;
}

/**
 * One factor measured on both panels over exactly the same days.
 *
 * `removed` is f - f~: the part of the raw factor the hierarchy attributes to the
 * blocks above it. For a level-0 factor there is nothing above it, `identical` is
 * true, and the two series are the same numbers.
 */
export interface FactorComparison {
  factor_id: string;
  name: string;
  block_id: string | null;
  hierarchy_level: number | null;
  identical: boolean;
  dates: string[];
  cumulative: { raw: number[]; orth: number[]; removed: number[] };
  stats: { raw: SeriesStats; orth: SeriesStats; removed: SeriesStats };
  alignment: {
    n_obs: number;
    first_date: string;
    last_date: string;
    correlation: number | null;
    variance_removed: number | null;
    tracking_vol_ann: number;
  };
  targets: Array<{
    factor_id: string; name: string; n_common: number;
    corr_raw: number | null; corr_orth: number | null;
  }>;
  implied_betas: Array<{
    factor_id: string; average_beta: number; r2: number | null; n_obs: number;
  }>;
}

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

/**
 * One pipeline run, with the parameterisation it ran under.
 *
 * `chips` and `describes` are composed server-side rather than in the browser:
 * unpacking a spec_id into "raw panel, 252-day window, monthly step" needs a join
 * the frontend has no business doing, and the assistant quotes the same strings.
 */
export interface RunDetail {
  run_id: string;
  job: string;
  mode: string | null;
  status: "running" | "succeeded" | "failed" | "partial";
  scope: Record<string, unknown>;
  spec: Record<string, unknown> | null;
  started_at: string;
  finished_at: string | null;
  duration_seconds: number | null;   // numeric on the wire; coerce before formatting
  rows_in: number;
  rows_out: number;
  n_failed: number;
  error: string | null;
  describes: string;
  outcome: string;
  chips: Array<{
    label: string; value: string; hint: string | null; emphasis: boolean;
  }>;
  item_counts: Record<string, number>;
  covers: { first_date: string | null; last_date: string | null } | null;
  items: Array<{
    item_key: string; status: string; rows_out: number | null;
    min_date: string | null; max_date: string | null; error: string | null;
  }>;
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
  run: (runId: string) => get<RunDetail>(`/api/meta/runs/${runId}`),
  factorSparklines: (basis: Basis = "excess") =>
    get<{ points: number; series: Record<string, number[]> }>(
      "/api/meta/factor-sparklines", { basis }),

  factorSeries: (id: string, p?: FactorQuery) =>
    get<{
      basis: Basis; dates: string[]; returns: number[];
      cumulative: number[]; compounded: number[];
    }>(`/api/factors/${id}/series`, p),
  factorStats: (id: string, p?: FactorQuery) =>
    get<any>(`/api/factors/${id}/stats`, p),
  factorRollingRisk: (id: string, windows = "21,63,252", p?: FactorQuery) =>
    get<{ dates: string[]; series: Record<string, (number | null)[]> }>(
      `/api/factors/${id}/rolling-risk`, { windows, ...p }),
  // bins is deliberately optional: omitting it lets the server pick the bin count
  // by the Freedman-Diaconis rule, which is the whole point of computing it there.
  factorHistogram: (id: string, p?: FactorQuery & { bins?: number }) =>
    get<any>(`/api/factors/${id}/histogram`, p),
  factorQQ: (id: string, p?: FactorQuery) => get<any>(`/api/factors/${id}/qq`, p),
  factorACF: (id: string, p?: FactorQuery & { lags?: number }) =>
    get<any>(`/api/factors/${id}/acf`, { lags: 30, ...p }),
  factorDiagnostics: (id: string) => get<any>(`/api/factors/${id}/diagnostics`),
  factorReference: (id: string) => get<any>(`/api/factors/${id}/reference-comparison`),
  // Deliberately takes no basis: the point of the endpoint is to serve both at once,
  // measured on the days where both exist.
  factorComparison: (id: string, p?: { start?: string; end?: string }) =>
    get<FactorComparison>(`/api/factors/${id}/comparison`, p),

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
