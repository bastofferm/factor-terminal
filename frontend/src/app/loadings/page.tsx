"use client";

/**
 * Loadings Lab: estimate factor betas for one security under a chosen window,
 * roll-forward step and estimator, and inspect how stable they are.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, PALETTE, Panel, Stat } from "@/components/Chart";
import { api, LoadingResult, num, pct, pval } from "@/lib/api";
import { SecuritySearch } from "@/components/SecuritySearch";
import { MetricCard, MetricStrip, alignFor } from "@/components/MetricCard";
import { METRICS } from "@/lib/metrics";
import { design } from "@/lib/verdict";
import { usePublishSnapshot } from "@/lib/chat-context";

const WINDOWS = [63, 126, 252, 504, 756];
const STEPS = [
  ["1d", "Daily"], ["1w", "Weekly (5d)"], ["1m", "Monthly (21d)"],
  ["3m", "Quarterly (63d)"], ["6m", "Semi-annual (126d)"],
] as const;

export default function LoadingsPage() {
  const [instrument, setInstrument] = useState("US:AAPL");
  const [windowDays, setWindowDays] = useState(252);
  const [step, setStep] = useState("1m");
  const [estimator, setEstimator] = useState("ols");
  const [weighting, setWeighting] = useState("equal");
  const [halflife, setHalflife] = useState(60);
  const [ridge, setRidge] = useState(0);
  const [dimson, setDimson] = useState(0);
  const [winsorize, setWinsorize] = useState(false);
  const [hacLags, setHacLags] = useState<string>("");
  const [orthogonalized, setOrthogonalized] = useState(true);

  const [result, setResult] = useState<LoadingResult | null>(null);
  const [stability, setStability] = useState<any>(null);
  const [shown, setShown] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);


  /**
   * Seconds on the button while a fit runs.
   *
   * A wide window rolled weekly is a thousand regressions, and the request does
   * not return until every one of them is written. A static "Estimating..." for
   * half a minute is indistinguishable from a hung page, and the honest fix for
   * that is to say how long it has been.
   */
  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const started = Date.now();
    const id = window.setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(id);
  }, [busy]);

  const run = () => {
    setBusy(true); setError(null);
    api.estimate({
      instrument_id: instrument,
      window_days: windowDays,
      step,
      estimator: ridge > 0 ? "ridge" : estimator,
      weighting,
      ewma_halflife: halflife,
      ridge_lambda: ridge,
      dimson_lags: dimson,
      winsorize,
      hac_lags: hacLags === "" ? null : Number(hacLags),
      orthogonalized,
    })
      .then((r) => {
        setResult(r);
        // Default to the six most significant loadings in the latest window;
        // forty lines at once is unreadable.
        setShown(r.latest.loadings.slice(0, 6).map((l) => l.factor_id));
        return api.stability(r.spec_id, instrument).catch(() => null);
      })
      .then(setStability)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  useEffect(run, []);

  const meta = result?.diagnostics ?? [];
  const latestMeta = meta[meta.length - 1] as any;
  // Read from the stored spec, not from the control: the result on screen may have
  // come from the cache under a spec estimated with the other setting.
  const estimatedOnOrth = (result?.spec as any)?.orthogonalized !== false;

  usePublishSnapshot(
    result && latestMeta
      ? {
          instrument_id: result.instrument_id,
          spec_id: result.spec_id,
          spec: result.spec,
          n_windows: meta.length,
          latest_window_end: result.latest.window_end,
          regression: {
            adj_r2: latestMeta.adj_r2, r2: latestMeta.r2,
            alpha_ann: latestMeta.alpha, t_alpha: latestMeta.t_alpha,
            resid_vol_ann: latestMeta.resid_vol_ann, n_obs: latestMeta.n_obs,
            durbin_watson: latestMeta.durbin_watson,
            condition_number: latestMeta.condition_number,
            max_vif: latestMeta.max_vif, f_p: latestMeta.f_p,
          },
          // Only the loadings worth discussing; forty rows of noise would crowd out
          // everything else in the prompt.
          top_loadings: result.latest.loadings
            .filter((l) => Math.abs(l.t_stat ?? 0) > 1.5)
            .slice(0, 12),
          stability: stability?.summary,
        }
      : null
  );

  const betaTraces = useMemo(() => {
    if (!result) return [];
    return shown.flatMap((f, i) => {
      const b = result.betas[f] ?? [];
      const se = result.standard_errors[f] ?? [];
      const colour = PALETTE[i % PALETTE.length];
      const upper = b.map((v, k) => (v === null || se[k] == null ? null : v + 1.96 * se[k]!));
      const lower = b.map((v, k) => (v === null || se[k] == null ? null : v - 1.96 * se[k]!));
      return [
        // 95% HAC band, drawn first so the line sits on top of it.
        { x: [...result.window_ends, ...[...result.window_ends].reverse()],
          y: [...upper, ...[...lower].reverse()],
          type: "scatter" as const, mode: "lines" as const, fill: "toself" as const,
          fillcolor: colour + "1A", line: { width: 0 },
          hoverinfo: "skip" as const, showlegend: false },
        { x: result.window_ends, y: b, type: "scatter" as const, mode: "lines" as const,
          name: f, line: { color: colour, width: 1.5 } },
      ];
    });
  }, [result, shown]);

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
      <aside className="space-y-3">
        <Panel title="Estimation">
          <div className="space-y-2.5">
            <div>
              <label className="label">Security</label>
              <SecuritySearch value={instrument} onChange={setInstrument} />
            </div>

            {/*
              Which factor panel the regression reads. This is the most consequential
              control on the page and the one whose effect is least obvious, so it
              sits above the window rather than among the estimator options — and it
              carries through: the covariance matrix used to score this spec's risk
              is built from the same panel, because betas from one panel with Sigma
              from the other make beta' Sigma beta meaningless.
            */}
            <div>
              <label className="label">Factor panel</label>
              <div className="mt-1 flex overflow-hidden rounded border border-line">
                {([
                  [true, "Orthogonalised",
                   "The block hierarchy's residuals. Well-conditioned design and a "
                   + "clean decomposition; each beta is incremental over the blocks "
                   + "above it."],
                  [false, "Raw",
                   "The factors before residualisation. Betas read directly - 1.02 to "
                   + "global equity means what it says - but the blocks overlap by "
                   + "construction, so watch the condition number and the VIFs."],
                ] as const).map(([v, label, tip]) => (
                  <button
                    key={String(v)}
                    title={tip}
                    onClick={() => setOrthogonalized(v)}
                    className={`flex-1 px-2 py-1 text-[11px] transition ${
                      orthogonalized === v
                        ? "bg-navy text-white"
                        : "bg-panel text-muted hover:bg-lineSoft"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <p className="mt-1 text-[10px] leading-tight text-muted">
                {orthogonalized
                  ? "Each beta is what the factor adds beyond the blocks above it."
                  : "Betas are in readable units, at the cost of collinear factors."}
              </p>
            </div>

            <div>
              <label className="label">Estimation window</label>
              <select className="field mt-1" value={windowDays}
                      onChange={(e) => setWindowDays(Number(e.target.value))}>
                {WINDOWS.map((w) => <option key={w} value={w}>{w} trading days</option>)}
              </select>
            </div>

            <div>
              <label className="label">Roll-forward step</label>
              <select className="field mt-1" value={step}
                      onChange={(e) => setStep(e.target.value)}>
                {STEPS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </div>

            <div>
              <label className="label">Estimator</label>
              <select className="field mt-1" value={estimator}
                      onChange={(e) => setEstimator(e.target.value)} disabled={ridge > 0}>
                <option value="ols">OLS</option>
                <option value="huber">Huber (robust)</option>
              </select>
            </div>

            <div>
              <label className="label">Ridge λ</label>
              <input type="number" step="0.1" min="0" className="field mt-1" value={ridge}
                     onChange={(e) => setRidge(Number(e.target.value))} />
              <p className="mt-1 text-[10px] leading-tight text-muted">
                Above zero, stabilises correlated factors and overrides the estimator.
              </p>
            </div>

            <div>
              <label className="label">Weighting</label>
              <select className="field mt-1" value={weighting}
                      onChange={(e) => setWeighting(e.target.value)}>
                <option value="equal">Equal</option>
                <option value="ewma">Exponential decay</option>
              </select>
              {weighting === "ewma" && (
                <input type="number" className="field mt-1" value={halflife}
                       onChange={(e) => setHalflife(Number(e.target.value))}
                       placeholder="half-life in days" />
              )}
            </div>

            <div>
              <label className="label">Dimson lags</label>
              <select className="field mt-1" value={dimson}
                      onChange={(e) => setDimson(Number(e.target.value))}>
                <option value={0}>None</option>
                <option value={1}>1 lag</option>
                <option value={2}>2 lags</option>
              </select>
              <p className="mt-1 text-[10px] leading-tight text-muted">
                For instruments pricing after the factor close; sums lagged betas.
              </p>
            </div>

            <div>
              <label className="label">HAC lags</label>
              <input className="field mt-1" value={hacLags} placeholder="automatic"
                     onChange={(e) => setHacLags(e.target.value)} />
            </div>

            <label className="flex items-center gap-2 text-[12px]">
              <input type="checkbox" checked={winsorize}
                     onChange={(e) => setWinsorize(e.target.checked)} />
              Winsorise returns at 1% / 99%
            </label>

            <button className="btn w-full" onClick={run} disabled={busy}>
              {busy
                ? elapsed > 2 ? `Estimating… ${elapsed}s` : "Estimating…"
                : "Estimate"}
            </button>
          </div>
        </Panel>

        {result && (
          <Panel title="Factors shown" caption="Ordered by significance in the latest window.">
            <div className="max-h-[320px] overflow-auto">
              {result.latest.loadings.map((l) => (
                <label key={l.factor_id}
                       className="flex items-center gap-2 py-0.5 text-[11px]">
                  <input
                    type="checkbox"
                    checked={shown.includes(l.factor_id)}
                    onChange={(e) =>
                      setShown((s) =>
                        e.target.checked
                          ? [...s, l.factor_id]
                          : s.filter((x) => x !== l.factor_id))}
                  />
                  <span className="flex-1 truncate font-mono">{l.factor_id}</span>
                  <span className={`font-mono tabular-nums ${
                    Math.abs(l.t_stat ?? 0) > 2 ? "font-semibold text-navy" : "text-muted"}`}>
                    {num(l.beta)}
                  </span>
                </label>
              ))}
            </div>
          </Panel>
        )}
      </aside>

      <div className="space-y-4">
        {error && (
          <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
            {error}
          </div>
        )}

        {latestMeta && (
          <MetricStrip>
            {/*
              One list rather than nine hand-placed cards, so every one of them
              gets its definition and its alignment without a per-card decision.
              maxVif is told which panel it is describing: on the raw panel a VIF
              of several hundred is the factor set, not a fault, and the risk
              pipeline no longer gates on it there - so neither does the colour.
            */}
            {([
              ["Windows", meta.length, undefined, "neutral", METRICS.windows],
              ["Adj. R²", num(latestMeta.adj_r2, 3), undefined,
               latestMeta.adj_r2 > 0.2 ? "good" : "neutral", METRICS.adjR2],
              ["Alpha (ann.)", pct(latestMeta.alpha),
               `t = ${num(latestMeta.t_alpha)}`, "neutral", METRICS.alpha],
              ["Residual vol", pct(latestMeta.resid_vol_ann), undefined, "neutral",
               METRICS.residualVol],
              ["Observations", latestMeta.n_obs?.toLocaleString("en-US"), undefined,
               "neutral", METRICS.observations],
              ["Durbin-Watson", num(latestMeta.durbin_watson),
               design.durbinWatson(latestMeta.durbin_watson).reason,
               design.durbinWatson(latestMeta.durbin_watson).tone,
               METRICS.durbinWatson],
              ["Condition no.", num(latestMeta.condition_number, 0),
               design.conditionNumber(latestMeta.condition_number).reason,
               design.conditionNumber(latestMeta.condition_number).tone,
               METRICS.conditionNumber],
              ["Max VIF", num(latestMeta.max_vif, 0),
               design.maxVif(latestMeta.max_vif, estimatedOnOrth).reason,
               design.maxVif(latestMeta.max_vif, estimatedOnOrth).tone,
               METRICS.maxVif],
              ["F p-value", pval(latestMeta.f_p), undefined, "neutral",
               METRICS.fPValue],
            ] as const).map(([label, value, sub, tone, metric], i) => (
              <MetricCard key={label} label={label} value={value} sub={sub}
                          tone={tone} metric={metric} align={alignFor(i)} />
            ))}
          </MetricStrip>
        )}

        {result && (
          <Panel
            title="Rolling factor loadings"
            caption={
              <>
                {/*
                  The panel is read back from the stored spec rather than from the
                  control, so what the caption claims is what actually produced the
                  numbers - including when the result came from the cache under a
                  spec estimated earlier.
                */}
                Estimated on the{" "}
                <b>
                  {(result.spec as any)?.orthogonalized === false
                    ? "raw"
                    : "orthogonalised"}
                </b>{" "}
                factor panel.{" "}
                {(result.spec as any)?.orthogonalized === false
                  ? "Each beta is the total exposure to that factor, and the blocks "
                    + "overlap by construction — read the condition number and the "
                    + "VIF column before trusting an individual coefficient."
                  : "Each beta is what that factor adds beyond the blocks above it "
                    + "in the hierarchy, not the total exposure to it."}{" "}
                Shaded bands are 95% intervals from Newey-West standard errors; a
                band that never excludes zero is a loading the data does not support.
              </>
            }
          >
            <Chart height={360} data={betaTraces as any}
                   layout={{ yaxis: { title: "beta" }, xaxis: { title: "window end" } }} />
          </Panel>
        )}

        <div className="grid gap-4 xl:grid-cols-2">
          {result && (
            <Panel
              title="Latest window"
              caption={`Window ending ${result.latest.window_end}. Bold where |t| exceeds 2.`}
            >
              <div className="max-h-[320px] overflow-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <th className="th">Factor</th><th className="th">Beta</th>
                      <th className="th">SE</th><th className="th">t</th>
                      <th className="th">p</th><th className="th">VIF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.latest.loadings.map((l) => {
                      const sig = Math.abs(l.t_stat ?? 0) > 2;
                      return (
                        <tr key={l.factor_id} className="border-t border-lineSoft">
                          <td className="cell font-sans">{l.factor_id}</td>
                          <td className={`cell ${sig ? "font-semibold text-navy" : ""}`}>
                            {num(l.beta, 3)}
                          </td>
                          <td className="cell text-muted">{num(l.se, 3)}</td>
                          <td className={`cell ${sig ? "font-semibold text-navy" : "text-muted"}`}>
                            {num(l.t_stat)}
                          </td>
                          <td className="cell text-muted">{pval(l.p_value)}</td>
                          <td className={`cell ${(l.vif ?? 0) > 10 ? "text-warn" : "text-muted"}`}>
                            {num(l.vif, 1)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}

          {meta.length > 0 && (
            <Panel
              title="Fit through time"
              caption="Adjusted R² and annualised residual volatility. A falling R² with stable betas usually means the instrument's risk moved into something the factor set does not carry."
            >
              <Chart
                height={280}
                data={[
                  { x: meta.map((m: any) => m.window_end), y: meta.map((m: any) => m.adj_r2),
                    type: "scatter", mode: "lines", name: "adj. R²",
                    line: { color: PALETTE[0], width: 1.4 } },
                  { x: meta.map((m: any) => m.window_end),
                    y: meta.map((m: any) => m.resid_vol_ann),
                    type: "scatter", mode: "lines", name: "residual vol", yaxis: "y2",
                    line: { color: PALETTE[1], width: 1.4 } },
                ]}
                layout={{
                  yaxis: { title: "adj. R²", range: [0, 1] },
                  yaxis2: { overlaying: "y", side: "right", title: "residual vol",
                            tickformat: ".0%", showgrid: false },
                }}
              />
            </Panel>
          )}
        </div>

        {stability?.series?.length > 0 && (
          <Panel
            title="Beta stability"
            caption={
              <>
                How far the whole loading vector moves between windows. The model
                treats a jump as a drift alert: either the security changed style, or
                the estimate is too noisy to act on. Windows are compared on the
                factors common to both — the factor set moves as coverage allows, and
                the overlap below says how wide the comparison basis was, since the L1
                sum falls mechanically when it narrows.
              </>
            }
            actions={
              <div className="flex gap-4">
                <Stat label="Median L1 shift" value={num(stability.summary.median_l1_shift)} />
                <Stat label="95th percentile" value={num(stability.summary.p95_l1_shift)} />
                <Stat label="Median vector ρ"
                      value={num(stability.summary.median_vector_correlation, 3)} />
                <Stat
                  label="Basis"
                  value={
                    stability.summary.min_overlap === stability.summary.max_overlap
                      ? `${stability.summary.max_overlap ?? "—"} factors`
                      : `${stability.summary.min_overlap}–${stability.summary.max_overlap}`
                  }
                  hint={
                    stability.summary.n_narrowed
                      ? `${stability.summary.n_narrowed} narrowed window(s)`
                      : "constant throughout"
                  }
                  tone={stability.summary.n_narrowed ? "neutral" : "good"}
                />
              </div>
            }
          >
            <Chart
              height={240}
              data={[
                { x: stability.series.map((s: any) => s.window_end),
                  y: stability.series.map((s: any) => s.beta_shift_l1),
                  type: "scatter", mode: "lines", name: "L1 change",
                  line: { color: PALETTE[3], width: 1.3 },
                  customdata: stability.series.map((s: any) => s.beta_overlap ?? "—"),
                  hovertemplate:
                    "L1 %{y:.2f} over %{customdata} shared factors<extra></extra>" },
                { x: stability.series.map((s: any) => s.window_end),
                  y: stability.series.map((s: any) => s.beta_corr_prev),
                  type: "scatter", mode: "lines", name: "correlation with previous",
                  yaxis: "y2", line: { color: PALETTE[4], width: 1.3 } },
              ]}
              layout={{
                yaxis: { title: "L1 change" },
                yaxis2: { overlaying: "y", side: "right", title: "vector ρ",
                          range: [-1, 1], showgrid: false },
              }}
            />
          </Panel>
        )}
      </div>
    </div>
  );
}
