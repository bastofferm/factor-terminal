"use client";

/**
 * Relative Value & Timing: which class is cheap against which, and whether the
 * next months argue for acting on it now.
 *
 * The Ten-Year Forecasts page says what each class should earn over a decade
 * and shows no one-year forecast, because valuation says next to nothing about a
 * year. This page sits beside it at the shorter horizon without contradicting
 * it: its scores are signals, not forecasts. Each view is a pair with four
 * standardised signals - value, carry, trend, macro - a composite score, the
 * stance and tilt that follow from it, and the ten-year anchor for comparison.
 *
 * The scores are computed and stored by the pipeline (backend/pipeline/
 * relative_value.py). The DeepSeek summary reads a stored run; it decides
 * nothing, and every number in it has been checked against the run's data.
 */

import { useEffect, useMemo, useState } from "react";
import { MetricCard, MetricStrip } from "@/components/MetricCard";
import { ChartSkeleton, Panel } from "@/components/Chart";
import {
  RelativeValueRun, RvDefaults, RvJob, RvSummary, RvSummaryJob, api, pct,
} from "@/lib/api";
import { usePublishSnapshot } from "@/lib/chat-context";
import { RV_METRICS as M } from "@/lib/relativeValueMetrics";
import { LicenceNotes } from "@/components/returns/shared";
import {
  ClassesTable, CreditPanel, MacroPanel, Scorecard, SummaryPanel, fz,
} from "@/components/returns/RelativeValue";

function differing(a: Record<string, any>, b: Record<string, any>): string[] {
  return Object.keys(a).filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b?.[k]));
}

export default function RelativeValuePage() {
  const [defaults, setDefaults] = useState<RvDefaults | null>(null);
  const [result, setResult] = useState<RelativeValueRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, number>>({});
  const [job, setJob] = useState<RvJob | null>(null);
  const running = job?.status === "running";

  const [summary, setSummary] = useState<RvSummary | null>(null);
  const [sumJob, setSumJob] = useState<RvSummaryJob | null>(null);
  const [sumError, setSumError] = useState<string | null>(null);
  const [unverified, setUnverified] = useState<string[]>([]);

  useEffect(() => {
    Promise.all([api.rvDefaults(), api.rvLatest()])
      .then(([d, r]) => {
        setDefaults(d);
        setResult(r);
        setSummary(r?.summary ?? null);
        if (r) {
          const o: Record<string, number> = {};
          for (const k of differing(r.assumptions, d.assumptions)) o[k] = r.assumptions[k];
          setOverrides(o);
        }
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  const effective = useMemo(() => ({ ...(defaults?.assumptions ?? {}), ...overrides }),
                            [defaults, overrides]);
  const stale = !!result && differing(effective, result.assumptions).length > 0;

  const run = async () => {
    setError(null);
    try {
      const { job_id } = await api.rvRun(Object.keys(overrides).length ? overrides : null);
      setJob({ job_id, status: "running", elapsed: 0, messages: [], result: null, error: null });
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  };

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setTimeout(async () => {
      try {
        const j = await api.rvGet(job.job_id);
        setJob(j);
        if (j.status === "succeeded" && j.result) {
          setResult(j.result);
          setSummary(null);
          setSumError(null);
          setUnverified([]);
        }
        if (j.status === "failed") setError(j.error);
      } catch (e: any) {
        setError(String(e.message ?? e));
        setJob(null);
      }
    }, 1000);
    return () => clearTimeout(t);
  }, [job]);

  const summarise = async (force: boolean) => {
    setSumError(null);
    setUnverified([]);
    try {
      const r = await api.rvSummarise(result?.run_id ?? null, force);
      if (r.summary) setSummary(r.summary);
      else if (r.job_id) setSumJob({ job_id: r.job_id, status: "running", elapsed: 0, messages: [],
                                     result: null, error: null });
    } catch (e: any) {
      setSumError(String(e.message ?? e));
    }
  };

  useEffect(() => {
    if (!sumJob || sumJob.status !== "running") return;
    const t = setTimeout(async () => {
      try {
        const j = await api.rvSummaryGet(sumJob.job_id);
        setSumJob(j);
        if (j.status === "succeeded" && j.result && "content" in j.result) setSummary(j.result);
        if (j.status === "failed") {
          setSumError(j.error);
          setUnverified(j.result && "unverified" in j.result ? j.result.unverified : []);
        }
      } catch (e: any) {
        setSumError(String(e.message ?? e));
        setSumJob(null);
      }
    }, 1500);
    return () => clearTimeout(t);
  }, [sumJob]);

  const reads = useMemo(() => Object.fromEntries((summary?.content.views ?? []).map((v) => [v.id, v.read])),
                        [summary]);

  usePublishSnapshot(result ? {
    run_id: result.run_id, data_as_of: result.data_as_of,
    ten_year_anchor_run: result.forecast_run?.run_id ?? null,
    regime: result.macro.regime,
    growth_z: result.macro.growth.z, inflation_z: result.macro.inflation.z,
    views: result.views.map((v) => ({
      id: v.id, label: v.label, legs: `${v.long}/${v.short ?? "cash"}`, stance: v.stance,
      score: v.score, change_1m: v.change, band: v.act ? "act" : "hold",
      value_z: v.signals.value?.z ?? null, carry_z: v.signals.carry?.z ?? null,
      trend_z: v.signals.trend?.z ?? null, macro_z: v.signals.macro?.z ?? null,
      tilt: v.tilt, anchor_excess: v.anchor?.pair_excess ?? null, anchor_per_risk: v.anchor?.per_risk ?? null,
    })),
    classes: result.classes.map((c) => ({ fund: c.fund, per_risk: c.per_risk, trend_z: c.trend_z })),
    credit: result.credit ? { hy_excess_after_loss: result.credit.hy.excess,
                              hy_cushion: result.credit.hy.cushion, ig_excess_after_loss: result.credit.ig.excess }
      : null,
    tracking_error: result.sizing.te_total,
    summary_headline: summary?.content.headline ?? null,
  } : null);

  const strongest = result?.views.filter((v) => v.score !== null)
    .sort((a, b) => Math.abs(b.score!) - Math.abs(a.score!))[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-[12px] font-medium text-navy">Relative value &amp; timing</span>
        <span className="text-[11px] text-muted">
          Signals over one to twelve months beside the ten-year anchor
          {result && <> · data to {result.data_as_of} · run {result.run_id} on {result.created_at?.slice(0, 10)}
            {result.forecast_run && <> · anchor: forecast run {result.forecast_run.run_id} of {result.forecast_run.as_of}</>}</>}
          {!result && !loading && " · no run yet"}
        </span>
        <button className="btn ml-auto min-w-[150px]" onClick={run} disabled={running || !defaults}>
          {running ? `Scoring… ${Math.round(job!.elapsed)}s` : stale ? "Re-run with changes" : "Run scores"}
        </button>
      </div>

      {error && <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">{error}</div>}
      {running && job!.messages.length > 0 && (
        <div className="rounded border border-line bg-white p-2 text-[11px] text-muted">
          {job!.messages[job!.messages.length - 1]}</div>
      )}
      {stale && !running && (
        <div className="rounded border border-warn/40 bg-warn/5 p-2 text-[11.5px] text-warn">
          The assumptions have changed since this run; the numbers below are the last run&apos;s until it
          is run again.</div>
      )}

      {result && (
        <MetricStrip>
          <MetricCard label="Macro regime" value={result.macro.regime ?? "—"} metric={M.regime}
                      sub={`growth ${fz(result.macro.growth.z)} · inflation ${fz(result.macro.inflation.z)}`} />
          <MetricCard label="Strongest view" value={strongest ? fz(strongest.score) : "—"} metric={M.score}
                      sub={strongest ? `${strongest.label} · ${strongest.stance}` : undefined} />
          <MetricCard label="Views to act on" value={`${result.views.filter((v) => v.act).length} of ${result.views.length}`}
                      metric={M.change} sub={`band ${fz(result.assumptions.no_trade)} a month`} />
          <MetricCard label="Tracking error" value={pct(result.sizing.te_total, 2)} metric={M.tilt} align="right"
                      sub={`budget ${pct(result.sizing.te_budget, 1)}${result.sizing.scale < 1
                        ? ` · tilts scaled ×${result.sizing.scale.toFixed(2)}` : ""}`} />
          <MetricCard label="VIX" value={result.macro.vix ? result.macro.vix.level.toFixed(1) : "—"} align="right"
                      sub={result.macro.vix ? `z ${fz(result.macro.vix.z)} · ${result.macro.vix.as_of}` : undefined} />
        </MetricStrip>
      )}

      {defaults && (
        <AssumptionsEditor effective={effective} defaults={defaults.assumptions}
                           setOverride={(k, v) => setOverrides((o) => {
                             const next = { ...o, [k]: v };
                             if (defaults.assumptions[k] === v) delete next[k];
                             return next;
                           })}
                           reset={() => setOverrides({})} changed={Object.keys(overrides)} disabled={running} />
      )}

      {loading && <Panel title="Relative value"><ChartSkeleton height={320} /></Panel>}
      {!loading && !result && !running && (
        <Panel title="No scores yet">
          <p className="text-[12px]">Nothing has been run. Run the scores to rebuild every view&apos;s signals
            month by month; it takes a few seconds. The ten-year anchor comes from the latest Ten-Year
            Forecasts run.</p>
        </Panel>
      )}

      {result && (
        <div className={"space-y-4 " + (stale ? "opacity-60 transition-opacity" : "")}>
          <SummaryPanel summary={summary} configured={!!defaults?.summary.configured}
                        model={defaults?.summary.model ?? ""} runId={result.run_id}
                        busy={sumJob?.status === "running" ? { elapsed: sumJob.elapsed } : null}
                        onSummarise={summarise} error={sumError} unverified={unverified} />
          <Scorecard run={result} reads={reads} />
          <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
            {result.credit ? <CreditPanel credit={result.credit} /> : <div />}
            <MacroPanel macro={result.macro} />
          </div>
          <ClassesTable rows={result.classes} />
          <Panel title="Read with care" caption="Generated with the run from what the data could and could not supply.">
            <ul className="list-disc space-y-0.5 pl-4 text-[11.5px] leading-snug text-muted">
              {result.caveats.map((c, i) => <li key={i}>{c}</li>)}
            </ul>
          </Panel>
          <LicenceNotes licences={result.licences} />
        </div>
      )}
    </div>
  );
}

/** The judgements, with their defaults. Only what differs from a default is sent. */
function AssumptionsEditor({
  effective, defaults, setOverride, reset, changed, disabled,
}: {
  effective: Record<string, any>;
  defaults: Record<string, any>;
  setOverride: (key: string, value: number) => void;
  reset: () => void;
  changed: string[];
  disabled: boolean;
}) {
  // percent fields are edited in percent and stored as decimals
  const field = (key: string, label: string, opts: { step?: number; pctField?: boolean; hint?: string } = {}) => {
    const { step = 0.05, pctField = false, hint } = opts;
    const v = effective[key];
    return (
      <div key={key}>
        <label className={"label " + (changed.includes(key) ? "text-warn" : "")}>{label}</label>
        <input type="number" step={step} className="field mt-1 w-20 text-right" disabled={disabled}
               value={v === undefined || v === null ? "" : pctField ? Math.round(v * 10000) / 100 : v}
               onChange={(e) => {
                 const n = Number(e.target.value);
                 if (e.target.value !== "" && isFinite(n)) setOverride(key, pctField ? n / 100 : n);
               }} />
        {hint && <div className="mt-0.5 text-[10px] text-muted">{hint}</div>}
      </div>
    );
  };
  return (
    <details className="rounded border border-line bg-panel px-4 py-2.5">
      <summary className="cursor-pointer text-[12px] text-navy">
        Assumptions{changed.length > 0 && <span className="ml-2 text-warn">{changed.length} changed</span>}
        <span className="ml-2 text-muted">- signal weights, stance thresholds, sizing and credit inputs</span>
      </summary>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-3">
        {field("w_value", "Weight value")}
        {field("w_carry", "Weight carry")}
        {field("w_trend", "Weight trend")}
        {field("w_macro", "Weight macro")}
        {field("neutral", "Neutral below", { hint: "|score|" })}
        {field("strong", "Full stance from")}
        {field("no_trade", "No-trade band", { hint: "score change a month" })}
        {field("min_history_months", "Min. history", { step: 12, hint: "months for a z" })}
        {field("k", "Tilt per score, pp", { pctField: true, step: 0.5 })}
        {field("a_max", "Max tilt, pp", { pctField: true, step: 0.5 })}
        {field("te_budget", "TE budget, %", { pctField: true, step: 0.25 })}
        {field("hy_pd", "HY default rate, %", { pctField: true, step: 0.1 })}
        {field("hy_pd_low", "HY lower PD, %", { pctField: true, step: 0.1 })}
        {field("ig_pd", "IG default rate, %", { pctField: true, step: 0.05 })}
        {field("lgd", "LGD, %", { pctField: true, step: 5 })}
        {field("hy_spread_duration", "HY spread dur.", { step: 0.25 })}
        {field("ig_spread_duration", "IG spread dur.", { step: 0.25 })}
      </div>
      <div className="mt-3 flex items-center gap-3 text-[11px] text-muted">
        <button className="btn-ghost" onClick={reset} disabled={disabled || !changed.length}>Reset to defaults</button>
        <span>Defaults: equal signal weights, neutral below {defaults.neutral}, full stance from {defaults.strong},
          band {defaults.no_trade}, {pct(defaults.k, 0)} per unit of score up to {pct(defaults.a_max, 0)}, TE budget{" "}
          {pct(defaults.te_budget, 1)}; HY default rate {pct(defaults.hy_pd, 1)} (Moody&apos;s baseline), LGD{" "}
          {pct(defaults.lgd, 0)}.</span>
      </div>
    </details>
  );
}
