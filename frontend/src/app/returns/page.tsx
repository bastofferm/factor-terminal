"use client";

/**
 * Return Forecasts: what each asset class can be expected to return over the
 * next ten years, built from blocks that are observable today, and what that
 * implies for a portfolio.
 *
 * The rest of the terminal forecasts risk, and says plainly that it does not
 * forecast returns. This page is the exception, at the one horizon where there
 * is something to say: over a year, valuation explains next to nothing of the
 * following return; over a decade, a good deal. The method follows the
 * published capital market assumptions (Research Affiliates, J.P. Morgan,
 * Invesco): income + growth + inflation + valuation change, each block with its
 * source and date, an expected value with a band rather than a point, and the
 * historical average kept beside it as the rear-view mirror it is.
 *
 * Four views, all mounted and hidden rather than unmounted, so a run or a
 * loaded track record survives a switch. `#implied`, `#perspectives` and
 * `#track-record` open the other views directly.
 *
 * Every judgement is an assumption with a documented default, editable here and
 * stored with each run. The weights between the return vectors are the user's:
 * a ten-year forecast leaves no held-out span for the model agent to choose
 * them on.
 */

import { useEffect, useMemo, useState } from "react";
import { MetricCard, MetricStrip } from "@/components/MetricCard";
import { Panel, ChartSkeleton } from "@/components/Chart";
import { ReturnDefaults, ReturnForecast, ReturnJob, api, num, pct } from "@/lib/api";
import { RETURN_METRICS as RM } from "@/lib/returnMetrics";
import { usePublishSnapshot } from "@/lib/chat-context";
import { BuildingBlocks } from "@/components/returns/BuildingBlocks";
import { ImpliedVectors } from "@/components/returns/ImpliedVectors";
import { Perspectives } from "@/components/returns/Perspectives";
import { TrackRecord } from "@/components/returns/TrackRecord";

type Mode = "blocks" | "implied" | "perspectives" | "track";

const MODES: readonly (readonly [Mode, string, string])[] = [
  ["blocks", "Building blocks", ""],
  ["implied", "Implied vectors", "#implied"],
  ["perspectives", "Perspectives", "#perspectives"],
  ["track", "Track record", "#track-record"],
];

/**
 * JSON with object keys sorted. A stored run comes back from a JSONB column,
 * which reorders the keys of every object, so a plain JSON.stringify would call
 * {conviction, concern} and {concern, conviction} different assumptions.
 */
function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (v && typeof v === "object") {
    return `{${Object.keys(v as object).sort()
      .map((k) => `${JSON.stringify(k)}:${canonical((v as any)[k])}`).join(",")}}`;
  }
  return JSON.stringify(v ?? null);
}

/** Keys whose value differs between two assumption sets. */
function differing(a: Record<string, any>, b: Record<string, any>): string[] {
  return Object.keys(a).filter((k) => canonical(a[k]) !== canonical(b[k]));
}

export default function ReturnsPage() {
  const [mode, setMode] = useState<Mode>("blocks");
  // Empty until the hash is read, so a deep link does not first mount the
  // default view hidden.
  const [seen, setSeen] = useState<Set<Mode>>(() => new Set<Mode>());
  const choose = (m: Mode) => {
    setMode(m);
    setSeen((s) => new Set(s).add(m));
    const hash = MODES.find(([k]) => k === m)?.[2] || " ";
    window.history.replaceState(null, "", hash);
    // Charts in a view shown before re-measure against a window that may have
    // been resized while they were hidden.
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  };
  useEffect(() => {
    const m = MODES.find(([, , h]) => h && h === window.location.hash);
    choose(m ? m[0] : "blocks");
  }, []);

  const [defaults, setDefaults] = useState<ReturnDefaults | null>(null);
  const [result, setResult] = useState<ReturnForecast | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [job, setJob] = useState<ReturnJob | null>(null);
  const running = job?.status === "running";

  useEffect(() => {
    Promise.all([api.returnsDefaults(), api.returnsLatest()])
      .then(([d, r]) => {
        setDefaults(d);
        setResult(r);
        // Open on what the latest run assumed, so the editor agrees with the numbers.
        if (r) {
          const o: Record<string, unknown> = {};
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
      const { job_id } = await api.returnsRun(Object.keys(overrides).length ? overrides : null);
      setJob({ job_id, status: "running", elapsed: 0, messages: [], result: null, error: null });
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  };

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setTimeout(async () => {
      try {
        const j = await api.returnsGet(job.job_id);
        setJob(j);
        if (j.status === "succeeded" && j.result) setResult(j.result);
        if (j.status === "failed") setError(j.error);
      } catch (e: any) {
        setError(String(e.message ?? e));
        setJob(null);
      }
    }, 1200);
    return () => clearTimeout(t);
  }, [job]);

  const setOverride = (key: string, value: unknown) => {
    setOverrides((o) => {
      const next = { ...o, [key]: value };
      if (defaults && canonical(defaults.assumptions[key]) === canonical(value)) delete next[key];
      return next;
    });
  };

  // The assistant sees the forecasts and their bands, never the licensed
  // valuation series behind them.
  usePublishSnapshot(result ? {
    as_of: result.as_of,
    horizon_years: result.horizon,
    cash: result.cash.total,
    lambda: result.lambda.value,
    classes: result.blocks.map((b) => ({
      key: b.key, expected: b.total, income: b.income, real_growth: b.real_growth,
      inflation: b.inflation, valuation: b.valuation, other: b.other + b.buyback,
      band_p5: b.band.p5, band_p95: b.band.p95, history: b.history.full?.geo ?? null,
      vol: b.vol,
    })),
    portfolios: result.portfolios.map((p) => ({ id: p.id, name: p.name, band_p50: p.band.p50,
                                                 band_p5: p.band.p5, band_p95: p.band.p95 })),
    view: mode,
  } : null);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div role="tablist" aria-label="Return forecast views"
             className="inline-flex rounded border border-line bg-white p-0.5">
          {MODES.map(([m, label]) => (
            <button key={m} role="tab" aria-selected={mode === m} onClick={() => choose(m)}
                    className={"rounded px-3 py-1 text-[12px] transition " + (mode === m
                      ? "bg-navy font-medium text-white"
                      : "text-navy hover:bg-lineSoft/60")}>
              {label}
            </button>
          ))}
        </div>
        <span className="text-[11px] text-muted">
          Ten-year horizon · USD · {result ? <>run {result.run_id} on {result.created_at?.slice(0, 10) ?? result.as_of}</>
            : "no run yet"}
        </span>
        <button className="btn ml-auto min-w-[170px]" onClick={run} disabled={running || !defaults}>
          {running ? `Running… ${Math.round(job!.elapsed)}s` : stale ? "Re-run with changes" : "Run forecast"}
        </button>
      </div>

      {error && <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">{error}</div>}
      {running && job!.messages.length > 0 && (
        <div className="rounded border border-line bg-white p-2 text-[11px] text-muted">
          {job!.messages[job!.messages.length - 1]}</div>
      )}
      {stale && !running && (
        <div className="rounded border border-warn/40 bg-warn/5 p-2 text-[11.5px] text-warn">
          The assumptions have changed since this run; the numbers below are the last run&apos;s until it is
          run again.</div>
      )}

      {result && (
        <MetricStrip>
          <MetricCard label="Cash, next ten years" value={pct(result.cash.total, 2)}
                      sub={result.cash.sources[0]?.source} />
          <MetricCard label="Expected inflation" value={pct(result.meta.inflation.value, 2)}
                      metric={RM.inflation} sub={`${result.meta.inflation.source} · ${result.meta.inflation.as_of}`} />
          <MetricCard label="Real growth per share" value={pct(result.meta.real_growth.used, 2)}
                      metric={RM.real_growth}
                      sub={`± ${pct(result.meta.real_growth.se_used, 2)} · Shiller ${result.meta.real_growth.from ?? ""}-${String(result.meta.real_growth.to ?? "").slice(0, 4)}`} />
          <MetricCard label="Risk aversion λ" value={num(result.lambda.value, 2)} metric={RM.lambda}
                      sub="calibrates the implied vectors" />
          <MetricCard label="US CAPE" align="right"
                      value={num(result.blocks.find((b) => b.key === "us_equity")?.check?.cape, 1)}
                      metric={RM.valuation}
                      sub={`norm ${num(Object.values(result.blocks.find((b) => b.key === "us_equity")?.check?.targets ?? {})[0] as number, 1)}`} />
        </MetricStrip>
      )}

      {defaults && (
        <AssumptionsEditor effective={effective} defaults={defaults.assumptions}
                           setOverride={setOverride} reset={() => setOverrides({})}
                           changed={Object.keys(overrides)} disabled={running} />
      )}

      {loading && <Panel title="Return forecasts"><ChartSkeleton height={360} /></Panel>}
      {!loading && !result && !running && (
        <Panel title="No forecast yet">
          <p className="text-[12px]">Nothing has been run. Run the forecast to build every class&apos;s
            blocks, the strategic covariance and the band; it takes about ten seconds.</p>
        </Panel>
      )}

      {/* A view mounts the first time it is shown and stays mounted after: a
          Plotly chart drawn inside a hidden element measures a width of zero and
          keeps it. */}
      <div className={stale ? "opacity-60 transition-opacity" : ""}>
        <div hidden={mode !== "blocks"}>
          {result && seen.has("blocks") && <BuildingBlocks result={result} />}</div>
        <div hidden={mode !== "implied"}>
          {result && seen.has("implied") && <ImpliedVectors result={result} />}</div>
        <div hidden={mode !== "perspectives"}>
          {result && seen.has("perspectives") &&
            <Perspectives result={result} overrides={effective} setOverride={setOverride}
                          running={running} stale={stale} onRun={run} />}
        </div>
      </div>
      <div hidden={mode !== "track"}>
        {seen.has("track") && <TrackRecord active={mode === "track"} />}
      </div>
    </div>
  );
}

/**
 * The judgements, with their defaults. Only what differs from a default is
 * sent; "Reset" returns to the documented defaults.
 */
function AssumptionsEditor({
  effective, defaults, setOverride, reset, changed, disabled,
}: {
  effective: Record<string, any>;
  defaults: Record<string, any>;
  setOverride: (key: string, value: unknown) => void;
  reset: () => void;
  changed: string[];
  disabled: boolean;
}) {
  const pctField = (key: string, label: string, step = 0.1, nullable = false, hint?: string) => (
    <Field label={label} changed={changed.includes(key)} hint={hint}>
      <input type="number" step={step} className="field mt-1 w-24 text-right" disabled={disabled}
             placeholder={nullable ? "measured" : undefined}
             value={effective[key] === null || effective[key] === undefined ? ""
               : Math.round(effective[key] * 10000) / 100}
             onChange={(e) => setOverride(key, e.target.value === "" && nullable ? null
               : Number(e.target.value) / 100)} />
    </Field>
  );
  const sample = (key: string, label: string, choices: number[]) => (
    <Field label={label} changed={changed.includes(key)}>
      <select className="field mt-1 w-28" disabled={disabled} value={effective[key]?.[0]}
              onChange={(e) => {
                const first = Number(e.target.value);
                setOverride(key, [first, ...choices.filter((c) => c !== first)]);
              }}>
        {choices.map((c) => <option key={c} value={c}>since {c}</option>)}
      </select>
    </Field>
  );
  return (
    <details className="rounded border border-line bg-panel px-4 py-2.5">
      <summary className="cursor-pointer text-[12px] text-navy">
        Assumptions{changed.length > 0 && <span className="ml-2 text-warn">{changed.length} changed</span>}
        <span className="ml-2 text-muted">- every judgement the forecast makes, with its default</span>
      </summary>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-3">
        <Field label="Reversion (years)" changed={changed.includes("reversion_years")}
               hint="share of the gap closed in 10y = 10 / T">
          <input type="number" min={5} max={40} step={1} className="field mt-1 w-20 text-right"
                 disabled={disabled} value={effective.reversion_years}
                 onChange={(e) => setOverride("reversion_years", Number(e.target.value))} />
        </Field>
        {sample("us_cape_samples", "US CAPE norm", [1881, 1950, 1990])}
        {sample("intl_cape_samples", "EU/JP CAPE norm", [1982, 1995, 2005])}
        <Field label="EU/JP norm as" changed={changed.includes("intl_cape_how")}>
          <select className="field mt-1 w-28" disabled={disabled} value={effective.intl_cape_how}
                  onChange={(e) => setOverride("intl_cape_how", e.target.value)}>
            <option value="median_log">median</option>
            <option value="mean_log">geometric mean</option>
          </select>
        </Field>
        <Field label="DY valuation (EM, REITs)" changed={changed.includes("dy_valuation_weight")}
               hint="1 = full drift, 0 = none">
          <input type="number" min={0} max={1} step={0.25} className="field mt-1 w-20 text-right"
                 disabled={disabled} value={effective.dy_valuation_weight}
                 onChange={(e) => setOverride("dy_valuation_weight", Number(e.target.value))} />
        </Field>
        {pctField("net_buyback", "Net buybacks %")}
        {pctField("real_growth", "Real growth %", 0.1, true)}
        {pctField("inflation", "Inflation %", 0.1, true, "blank = breakeven")}
        {pctField("reit_real_growth", "REIT real growth %")}
        {pctField("yield_change", "Yield change by year 10, pp", 0.25)}
        {pctField("agg_credit_share", "AGG credit share %", 5)}
        {pctField("ig_default_loss", "IG default loss %", 0.05)}
        {pctField("em_default_loss", "EM default loss %", 0.1)}
        {pctField("market_equity", "Market: equity share %", 5)}
      </div>
      <div className="mt-3 flex items-center gap-3 text-[11px] text-muted">
        <button className="btn-ghost" onClick={reset} disabled={disabled || !changed.length}>
          Reset to defaults</button>
        <span>Defaults: reversion {defaults.reversion_years} years, US norm since {defaults.us_cape_samples?.[0]},
          EU/JP median since {defaults.intl_cape_samples?.[0]}, no net buybacks, real growth and inflation
          measured, AGG credit {pct(defaults.agg_credit_share, 0)} with {pct(defaults.ig_default_loss, 2)} loss,
          EM loss {pct(defaults.em_default_loss, 1)}.</span>
      </div>
    </details>
  );
}

function Field({ label, changed, hint, children }: {
  label: string; changed: boolean; hint?: string; children: React.ReactNode;
}) {
  return (
    <div>
      <label className={"label " + (changed ? "text-warn" : "")}>{label}</label>
      <div>{children}</div>
      {hint && <div className="mt-0.5 text-[10px] text-muted">{hint}</div>}
    </div>
  );
}
