"use client";

/**
 * Operations: is the model current, and one button that makes it current.
 *
 * The nightly chain has always existed as a command. What did not exist was a way
 * to see from the browser whether last night's run happened, and a way to run it
 * now when it did not — which is the difference between a model someone maintains
 * and a model someone remembers to maintain.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Panel } from "@/components/Chart";
import { MetricCard, MetricStrip, alignFor } from "@/components/MetricCard";
import { METRICS } from "@/lib/metrics";
import { api, OpsStatus, RefreshRun } from "@/lib/api";
import { usePublishSnapshot } from "@/lib/chat-context";

/** While a refresh runs the page polls; idle it does not. */
const POLL_MS = 1500;

const fmtDate = (v: string | null | undefined) =>
  !v ? "—" : String(v).slice(0, 10);

const fmtWhen = (v: string | null | undefined) => {
  if (!v) return "—";
  const d = new Date(v);
  return isNaN(d.getTime())
    ? String(v)
    : d.toLocaleString(undefined, { month: "short", day: "numeric",
                                    hour: "2-digit", minute: "2-digit" });
};

const fmtSecs = (v: number | null | undefined) =>
  v === null || v === undefined || !isFinite(v)
    ? "—"
    : v < 90 ? `${v.toFixed(0)}s` : `${(v / 60).toFixed(1)}m`;

/** Age of a date in whole days, against the newest date anywhere in the panel. */
const ageDays = (d: string | null | undefined, asOf: string | undefined) => {
  if (!d || !asOf) return null;
  const a = new Date(d).getTime(), b = new Date(asOf).getTime();
  if (isNaN(a) || isNaN(b)) return null;
  return Math.round((b - a) / 86_400_000);
};

export default function OpsPage() {
  const [status, setStatus] = useState<OpsStatus | null>(null);
  const [run, setRun] = useState<RefreshRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const logRef = useRef<HTMLPreElement | null>(null);

  const load = useCallback(() => {
    api.opsStatus()
      .then(setStatus)
      .catch((e) => setError(String(e.message ?? e)));
    // Idle, this returns the chain with every stage pending — which is what the
    // panel should draw before anyone presses the button. A progress list that
    // only exists once a run starts cannot tell you what a run would do.
    api.opsRefresh().then(setRun).catch(() => {});
  }, []);

  useEffect(load, [load]);

  // Poll only while something is happening. A dashboard that hammers the API when
  // nothing is running is a dashboard nobody leaves open.
  useEffect(() => {
    if (run?.status !== "running") return;
    const id = window.setInterval(() => {
      api.opsRefresh()
        .then((r) => {
          setRun(r);
          if (r.status !== "running") load();   // refresh the freshness table once
        })
        .catch(() => {});
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [run?.status, load]);

  // Follow the log unless the reader has scrolled up to look at something.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [run?.log?.length]);

  const start = (full: boolean) => {
    setStarting(true); setError(null);
    api.opsStartRefresh(full)
      .then(setRun)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setStarting(false));
  };

  const cancel = () => {
    api.opsCancelRefresh()
      .then(() => api.opsRefresh().then(setRun))
      .catch((e) => setError(String(e.message ?? e)));
  };

  const running = run?.status === "running";
  const layers = status?.layers ?? [];
  // Only the layers the nightly chain owns. A security nobody has estimated since
  // July is not a stale model, and scoring it as one would make the headline number
  // permanently red for a reason nobody can act on.
  const oldest = layers.reduce<number | null>((worst, l) => {
    if (!l.nightly) return worst;
    const a = ageDays(l.last_date, status?.as_of);
    return a === null ? worst : worst === null ? a : Math.max(worst, a);
  }, null);

  usePublishSnapshot(
    status
      ? {
          page: "operations",
          as_of: status.as_of,
          layers: status.layers,
          last_job_runs: status.jobs,
          n_stale_instruments: status.stale_instruments.length,
          refresh: run
            ? { status: run.status, elapsed_seconds: run.elapsed_seconds,
                stages: run.stages.map((s) => ({ key: s.key, state: s.state })) }
            : null,
        }
      : null
  );

  return (
    <div className="space-y-4">
      <MetricStrip>
        <MetricCard
          label="Panel as of"
          metric={METRICS.panelAsOf}
          value={fmtDate(status?.as_of)}
          align={alignFor(0, 4)}
        />
        <MetricCard
          label="Widest gap"
          metric={METRICS.widestGap}
          value={oldest === null ? "—" : `${oldest}d`}
          tone={oldest === null ? "neutral"
                : oldest <= 1 ? "good" : oldest <= 5 ? "warn" : "bad"}
          align={alignFor(1, 4)}
        />
        <MetricCard
          label="Stale instruments"
          metric={METRICS.staleInstruments}
          value={status ? String(status.stale_instruments.length) : "—"}
          tone={!status ? "neutral"
                : status.stale_instruments.length === 0 ? "good" : "warn"}
          align={alignFor(2, 4)}
        />
        <MetricCard
          label="Last refresh"
          metric={METRICS.lastRefresh}
          value={fmtWhen(status?.jobs?.[0]?.finished_at ?? status?.jobs?.[0]?.started_at)}
          align={alignFor(3, 4)}
        />
      </MetricStrip>

      {error && (
        <div className="rounded border border-fail/40 bg-fail/5 px-3 py-2 text-[12px] text-fail">
          {error}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        {/* ---------------------------------------------------------------- */}
        <Panel
          title="Daily refresh"
          index={0}
          caption={
            <>
              The same chain the scheduler runs at 23:00 local, started here instead.
              Stages run in dependency order and the run <b>stops at the first fatal
              failure</b> — a factor built on half-refreshed inputs is worse than
              yesterday&rsquo;s factor, because it looks current and is not.
              <br />
              Equivalent on the command line:{" "}
              <code className="rounded bg-lineSoft px-1 py-px text-[10.5px]">
                python -m backend.pipeline.scheduler --once
              </code>
            </>
          }
          actions={
            <div className="flex gap-2">
              {running ? (
                <button onClick={cancel}
                        className="rounded border border-line bg-white px-2.5 py-1
                                   text-[12px] text-navy transition hover:border-navy2">
                  Stop
                </button>
              ) : (
                <button
                  onClick={() => start(true)}
                  disabled={starting}
                  title="Ignore the watermarks and re-fetch every history from the
                         start. Minutes rather than seconds; needed only after a
                         schema change or a vendor restatement."
                  className="rounded border border-line bg-white px-2.5 py-1
                             text-[12px] text-muted transition hover:border-navy2
                             hover:text-navy disabled:opacity-50">
                  Full rebuild
                </button>
              )}
              <button
                onClick={() => start(false)}
                disabled={running || starting}
                className="rounded bg-navy px-3 py-1 text-[12px] text-white
                           transition hover:bg-navy2 disabled:opacity-50"
              >
                {running
                  ? `Running · ${fmtSecs(run?.elapsed_seconds)}`
                  : starting ? "Starting…" : "Update all series"}
              </button>
            </div>
          }
        >
          <ol className="space-y-1.5">
            {(run?.stages ?? []).map((s, i) => (
              <li key={s.key}
                  className="flex items-start gap-2.5 rounded px-1.5 py-1
                             transition hover:bg-lineSoft/60"
                  title={s.detail}>
                <span className="mt-[5px] shrink-0">
                  <StageDot state={s.state} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-baseline justify-between gap-3">
                    <span className={`text-[12px] ${
                      s.state === "running" ? "font-semibold text-navy"
                      : s.state === "failed" ? "text-fail"
                      : s.state === "ok" ? "text-ink" : "text-muted"}`}>
                      {i + 1}. {s.label}
                    </span>
                    <span className="shrink-0 font-mono text-[10.5px] text-muted">
                      {s.note ?? fmtSecs(s.seconds)}
                    </span>
                  </span>
                  <span className="mt-0.5 block text-[10.5px] leading-snug text-muted">
                    {s.detail}
                  </span>
                </span>
              </li>
            ))}
          </ol>

          {run && run.run_id && (
            <div className="mt-3">
              <div className="mb-1 flex items-baseline justify-between">
                <span className="text-2xs font-semibold uppercase tracking-label text-muted">
                  Run log
                </span>
                <span className="font-mono text-[10.5px] text-muted">
                  {run.run_id} · {run.status}
                  {run.returncode !== null && run.returncode !== undefined
                    ? ` · exit ${run.returncode}` : ""}
                </span>
              </div>
              <pre ref={logRef}
                   className="max-h-56 overflow-auto rounded border border-line
                              bg-white px-2.5 py-2 font-mono text-[10.5px]
                              leading-relaxed text-ink">
{run.log.length ? run.log.join("\n") : "waiting for the first stage…"}
              </pre>
            </div>
          )}
        </Panel>

        {/* ---------------------------------------------------------------- */}
        <Panel
          title="Freshness by layer"
          index={1}
          caption={
            <>
              Each layer feeds the one below it. A gap that appears here appears in
              every number the model produces afterwards, which is why this table is
              read before the loadings are.
            </>
          }
        >
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-line text-left text-2xs uppercase
                             tracking-label text-muted">
                <th className="pb-1.5 font-semibold">Layer</th>
                <th className="pb-1.5 text-right font-semibold">Series</th>
                <th className="pb-1.5 text-right font-semibold">Rows</th>
                <th className="pb-1.5 text-right font-semibold">Newest</th>
                <th className="pb-1.5 text-right font-semibold">Age</th>
              </tr>
            </thead>
            <tbody>
              {layers.map((l) => {
                const age = ageDays(l.last_date, status?.as_of);
                return (
                  <tr key={l.layer} className="border-b border-lineSoft last:border-0">
                    <td className={`py-1.5 ${l.nightly ? "" : "text-muted"}`}
                        title={l.nightly
                          ? "Refreshed by the nightly chain."
                          : "Pulled the first time a security is estimated, not on "
                            + "a schedule. Its age says who has been using the "
                            + "Loadings Lab, not whether the model is current."}>
                      {l.layer}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-muted">
                      {l.n_series.toLocaleString()}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-muted">
                      {l.n_rows.toLocaleString()}
                    </td>
                    <td className="py-1.5 text-right tabular-nums">
                      {fmtDate(l.last_date)}
                    </td>
                    <td className={`py-1.5 text-right tabular-nums ${
                      age === null || !l.nightly ? "text-muted"
                      : age <= 1 ? "text-pass" : age <= 5 ? "text-warn" : "text-fail"}`}>
                      {age === null ? "—" : `${age}d`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <h3 className="mb-1.5 mt-4 text-2xs font-semibold uppercase tracking-label text-muted">
            Last run of each job
          </h3>
          <table className="w-full text-[12px]">
            <tbody>
              {(status?.jobs ?? []).map((j) => (
                <tr key={j.job} className="border-b border-lineSoft last:border-0">
                  <td className="py-1.5 font-mono text-[11px]">{j.job}</td>
                  <td className="py-1.5 text-muted">{j.mode}</td>
                  <td className="py-1.5 text-right tabular-nums text-muted">
                    {fmtWhen(j.finished_at ?? j.started_at)}
                  </td>
                  <td className="py-1.5 text-right tabular-nums text-muted">
                    {fmtSecs(j.duration_seconds)}
                  </td>
                  <td className="py-1.5 pl-2 text-right">
                    <span className={
                      j.status.startsWith("succe") ? "text-pass"
                      : j.status === "running" ? "text-navy" : "text-fail"}>
                      {j.status}
                      {j.n_failed ? ` · ${j.n_failed} failed` : ""}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      {(status?.stale_instruments?.length ?? 0) > 0 && (
        <Panel
          title="Live instruments falling behind"
          index={2}
          caption={
            <>
              More than five days behind the panel while still marked live. A dead
              series is not listed here — it is flagged and excluded, which is a
              decision. A live one drifting is a vendor, a credential or a rename,
              and it is the thing worth chasing after a refresh.
            </>
          }
        >
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-line text-left text-2xs uppercase
                             tracking-label text-muted">
                <th className="pb-1.5 font-semibold">Instrument</th>
                <th className="pb-1.5 font-semibold">Vendor ticker</th>
                <th className="pb-1.5 text-right font-semibold">Last observation</th>
                <th className="pb-1.5 text-right font-semibold">Behind</th>
              </tr>
            </thead>
            <tbody>
              {status!.stale_instruments.map((s) => (
                <tr key={s.instrument_id} className="border-b border-lineSoft last:border-0">
                  <td className="py-1.5 font-mono text-[11px]">{s.instrument_id}</td>
                  <td className="py-1.5 text-muted">{s.source_ticker}</td>
                  <td className="py-1.5 text-right tabular-nums">{fmtDate(s.last_obs)}</td>
                  <td className="py-1.5 text-right tabular-nums text-warn">
                    {s.days_behind}d
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}

      <Panel
        title="What a refresh does not do"
        index={3}
        caption="Scope, stated so that a green run is not read as more than it is."
      >
        <p className="text-[12px] leading-relaxed text-ink">
          The chain refreshes <b>inputs and factors</b>. It does not re-estimate
          loadings and it does not re-score risk, because both are per-specification
          and there are more specifications than anyone wants recomputed nightly.
          A loading view is re-estimated on demand from the Loadings Lab, which is a
          cache read when the specification has been seen before. Nor does it
          back-fill a history that a vendor has restated — that is what{" "}
          <b>Full rebuild</b> is for, and it ignores every watermark.
        </p>
      </Panel>
    </div>
  );
}

/** Stage state as a dot: pending hollow, running pulsing, then pass or fail. */
function StageDot({ state }: { state: string }) {
  if (state === "running") {
    return (
      <span className="relative inline-flex h-[7px] w-[7px]">
        <span className="absolute inline-flex h-full w-full animate-ping
                         rounded-full bg-navy opacity-60" />
        <span className="relative inline-flex h-[7px] w-[7px] rounded-full bg-navy" />
      </span>
    );
  }
  const tone =
    state === "ok" ? "bg-pass" : state === "failed" ? "bg-fail" : "bg-line";
  return <span className={`inline-block h-[7px] w-[7px] rounded-full ${tone}`} />;
}
