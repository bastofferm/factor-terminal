"use client";

/**
 * The pipeline-run inspector: a list you pick from, and what the pick did.
 *
 * Twelve rows of "succeeded" say nothing on their own. The interesting fact about
 * four consecutive `run_risk` rows is that they scored different specs — one on the
 * raw factor panel, one on the orthogonalised one — and nothing in the list shows
 * it, because that choice lives behind a spec_id in `scope`. Selecting a row and
 * reading its parameters beside it is the only way to see it.
 *
 * The prose and the chips are composed server-side (`backend/app/runs.py`). Turning
 * a spec_id into "raw panel · 252d window · monthly step" needs a join the browser
 * has no business doing, and the assistant grounds on the same strings.
 */

import { useEffect, useState } from "react";
import { Panel, Stat } from "@/components/Chart";
import { api, RunDetail as Run } from "@/lib/api";

export function RunTable({
  runs, selected, onSelect,
}: {
  runs: any[];
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="overflow-auto" style={{ maxHeight: 360 }}>
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="th">Job</th>
            <th className="th">Status</th>
            <th className="th">Started</th>
            <th className="th">Rows</th>
            <th className="th">Failed</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => {
            const active = r.run_id === selected;
            const tone = statusTone(r.status);
            return (
              <tr
                key={r.run_id}
                onClick={() => onSelect(r.run_id)}
                // A row that drives the panel beside it is a control, so it takes
                // focus and answers the keyboard rather than the mouse alone.
                tabIndex={0}
                role="button"
                aria-pressed={active}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(r.run_id);
                  }
                }}
                className={
                  "cursor-pointer border-t border-l-[3px] border-lineSoft outline-none transition " +
                  (active
                    ? "border-l-navy bg-navy/[0.06]"
                    : "border-l-transparent hover:bg-lineSoft/50 focus:bg-lineSoft/50")
                }
              >
                <td className={"cell " + (active ? "font-semibold text-navy" : "")}>
                  {r.job}
                </td>
                <td className={"cell " + tone}>{r.status}</td>
                <td className="cell text-muted">{stamp(r.started_at)}</td>
                <td className="cell tabular-nums">
                  {(r.rows_out ?? 0).toLocaleString("en-US")}
                </td>
                <td className={"cell tabular-nums " + (r.n_failed ? "text-fail" : "")}>
                  {r.n_failed ?? 0}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RunDetail({
  runId, failures,
}: {
  runId: string | null;
  failures: any[];
}) {
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetched per selection rather than shipped with the list: resolving every run's
  // spec would mean a join per row for twelve rows nobody has opened, and the item
  // list can run to sixty entries each.
  useEffect(() => {
    if (!runId) {
      setRun(null);
      return;
    }
    setRun(null);
    setError(null);
    api.run(runId).then(setRun).catch((e) => setError(String(e.message ?? e)));
  }, [runId]);

  if (!runId) return <Shell>Select a run on the left.</Shell>;
  if (error) return <Shell tone="text-fail">{error}</Shell>;
  if (!run) return <Shell>Loading…</Shell>;

  const failed = run.items.filter((i) => i.status === "failed");

  return (
    <Panel
      title={run.job}
      caption={run.describes || undefined}
      actions={
        <span
          className={
            "font-mono text-[11px] font-semibold uppercase " + statusTone(run.status)
          }
        >
          {run.status}
        </span>
      }
    >
      <p className="text-[12px] text-navy">{run.outcome}</p>

      {run.chips.length > 0 && (
        <div className="mt-3">
          <div className="label mb-1.5">Parameters</div>
          <div className="flex flex-wrap gap-1.5">
            {run.chips.map((c) => (
              <span
                key={c.label}
                title={c.hint ?? undefined}
                className={
                  "inline-flex items-baseline gap-1.5 rounded border px-2 py-1 " +
                  "text-[11px] leading-none " +
                  (c.emphasis
                    ? "border-navy/30 bg-navy/[0.06] text-navy"
                    : "border-line bg-white text-muted")
                }
              >
                <span className="text-2xs uppercase tracking-label opacity-70">
                  {c.label}
                </span>
                <span className="font-mono font-semibold">{c.value}</span>
              </span>
            ))}
          </div>
          <p className="mt-1.5 text-[10px] leading-tight text-muted">
            Shaded chips are the choices that change the answer rather than the
            runtime. Hover any of them for what it does.
          </p>
        </div>
      )}

      <div className="mt-3 grid grid-cols-2 gap-3 border-t border-lineSoft pt-3 sm:grid-cols-4">
        <Stat label="Started" value={stamp(run.started_at)} />
        <Stat label="Duration" value={duration(run.duration_seconds)} />
        <Stat label="Rows written" value={(run.rows_out ?? 0).toLocaleString("en-US")} />
        <Stat
          label="Items"
          value={
            Object.entries(run.item_counts)
              .map(([k, n]) => n + " " + k)
              .join(", ") || "—"
          }
          tone={run.n_failed ? "bad" : "neutral"}
        />
      </div>

      {run.covers?.first_date && (
        <p className="mt-2 text-[11px] text-muted">
          Covers <span className="font-mono text-navy">{run.covers.first_date}</span>
          {" to "}
          <span className="font-mono text-navy">{run.covers.last_date}</span>, across
          the items it touched.
        </p>
      )}

      {run.error && (
        <div className="mt-3 rounded border border-fail/30 bg-fail/5 px-3 py-2 text-[11px] text-fail">
          {run.error}
        </div>
      )}

      {failed.length > 0 && (
        <div className="mt-3">
          <div className="label mb-1 text-fail">Failed in this run</div>
          <ul className="space-y-1">
            {failed.map((i) => (
              <li key={i.item_key} className="text-[11px]">
                <span className="font-mono text-navy">{i.item_key}</span>
                {i.error && <span className="text-fail"> — {i.error}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {run.items.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-[11px] text-muted hover:text-navy">
            {run.items.length} item{run.items.length === 1 ? "" : "s"}, largest first
          </summary>
          <div className="mt-1.5 max-h-[180px] overflow-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="th">Item</th>
                  <th className="th">Status</th>
                  <th className="th">Rows</th>
                  <th className="th">Covers</th>
                </tr>
              </thead>
              <tbody>
                {run.items.map((i) => (
                  <tr key={i.item_key} className="border-t border-lineSoft">
                    <td className="cell max-w-[200px] truncate" title={i.item_key}>
                      {i.item_key}
                    </td>
                    <td className={"cell " + statusTone(i.status)}>{i.status}</td>
                    <td className="cell tabular-nums">
                      {(i.rows_out ?? 0).toLocaleString("en-US")}
                    </td>
                    <td className="cell text-muted">
                      {i.min_date ? i.min_date + " to " + i.max_date : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}

      {/* An item that failed three runs ago is invisible from the run you happen to
          be looking at, so open failures still need somewhere to surface. */}
      {failures.length > 0 && failed.length === 0 && (
        <div className="mt-3 rounded border border-warn/30 bg-warn/5 px-3 py-2 text-[11px] text-warn">
          {failures.length} item{failures.length === 1 ? "" : "s"} still failing from
          earlier runs: {failures.slice(0, 4).map((f) => f.item_key).join(", ")}
          {failures.length > 4 ? " and " + (failures.length - 4) + " more" : ""}.
        </div>
      )}
    </Panel>
  );
}

function Shell({ children, tone }: { children: React.ReactNode; tone?: string }) {
  return (
    <Panel title="Run detail">
      <div className={"py-10 text-center text-[12px] " + (tone ?? "text-muted")}>
        {children}
      </div>
    </Panel>
  );
}

function statusTone(status: string): string {
  if (status === "succeeded") return "text-pass";
  if (status === "failed") return "text-fail";
  if (status === "partial") return "text-warn";
  return "text-muted";
}

function stamp(value: string | null): string {
  return value ? String(value).slice(0, 19).replace("T", " ") : "—";
}

/**
 * Coerced rather than trusted. The API types this as a number and PostgreSQL
 * returns `EXTRACT(EPOCH ...)` as numeric, which asyncpg renders as a JSON *string*
 * — so an uncoerced `.toFixed()` threw and took the whole page down with it. The
 * cast is fixed server-side; this stays because a formatter should not be able to
 * blank a page over a type it did not expect.
 */
function duration(seconds: number | string | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const s = Number(seconds);
  if (!isFinite(s)) return "—";
  if (s < 1) return "<1s";
  if (s < 90) return s.toFixed(0) + "s";
  if (s < 5400) return (s / 60).toFixed(0) + "m";
  return (s / 3600).toFixed(1) + "h";
}
