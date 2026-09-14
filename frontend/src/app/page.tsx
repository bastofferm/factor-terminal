"use client";

/**
 * Data Health.
 *
 * The first thing an analyst should see, because every number elsewhere is only as
 * good as its inputs. Staleness is measured against the newest observation in the
 * panel rather than today, so a weekend does not make the model look broken.
 */

import { useEffect, useState } from "react";
import { Panel, Stat, VerdictBadge } from "@/components/Chart";
import { api } from "@/lib/api";
import { usePublishSnapshot } from "@/lib/chat-context";

export default function DataHealthPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.dataHealth().then(setData).catch((e) => setError(String(e.message ?? e)));
  }, []);

  // Published before the early returns below, because a hook cannot run
  // conditionally.
  usePublishSnapshot(
    data
      ? {
          as_of: data.as_of,
          instruments_total: (data.instruments ?? []).length,
          instruments_dead: (data.dead_instruments ?? []).length,
          dead: (data.dead_instruments ?? []).map((i: any) => ({
            id: i.instrument_id, last_obs: i.last_obs, days_behind: i.days_behind,
          })),
          level_series: (data.level_series ?? []).length,
          diagnostic_summary: data.diagnostic_summary,
          recent_runs: (data.runs ?? []).slice(0, 5).map((r: any) => ({
            job: r.job, status: r.status, rows_out: r.rows_out, n_failed: r.n_failed,
          })),
          open_failures: (data.failures ?? []).length,
        }
      : null
  );

  if (error)
    return (
      <div className="rounded border border-fail/30 bg-fail/5 p-4 text-[12px] text-fail">
        <b>Cannot reach the API.</b> {error}
        <div className="mt-2 font-mono text-[11px] text-muted">
          uvicorn backend.app.main:app --port 8100
        </div>
      </div>
    );
  if (!data) return <div className="p-8 text-center text-muted">Loading…</div>;

  const instruments: any[] = data.instruments ?? [];
  const levels: any[] = data.level_series ?? [];
  const dead: any[] = data.dead_instruments ?? [];
  const runs: any[] = data.runs ?? [];
  const failures: any[] = data.failures ?? [];

  const stale = instruments.filter((i) => i.is_live && (i.days_behind ?? 0) > 3);
  const priceReturn = instruments.filter((i) => !i.is_total_return && i.role !== "analysis");

  const verdicts = (data.diagnostic_summary ?? []).filter(
    (v: any) => v.series_type === "factor" && v.window_days === 252
  );
  const vCount = (k: string) =>
    verdicts.find((v: any) => v.verdict === k)?.n ?? 0;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 rounded border border-line bg-panel px-4 py-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Panel as of" value={data.as_of ?? "—"} />
        <Stat
          label="Live instruments"
          value={`${instruments.filter((i) => i.is_live).length}/${instruments.length}`}
          tone={dead.length ? "bad" : "good"}
        />
        <Stat label="Level series" value={levels.length} />
        <Stat
          label="Factors passing"
          value={vCount("pass")}
          hint="trailing 252d"
          tone="good"
        />
        {/* Amber for a warning count above zero, so the three tones on this row
            mean the same thing they mean in the battery itself. */}
        <Stat label="Warnings" value={vCount("warn")} tone={vCount("warn") ? "warn" : "good"} />
        <Stat label="Failing" value={vCount("fail")} tone={vCount("fail") ? "bad" : "good"} />
      </div>

      {dead.length > 0 && (
        <Panel
          title="Dead instruments"
          caption={
            <>
              No longer updating. Left in the registry with their reason rather than
              deleted — a silently vanished series is how a covariance matrix ends up
              built on data that stopped years ago.
            </>
          }
        >
          <Table
            rows={dead}
            cols={[
              ["instrument_id", "Instrument"],
              ["asset_class", "Class"],
              ["last_obs", "Last observation"],
              ["days_behind", "Days behind"],
              ["n_obs", "Observations"],
              ["notes", "Reason"],
            ]}
            tone={() => "fail"}
          />
        </Panel>
      )}

      {stale.length > 0 && (
        <Panel
          title="Lagging inputs"
          caption="More than three trading days behind the panel. Warehouse-sourced series (ECB, BOJ, MOF) lag by design; anything else here means an ingest did not run."
        >
          <Table
            rows={stale.slice(0, 20)}
            cols={[
              ["instrument_id", "Instrument"],
              ["asset_class", "Class"],
              ["last_obs", "Last observation"],
              ["days_behind", "Days behind"],
            ]}
            tone={() => "warn"}
          />
        </Panel>
      )}

      {priceReturn.length > 0 && (
        <Panel
          title="Price-return inputs"
          caption="Index levels without dividends. Excluded from equity factor construction on purpose — using them would bias every equity beta down by the dividend yield."
        >
          <div className="flex flex-wrap gap-1">
            {priceReturn.map((i) => (
              <span
                key={i.instrument_id}
                title={i.notes ?? ""}
                className="rounded border border-line bg-white px-1.5 py-0.5 font-mono text-[11px] text-muted"
              >
                {i.instrument_id}
              </span>
            ))}
          </div>
        </Panel>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Recent pipeline runs">
          <Table
            rows={runs.slice(0, 12)}
            cols={[
              ["job", "Job"],
              ["status", "Status"],
              ["started_at", "Started"],
              ["rows_out", "Rows"],
              ["n_failed", "Failed"],
            ]}
            format={{
              started_at: (v: string) => (v ? String(v).slice(0, 19).replace("T", " ") : "—"),
              rows_out: (v: number) => (v ?? 0).toLocaleString(),
            }}
            tone={(r) =>
              r.status === "succeeded" ? "pass" : r.status === "running" ? "" : "warn"
            }
          />
        </Panel>

        <Panel
          title="Outstanding failures"
          caption={failures.length ? undefined : "Nothing failing."}
        >
          {failures.length ? (
            <Table
              rows={failures.slice(0, 12)}
              cols={[
                ["job", "Job"],
                ["item_key", "Item"],
                ["error", "Error"],
              ]}
              tone={() => "fail"}
            />
          ) : (
            <div className="py-6 text-center text-[12px] text-pass">
              All pipeline items succeeded.
            </div>
          )}
        </Panel>
      </div>

      <Panel
        title="Level series coverage"
        caption="Yields, spreads and stress indices. These are I(1) levels and never enter the model directly; the transform column is the rule that makes each one stationary."
      >
        <Table
          rows={levels}
          cols={[
            ["series_id", "Series"],
            ["name", "Name"],
            ["category", "Category"],
            ["transform", "Transform"],
            ["last_obs", "Last"],
            ["n_obs", "Obs"],
          ]}
          maxHeight={280}
        />
      </Panel>
    </div>
  );
}

function Table({
  rows,
  cols,
  format = {},
  tone,
  maxHeight = 360,
}: {
  rows: any[];
  cols: [string, string][];
  format?: Record<string, (v: any) => string>;
  tone?: (row: any) => string;
  maxHeight?: number;
}) {
  return (
    <div className="overflow-auto" style={{ maxHeight }}>
      <table className="w-full border-collapse">
        <thead>
          <tr>
            {cols.map(([k, label]) => (
              <th key={k} className="th">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-lineSoft hover:bg-lineSoft/50">
              {cols.map(([k]) => {
                const raw = r[k];
                const text = format[k] ? format[k](raw) : raw ?? "—";
                const t = k === "status" || k === "instrument_id" ? tone?.(r) : "";
                return (
                  <td
                    key={k}
                    className={`cell ${t ? `text-${t}` : ""} ${
                      k === "notes" || k === "error" || k === "name"
                        ? "max-w-[420px] truncate font-sans"
                        : ""
                    }`}
                    title={typeof text === "string" ? text : undefined}
                  >
                    {String(text)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
