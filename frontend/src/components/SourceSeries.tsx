"use client";

/**
 * The stored series, in its own units, before anything was done to it.
 *
 * Every other panel on the Raw Explorer plots a return — which is already a
 * transformation, a log difference of the thing that was actually downloaded. This
 * plots what sits in the warehouse table: the adjusted close in dollars, the quoted
 * yield in percent, the exchange rate. It is the last step back before the data
 * stops being ours, and it is what someone doubting a factor wants to see.
 *
 * A factor built from several inputs gets a dropdown rather than several charts:
 * `arp_trend` reads thirteen instruments on wildly different scales, and thirteen
 * lines on one axis would be a smear.
 */

import { useEffect, useState } from "react";
import { Chart, ChartSkeleton, Panel } from "@/components/Chart";

export interface SourceRef {
  id: string;
  kind: "instrument" | "level" | "fx";
  label?: string | null;
  name?: string | null;
  source?: string | null;
}

interface SourceData {
  ref_kind: string;
  id: string;
  label: string;
  source: string;
  unit: string;
  quantity: string;
  table: string;
  transform?: string | null;
  is_total_return?: boolean | null;
  dates: string[];
  values: number[];
  unadjusted: (number | null)[] | null;
  n_obs: number;
  first_date: string;
  last_date: string;
}

export function SourceSeries({
  sources, accent, episodes, index,
}: {
  sources: SourceRef[];
  accent: string;
  episodes?: any;
  index?: number;
}) {
  const [pick, setPick] = useState<string>("");
  const [data, setData] = useState<SourceData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Reset to the first input whenever the series changes, so switching factors
  // cannot leave a chart of the previous one's input on screen.
  useEffect(() => {
    setPick(sources[0] ? key(sources[0]) : "");
  }, [sources]);

  useEffect(() => {
    if (!pick) { setData(null); return; }
    const [kind, ...rest] = pick.split("|");
    setData(null);
    setError(null);
    fetch(`/api/raw/source/${kind}/${encodeURIComponent(rest.join("|"))}`)
      .then((r) => (r.ok ? r.json()
                         : r.json().then((b) => Promise.reject(new Error(b.detail)))))
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)));
  }, [pick]);

  if (!sources.length) return null;

  return (
    <Panel
      index={index}
      title="Source data"
      caption={
        data ? (
          <>
            <b>{data.quantity}</b> in {data.unit}, exactly as stored in{" "}
            <span className="font-mono text-[11px]">{data.table}</span> from{" "}
            {data.source}.{" "}
            {/*
              How this becomes a return differs by kind, and saying "a log
              difference" for all three would be wrong for the one where it matters
              most: a yield is not log-differenced, it is priced through a synthetic
              par bond's duration and convexity, which is the whole reason the rates
              block is a return series at all.
            */}
            {data.ref_kind === "level"
              ? "A level, never used directly: the rates block prices it through a "
                + "synthetic par bond's duration and convexity, and the others "
                + "difference and standardise it."
              : "Everything else on this page is derived from this by a log "
                + "difference."}
            {data.unadjusted && (
              <>
                {" "}The unadjusted close is drawn beside it: the gap between the two
                is the accumulated distributions, and a series where they sit exactly
                on top of each other is a price-return index, which is why those are
                excluded from construction.
              </>
            )}
          </>
        ) : (
          "The stored series in its own units, before any transformation."
        )
      }
      actions={
        sources.length > 1 ? (
          <select
            className="field w-auto max-w-[240px]"
            value={pick}
            onChange={(e) => setPick(e.target.value)}
          >
            {sources.map((s) => (
              <option key={key(s)} value={key(s)}>
                {s.id}
                {s.kind !== "instrument" ? ` (${s.kind})` : ""}
              </option>
            ))}
          </select>
        ) : undefined
      }
    >
      {error && <p className="py-6 text-center text-[12px] text-fail">{error}</p>}
      {!data && !error && <ChartSkeleton height={260} />}
      {data && (
        <>
          <Chart
            height={260}
            episodes={episodes}
            data={[
              {
                x: data.dates, y: data.values, type: "scatter", mode: "lines",
                name: data.unadjusted ? "adjusted" : data.label,
                line: { color: accent, width: 1.4 },
                hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
              },
              ...(data.unadjusted
                ? [{
                    x: data.dates, y: data.unadjusted, type: "scatter" as const,
                    mode: "lines" as const, name: "unadjusted close",
                    line: { color: "#9AA0AE", width: 1, dash: "dot" as const },
                    hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
                  }]
                : []),
            ]}
            layout={{
              yaxis: { title: data.unit },
              margin: { l: 62, r: 14, t: 8, b: 34 },
              legend: { orientation: "h", y: -0.22 },
            }}
          />
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-muted">
            <span>
              Series at the provider{" "}
              <span className="font-mono text-navy">{data.label}</span>
            </span>
            <span>
              {data.first_date} to {data.last_date} ·{" "}
              {data.n_obs.toLocaleString()} observations
            </span>
            {data.transform && (
              <span>
                Made stationary by{" "}
                <span className="font-mono text-navy">{data.transform}</span>
              </span>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}

/**
 * Kind and id together, because ids are not unique across kinds and a bare id
 * would not say which table to read. `|` rather than `/`: level-series ids already
 * contain a colon and instrument ids contain dots and equals signs, but none of
 * them contains a pipe.
 */
function key(s: SourceRef): string {
  return `${s.kind}|${s.id}`;
}
