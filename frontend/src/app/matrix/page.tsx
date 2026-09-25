"use client";

/**
 * Covariance & PCA: the correlation structure of the factor panel, its eigen
 * spectrum, and how a chosen pair has co-moved through time.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, ChartSkeleton, PALETTE, Panel } from "@/components/Chart";
import { api, Basis, MatrixResult, num, pct } from "@/lib/api";
import { BlockAttribution } from "@/components/BlockAttribution";
import { MetricCard, MetricStrip, alignFor } from "@/components/MetricCard";
import { METRICS } from "@/lib/metrics";
import { usePublishSnapshot } from "@/lib/chat-context";

const METHODS = [
  ["blend", "Blend (EWMA + shrunk)"],
  ["ledoit_wolf", "Ledoit-Wolf shrinkage"],
  ["ewma", "EWMA"],
  ["sample", "Sample"],
] as const;

// Diverging scale, white at zero: the sign of a correlation is the first thing
// read off a heatmap, so it must not be encoded by lightness alone.
const DIVERGING: [number, string][] = [
  [0, "#8C3A2E"], [0.25, "#C8836F"], [0.5, "#F7F5F0"],
  [0.75, "#6E93B8"], [1, "#23456B"],
];

export default function MatrixPage() {
  const [method, setMethod] = useState<string>("blend");
  const [order, setOrder] = useState<string>("block");
  const [start, setStart] = useState<string>("2015-01-01");
  const [basis, setBasis] = useState<Basis>("orth");
  const [data, setData] = useState<MatrixResult | null>(null);
  const [pair, setPair] = useState<[string, string] | null>(null);
  const [rolling, setRolling] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = () => {
    setBusy(true); setError(null);
    api.matrix({ start: start || null, method, order, basis })
      .then((m) => {
        setData(m);
        if (!pair && m.names.length >= 2) setPair([m.names[0], m.names[1]]);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  useEffect(load, [method, order, start, basis]);

  useEffect(() => {
    if (!pair) return;
    api.rollingCorrelation({ factor_a: pair[0], factor_b: pair[1], window: 252, basis })
      .then(setRolling)
      .catch(() => setRolling(null));
  }, [pair, basis]);

  const heatmap = useMemo(() => {
    if (!data) return null;
    // Plotly draws row 0 at the bottom; reverse so the matrix reads top-left down,
    // which is how anyone expects to see a correlation matrix. Both axes carry the
    // same factors in the same order — the reversal is which way the rows are
    // drawn, not a different ordering — so the diagonal runs corner to corner and
    // a block is a square on it.
    return {
      z: [...data.correlation].reverse(),
      x: data.names,
      y: [...data.names].reverse(),
    };
  }, [data]);

  /**
   * One box per block, drawn on the diagonal.
   *
   * The blocks arrive aligned with `names`, so a block is a contiguous run once
   * the ordering is "by block" — and only then. Under cluster or registry
   * ordering a block is scattered, and boxing a run that happens to be adjacent
   * would draw a grouping that is not there, so the boxes are withheld instead.
   *
   * Coordinates are category indices. The y axis is reversed, so a run covering
   * columns a..b covers rows N-1-b .. N-1-a.
   */
  const blockBoxes = useMemo(() => {
    if (!data?.blocks || order !== "block") return [];
    const n = data.names.length;
    const runs: { block: string; a: number; b: number }[] = [];
    data.blocks.forEach((b: string | null, i: number) => {
      const last = runs[runs.length - 1];
      if (last && last.block === b) last.b = i;
      else runs.push({ block: b ?? "", a: i, b: i });
    });
    return runs.map((r) => ({
      type: "rect" as const,
      xref: "x" as const, yref: "y" as const,
      x0: r.a - 0.5, x1: r.b + 0.5,
      y0: n - 1 - r.b - 0.5, y1: n - 1 - r.a + 0.5,
      line: { color: "#2A2F3A", width: 2 },
      fillcolor: "rgba(0,0,0,0)",
      layer: "above" as const,
    }));
  }, [data, order]);

  const d = data?.diagnostics;

  usePublishSnapshot(
    data
      ? {
          method: data.method,
          // Which panel these numbers describe. Without it the assistant cannot tell
          // a raw correlation of 0.71 from an orthogonalised one of 0.04.
          basis: (data as any).basis ?? basis,
          n_factors: data.names.length,
          factors: data.names,
          n_obs: data.n_obs,
          sample_start: data.start,
          sample_end: data.end,
          diagnostics: data.diagnostics,
          rolling_pair: rolling
            ? { a: rolling.factor_a, b: rolling.factor_b,
                full_sample: rolling.full_sample, window: rolling.window }
            : undefined,
        }
      : null
  );

  return (
    <div className="space-y-4">
      <Panel
        title="Estimator"
        caption={
          <>
            {basis === "orth" ? (
              <>
                <b>Σ on the orthogonalised factors, which is the model&rsquo;s own
                covariance.</b> It has to match the panel the loadings were estimated
                on: portfolio risk is β&prime;Σβ, so Σ must be the covariance of the
                same series the βs refer to. Switching to raw shows the structure
                <i> before</i> the hierarchy removes the overlap — which is the
                argument for orthogonalising, but not a Σ to pair with orthogonalised
                βs.
              </>
            ) : (
              <>
                <b>Σ on the raw factors, before the hierarchy removes the
                overlap.</b> The blocks share exposure by construction, so expect a
                denser off-diagonal, a worse condition number and a larger PC1 share
                than the orthogonalised panel shows. Pair this Σ only with loadings
                estimated on the raw panel — the Loadings Lab has the matching
                switch.
              </>
            )}
            <br />
            Shrinkage matters here: with 39 factors on a rolling window the sample
            covariance is badly conditioned, and an optimiser handed an indefinite
            matrix will happily build a portfolio out of the negative eigenvalue.
          </>
        }
      >
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="label">Factor panel</label>
            <div className="mt-1 flex overflow-hidden rounded border border-line">
              {([["orth", "Orthogonalised"], ["excess", "Raw"]] as const).map(
                ([v, label]) => (
                  <button
                    key={v}
                    onClick={() => setBasis(v)}
                    className={`px-3 py-[5px] text-[11px] transition ${
                      basis === v
                        ? "bg-navy text-white"
                        : "bg-panel text-muted hover:bg-lineSoft"
                    }`}
                  >
                    {label}
                  </button>
                )
              )}
            </div>
          </div>
          <div>
            <label className="label">Method</label>
            <select className="field mt-1 w-56" value={method}
                    onChange={(e) => setMethod(e.target.value)}>
              {METHODS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Ordering</label>
            <select className="field mt-1 w-44" value={order}
                    onChange={(e) => setOrder(e.target.value)}>
              <option value="block">By block</option>
              <option value="cluster">Hierarchical clustering</option>
              <option value="none">As declared</option>
            </select>
          </div>
          <div>
            <label className="label">Start</label>
            <input type="date" className="field mt-1 w-40" value={start}
                   onChange={(e) => setStart(e.target.value)} />
          </div>
          <button className="btn" onClick={load} disabled={busy}>
            {busy ? "Estimating…" : "Re-estimate"}
          </button>
        </div>
      </Panel>

      {error && (
        <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
          {error}
        </div>
      )}

      {d && data && (
        <MetricStrip>
          {([
            ["Factors",
             data.n_available && data.n_available !== data.names.length
               ? `${data.names.length} of ${data.n_available}`
               : data.names.length,
             "in the estimated matrix", "neutral", undefined],
            ["Observations", data.n_obs.toLocaleString("en-US"),
             "days with every factor present", "neutral", undefined],
            ["Span", `${data.start} → ${data.end}`, "sample estimated on",
             "neutral", undefined],
            ["Shrinkage δ", num(d.shrink_intensity, 3),
             "0 = sample, 1 = target", "neutral", METRICS.shrinkage],
            // Not design.conditionNumber: that gate is about a security's
            // regression design and fires at 200. Four figures here is ordinary
            // for 39 correlated factors, and borrowing the other verdict would
            // print "excluded from risk scoring" under a number that is not.
            ["Condition number", num(d.condition_number, 0),
             "largest over smallest eigenvalue",
             d.condition_number > 5000 ? "warn" : "neutral",
             METRICS.covConditionNumber],
            ["Min eigenvalue", d.min_eigenvalue.toExponential(2),
             d.min_eigenvalue < 0 ? "indefinite" : "no negative directions",
             d.min_eigenvalue < 0 ? "bad" : "good", METRICS.minEigenvalue],
            ["PSD", d.is_psd ? "yes" : "no",
             d.psd_repaired ? "repaired by eigenvalue clipping" : "as estimated",
             d.is_psd ? "good" : "bad", METRICS.psd],
            ["PC1 share", pct(d.pc1_share),
             "one dominant direction means one factor", "neutral",
             METRICS.pc1Share],
          ] as const).map(([label, value, sub, tone, metric], i) => (
            <MetricCard key={label} label={label} value={value} sub={sub}
                        tone={tone} metric={metric} align={alignFor(i)} />
          ))}
        </MetricStrip>
      )}

      {/* A factor missing from the matrix is not a detail. Everything downstream
          treats this as the model's covariance, so which factors are in it has to
          be on the screen, with the trade that would include them. */}
      {data && (data.dropped?.length ?? 0) > 0 && (
        <div className="rounded border border-line bg-panel px-3 py-2 text-[11px]
                        leading-relaxed text-ink">
          <b>{data.dropped!.length === 1 ? "One factor is" : `${data.dropped!.length} factors are`}</b>{" "}
          not in this matrix. A factor needs data across 90% of the window, or it
          would truncate the sample for every pair it appears in:
          <ul className="mt-1 space-y-0.5">
            {data.dropped!.map((f) => (
              <li key={f.factor_id} className="font-mono text-[10.5px]">
                {f.factor_id} — {(f.coverage * 100).toFixed(0)}% covered, starts{" "}
                {f.first_date}
              </li>
            ))}
          </ul>
          {data.earliest_start_for_all && (
            <div className="mt-1.5">
              Every factor has data from{" "}
              <b>{data.earliest_start_for_all}</b>, which buys completeness with a
              shorter sample.{" "}
              <button
                onClick={() => setStart(data.earliest_start_for_all!)}
                className="rounded border border-line bg-white px-2 py-0.5
                           text-[10.5px] text-navy transition hover:border-navy2">
                Estimate from {data.earliest_start_for_all}
              </button>
            </div>
          )}
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel
          title="Correlation matrix"
          caption={
            <>
              Click a cell to plot that pair&rsquo;s rolling correlation below. Blue
              is positive, red negative, white zero. Both axes carry the same{" "}
              {data?.names.length ?? 0} factors in the same order, so the cells are
              square and the diagonal runs corner to corner.
              {order === "block"
                ? " Heavy outlines mark the blocks; everything inside one is a within-block correlation."
                : " Block outlines are drawn only under block ordering — under this ordering a block is not a contiguous run, and boxing one would draw a grouping that is not there."}
            </>
          }
        >
          {heatmap && data && (
            <Chart
              height={data.names.length * 17 + 160}
              data={[{
                type: "heatmap", z: heatmap.z, x: heatmap.x, y: heatmap.y,
                colorscale: DIVERGING, zmid: 0, zmin: -1, zmax: 1,
                hovertemplate: "%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
                colorbar: { thickness: 10, len: 0.6, tickfont: { size: 9 } },
              }]}
              layout={{
                margin: { l: 130, r: 10, t: 10, b: 130 },
                // dtick 1 forces every factor to be labelled. Without it Plotly
                // drops every other row label when the box is short, which reads
                // as a matrix with more columns than rows.
                xaxis: { tickfont: { size: 9 }, tickangle: -55, showgrid: false,
                         dtick: 1 },
                // scaleanchor holds one unit on y to one unit on x, so the cells
                // stay square whatever width the panel ends up with. Without
                // `constrain: domain`: that constrains the axis by shrinking its
                // domain instead of widening its range, which collapses the plot
                // into a strip.
                yaxis: { tickfont: { size: 9 }, showgrid: false, dtick: 1,
                         scaleanchor: "x", scaleratio: 1 },
                shapes: blockBoxes,
                hovermode: "closest", showlegend: false,
              }}
            />
          )}
        </Panel>

        <Panel
          title="The blocks on their own"
          caption={
            <>
              The same correlations, one panel per block, each on the identical
              &minus;1 to +1 scale as the matrix beside it. The big matrix shows
              where a block sits relative to everything else; these show what is
              going on <i>inside</i> it, which the full grid compresses into a
              square too small to read. A block that is uniformly blue moves as
              one thing and a single factor can stand for it; a pale one is a
              filing category whose members happen not to move together.
            </>
          }
        >
          {!data && <ChartSkeleton height={520} />}
          {data && <BlockGrid data={data} />}
        </Panel>
      </div>

      <Panel
        title="Rolling pairwise correlation"
        caption="A stable full-sample correlation can hide a pair that was uncorrelated for a decade and went to 0.8 in a crisis. That is the behaviour that makes a single correlation matrix dangerous in exactly the periods it matters most."
        actions={
          data && pair && (
            <div className="flex gap-2">
              <select className="field w-44" value={pair[0]}
                      onChange={(e) => setPair([e.target.value, pair[1]])}>
                {data.names.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <select className="field w-44" value={pair[1]}
                      onChange={(e) => setPair([pair[0], e.target.value])}>
                {data.names.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          )
        }
      >
        {rolling && (
          <>
            <div className="mb-2 text-[12px] text-muted">
              Full-sample correlation{" "}
              <b className="font-mono text-navy">{num(rolling.full_sample, 3)}</b>
            </div>
            <Chart
              height={260}
              data={[
                { x: rolling.dates, y: rolling.correlation, type: "scatter",
                  mode: "lines", name: "252d rolling ρ",
                  line: { color: PALETTE[0], width: 1.3 } },
                { x: [rolling.dates[0], rolling.dates[rolling.dates.length - 1]],
                  y: [rolling.full_sample, rolling.full_sample], type: "scatter",
                  mode: "lines", name: "full sample",
                  line: { color: "#9AA0AE", width: 1, dash: "dash" } },
              ]}
              layout={{ yaxis: { title: "correlation", range: [-1, 1] } }}
            />
          </>
        )}
      </Panel>

      {/* Everything above describes the factors. What follows describes what
          they cost, which needs an exposure vector and therefore a security. */}
      <div className="border-t border-line pt-4">
        <h2 className="mb-1 text-2xs font-semibold uppercase tracking-label text-muted">
          Block risk budgeting
        </h2>
        <p className="mb-3 max-w-4xl text-[11px] leading-relaxed text-muted">
          The panels above are statements about the factor panel itself. These
          ask what it costs: which block of the matrix drives a security&rsquo;s
          risk, how much the blocks spill into one another, whether that changes
          when markets are loud, and which factor inside a block is doing the
          work. The first three need an exposure to be about anything, so they
          are measured through one representative security per asset class.
        </p>
        <BlockAttribution start={start} method={method} basis={basis} />
      </div>
    </div>
  );
}


/**
 * One small heatmap per block, on the same scale as the full matrix.
 *
 * The big grid answers where a block sits against everything else. It cannot
 * answer what is happening inside one: nine factors squeezed into a seven-pixel
 * square is a colour, not a reading. These are the diagonal sub-matrices pulled
 * out and given room.
 *
 * Nothing is refetched. The sub-matrices are slices of the correlation already
 * on screen, so the two panels cannot disagree — and a block drawn here is
 * exactly the block outlined there.
 */
function BlockGrid({ data }: { data: MatrixResult }) {
  const blocks = useMemo(() => {
    const byBlock = new Map<string, number[]>();
    data.blocks.forEach((b, i) => {
      const key = b ?? "unassigned";
      if (!byBlock.has(key)) byBlock.set(key, []);
      byBlock.get(key)!.push(i);
    });
    return [...byBlock.entries()].map(([id, idx]) => {
      // Mean off-diagonal correlation: how much the block moves as one thing.
      // A single-factor block has no pair to average, and reporting 1.0 would
      // claim a relationship with no second party to it.
      let sum = 0, n = 0;
      for (let a = 0; a < idx.length; a++)
        for (let b2 = a + 1; b2 < idx.length; b2++) {
          sum += data.correlation[idx[a]][idx[b2]];
          n += 1;
        }
      return {
        id,
        names: idx.map((i) => data.names[i]),
        z: idx.map((r) => idx.map((c) => data.correlation[r][c])),
        mean: n ? sum / n : null,
      };
    }).sort((a, b) => b.names.length - a.names.length);
  }, [data]);

  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-3">
      {blocks.map((b) => (
        <div key={b.id}>
          <div className="flex items-baseline justify-between gap-1">
            <span className="truncate text-[10.5px] font-semibold text-navy">
              {b.id}
            </span>
            <span className="shrink-0 font-mono text-[9.5px] text-muted"
                  title="Mean correlation between the distinct pairs in this block">
              {b.names.length}f{b.mean === null ? "" : ` · ρ̄ ${b.mean.toFixed(2)}`}
            </span>
          </div>
          <Chart
            height={132}
            data={[{
              type: "heatmap",
              z: [...b.z].reverse(),
              x: b.names,
              y: [...b.names].reverse(),
              colorscale: DIVERGING, zmid: 0, zmin: -1, zmax: 1,
              showscale: false,
              hovertemplate: "%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
            }]}
            layout={{
              margin: { l: 4, r: 4, t: 2, b: 4 },
              xaxis: { showticklabels: false, showgrid: false, dtick: 1 },
              yaxis: { showticklabels: false, showgrid: false, dtick: 1,
                       scaleanchor: "x", scaleratio: 1 },
              hovermode: "closest", showlegend: false,
            }}
          />
        </div>
      ))}
    </div>
  );
}
