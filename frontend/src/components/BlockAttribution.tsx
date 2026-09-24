"use client";

/**
 * Block risk budgeting: which block of the covariance drives a security's risk,
 * and which factor drives the block.
 *
 * The section is in two halves and the split is the thing that makes it
 * readable. Everything above "Inside the block" is a statement about a held
 * exposure and changes with the security in the selector. The PC1 share, the
 * centrality and the eigenvector weights are properties of the covariance and
 * do not move at all - a block can be tightly coupled internally and still cost
 * a particular holding nothing.
 *
 * The securities are one per asset class and none of them is an input to any
 * factor. That constraint is doing real work: a proxy that sits inside the
 * factor construction loads about one on its own factor by definition, and its
 * block split would restate the recipe rather than measure anything.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, ChartSkeleton, PALETTE, Panel } from "@/components/Chart";
import { api, Basis, BlockResult, num, pct } from "@/lib/api";

const DIVERGING: [number, string][] = [
  [0, "#8C3A2E"], [0.25, "#C8836F"], [0.5, "#F7F5F0"],
  [0.75, "#6E93B8"], [1, "#23456B"],
];

/** Signed share: the sign carries information, so it is never dropped. */
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${pct(Math.abs(v))}`;

export function BlockAttribution({ start, method, basis }: {
  start: string; method: string; basis: Basis;
}) {
  const [security, setSecurity] = useState<string | null>("VT");
  const [data, setData] = useState<BlockResult | null>(null);
  const [openBlock, setOpenBlock] = useState<string | null>(null);
  const [view, setView] = useState<"variance" | "correlation">("variance");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setBusy(true); setError(null);
    api.matrixBlocks({
      instrument_id: security, start: start || null, method, basis, regimes: true,
    })
      .then((d) => {
        setData(d);
        setOpenBlock((b) => (b && d.blocks.some((x) => x.block_id === b))
          ? b : d.blocks[0]?.block_id ?? null);
      })
      .catch((e) => { setError(String(e.message ?? e)); setData(null); })
      .finally(() => setBusy(false));
  }, [security, start, method, basis]);

  const proxies = data?.proxies ?? [];
  const chosen = proxies.find((p) => p.instrument_id === security);
  const block = data?.blocks.find((b) => b.block_id === openBlock) ?? null;

  /**
   * The leader, and whether it is a leader by much. A block at 40% where the
   * next is 38% is not "the driver", and saying so would be the chart making a
   * claim the numbers do not support.
   */
  const leader = useMemo(() => {
    if (!data?.blocks.length) return null;
    const sorted = [...data.blocks].sort((a, b) => Math.abs(b.share) - Math.abs(a.share));
    const [first, second] = sorted;
    return {
      block: first,
      decisive: !second || Math.abs(first.share) > Math.abs(second.share) * 1.5,
      runnerUp: second ?? null,
    };
  }, [data]);

  return (
    <div className="space-y-4">
      {/* ---------------------------------------------------------------- */}
      <Panel
        title="Risk budget by block"
        caption={
          <>
            A covariance matrix has no risk contributions of its own — it has
            variances, correlations and eigenvalues, and none of those is a
            contribution until something is held against it. So this decomposes{" "}
            <span className="font-mono text-[10.5px]">β&apos;Σβ</span> for one
            security. Volatility is homogeneous of degree one in the exposures,
            so by Euler&rsquo;s theorem the block contributions{" "}
            <b>sum to the total exactly</b> rather than approximately, and a
            negative one means that block is hedging the rest.
          </>
        }
        actions={
          <div className="flex flex-wrap items-center gap-1">
            {proxies.map((p) => (
              <button
                key={p.instrument_id}
                onClick={() => setSecurity(p.instrument_id)}
                title={p.note}
                className={`rounded border px-2 py-0.5 text-[11px] transition ${
                  security === p.instrument_id
                    ? "border-navy bg-navy text-white"
                    : "border-line bg-white text-muted hover:border-navy2 hover:text-navy"
                }`}>
                {p.asset_class}
              </button>
            ))}
            <button
              onClick={() => setSecurity(null)}
              title="Every exposure set to one: which blocks are intrinsically
                     noisy, rather than which drive a holding."
              className={`rounded border px-2 py-0.5 text-[11px] transition ${
                security === null
                  ? "border-navy bg-navy text-white"
                  : "border-line bg-white text-muted hover:border-navy2 hover:text-navy"
              }`}>
              Equal exposure
            </button>
          </div>
        }
      >
        {error && <div className="text-[12px] text-fail">{error}</div>}
        {busy && !data && <ChartSkeleton height={300} />}

        {data && (
          <>
            <div className="mb-3 flex flex-wrap items-baseline gap-x-6 gap-y-1
                            text-[11px] text-muted">
              <span>
                <b className="text-[13px] text-navy">{pct(data.sigma)}</b>{" "}
                factor volatility
                {chosen && <> · {chosen.instrument_id} — {chosen.label}</>}
                {!security && <> · equal exposure across {data.n_factors} factors</>}
              </span>
              {data.fit && (
                <span>
                  fit R² {num(data.fit.r2, 3)}, residual {pct(data.fit.resid_vol_ann)}{" "}
                  <span className="text-[10px]">
                    (what the factors do <i>not</i> explain)
                  </span>
                </span>
              )}
              <span>
                diversification {pct(data.diversification)} — blocks apart would
                be {pct(data.undiversified)}
              </span>
            </div>

            {data.fit && (data.fit.r2 ?? 1) < 0.8 && (
              <div className="mb-3 rounded border border-line bg-lineSoft/40 px-2.5
                              py-1.5 text-[10.5px] leading-relaxed text-ink">
                The factor set explains {pct(data.fit.r2)} of this security. The
                split below is a split of that part only — the remaining{" "}
                {pct(data.fit.resid_vol_ann)} is specific risk and belongs to no
                block.
              </div>
            )}

            {leader && (
              <div className="mb-2 text-[11px] text-ink">
                Largest driver:{" "}
                <b>{leader.block.name}</b> at {signed(leader.block.share)} of risk
                {leader.decisive
                  ? ", clear of everything else."
                  : leader.runnerUp
                    ? `, but ${leader.runnerUp.name} is close behind at ${signed(leader.runnerUp.share)} — read these two together.`
                    : "."}
              </div>
            )}

            <Chart
              height={Math.max(220, data.blocks.length * 30 + 90)}
              data={[
                {
                  type: "bar", orientation: "h",
                  y: [...data.blocks].reverse().map((b) => b.name),
                  x: [...data.blocks].reverse().map((b) => b.share),
                  marker: {
                    color: [...data.blocks].reverse().map((b) =>
                      b.share >= 0 ? PALETTE[0] : "#8C3A2E"),
                  },
                  name: "contribution",
                  hovertemplate: "%{y}<br>contributes %{x:+.1%} of risk<extra></extra>",
                },
                {
                  type: "scatter", mode: "markers",
                  y: [...data.blocks].reverse().map((b) => b.name),
                  x: [...data.blocks].reverse().map((b) =>
                    data.sigma ? b.standalone / data.sigma : 0),
                  marker: { symbol: "line-ns-open", size: 11,
                            color: "#2A2F3A", line: { width: 2 } },
                  name: "standalone",
                  hovertemplate:
                    "%{y}<br>alone it would be %{x:.1%} of the total<extra></extra>",
                },
              ]}
              layout={{
                margin: { l: 120, r: 16, t: 8, b: 34 },
                xaxis: { tickformat: "+.0%", zeroline: true, title: "share of risk" },
                yaxis: { automargin: true },
                barmode: "overlay",
                legend: { orientation: "h", y: -0.25 },
              }}
            />
            <p className="mt-1 text-[10.5px] leading-snug text-muted">
              Bars are contributions and add to 100%. The ticks are what each
              block would be worth <i>on its own</i>, which does not add up and
              is not meant to: a tick to the right of its bar is a block whose
              risk is being partly cancelled by the others.
            </p>
          </>
        )}
      </Panel>

      {/* ---------------------------------------------------------------- */}
      <div className="grid gap-4 xl:grid-cols-2">
        <Panel
          title="Between the blocks"
          caption={
            view === "variance"
              ? "Total variance split across every pair. The diagonal is a block's own variance; everything off it is cross-block covariance, and the whole grid sums to 100%. A clean block structure is diagonally dominant — large off-diagonals mean two blocks are riding the same move, which is what the orthogonalisation exists to prevent."
              : "Correlation between the blocks' own return streams. This is availability, not use: a block can correlate strongly with another and still contribute nothing, if the exposure to it is small. Read it beside the variance view, never instead of it."
          }
          actions={
            <div className="flex gap-1">
              {(["variance", "correlation"] as const).map((v) => (
                <button key={v} onClick={() => setView(v)}
                        className={`rounded border px-2 py-0.5 text-[11px] transition ${
                          view === v ? "border-navy bg-navy text-white"
                                     : "border-line bg-white text-muted hover:border-navy2"}`}>
                  {v === "variance" ? "Variance" : "Correlation"}
                </button>
              ))}
            </div>
          }
        >
          {!data && <ChartSkeleton height={330} />}
          {data && (
            <Chart
              height={Math.max(330, data.block_order.length * 26 + 150)}
              data={[{
                type: "heatmap",
                z: view === "variance"
                  ? [...data.variance_share].reverse()
                  : [...data.block_correlation].reverse(),
                x: data.block_names,
                y: [...data.block_names].reverse(),
                colorscale: DIVERGING, zmid: 0,
                ...(view === "correlation" ? { zmin: -1, zmax: 1 } : {}),
                hovertemplate: view === "variance"
                  ? "%{y} × %{x}<br>%{z:+.2%} of total variance<extra></extra>"
                  : "%{y} × %{x}<br>ρ = %{z:.2f}<extra></extra>",
                colorbar: { thickness: 10, len: 0.6, tickfont: { size: 9 } },
              }]}
              layout={{
                margin: { l: 110, r: 10, t: 8, b: 110 },
                xaxis: { tickfont: { size: 9 }, tickangle: -45, dtick: 1,
                         showgrid: false },
                yaxis: { tickfont: { size: 9 }, dtick: 1, showgrid: false,
                         scaleanchor: "x", scaleratio: 1 },
              }}
            />
          )}
        </Panel>

        {/* -------------------------------------------------------------- */}
        <Panel
          title="Calm against stressed"
          caption="The same decomposition on the quietest days and the loudest, split by the rolling volatility of the panel's own first principal component — the move every factor shares, so the regime is not defined by the block being studied. What matters here is not that risk rises in a crisis, which it always does, but whether the shape changes."
        >
          {!data && <ChartSkeleton height={330} />}
          {data && !data.regimes?.available && (
            <div className="flex h-[330px] items-center justify-center px-8
                            text-center text-[11px] leading-relaxed text-muted">
              Not enough days on one side of the split to estimate both
              covariances. {data.regimes?.reason}
            </div>
          )}
          {data?.regimes?.available && data.regimes.calm && data.regimes.stress && (
            <RegimePanel data={data} />
          )}
        </Panel>
      </div>

      {/* ---------------------------------------------------------------- */}
      <Panel
        title="Inside the block"
        caption={
          <>
            The upper half of this table moves with the security; the lower
            three columns do not. <b>Contribution</b> and <b>share</b> are what
            this factor costs the holding. <b>PC1 weight</b> and{" "}
            <b>centrality</b> are properties of the covariance: which factor
            carries the block&rsquo;s first eigenvector, and how strongly each
            one moves with its neighbours. A factor with high centrality is the
            one that speaks for the block; a factor near zero is in it by
            economic classification and moves on its own.
          </>
        }
        actions={
          <select value={openBlock ?? ""} onChange={(e) => setOpenBlock(e.target.value)}
                  className="rounded border border-line bg-white px-2 py-1 text-[12px]">
            {(data?.blocks ?? []).map((b) => (
              <option key={b.block_id} value={b.block_id}>{b.name}</option>
            ))}
          </select>
        }
      >
        {!block && <ChartSkeleton height={260} />}
        {block && data && <BlockDetail block={block} sigma={data.sigma} />}
      </Panel>
    </div>
  );
}

/** Block shares in both regimes, as a slope: the movement is the point. */
function RegimePanel({ data }: { data: BlockResult }) {
  const calm = data.regimes!.calm!;
  const stress = data.regimes!.stress!;
  const names = data.block_names;

  // Only the blocks that actually moved. A slope chart of nine flat lines
  // hides the two that did something.
  const moved = calm.blocks
    .map((b, i) => ({
      block: names[i] ?? b, calm: calm.share[i], stress: stress.share[i],
      delta: stress.share[i] - calm.share[i],
    }))
    .filter((r) => Math.abs(r.delta) > 0.005)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  return (
    <>
      <div className="mb-2 grid grid-cols-3 gap-2 text-[11px]">
        <Stat label="Volatility" a={pct(calm.sigma)} b={pct(stress.sigma)} />
        <Stat label="Mean |cross-block ρ|"
              a={num(calm.mean_cross_correlation, 3)}
              b={num(stress.mean_cross_correlation, 3)} />
        <Stat label="Days" a={String(data.regimes!.n_calm)}
              b={String(data.regimes!.n_stress)} />
      </div>

      {stress.mean_cross_correlation > calm.mean_cross_correlation + 0.01 && (
        <p className="mb-2 text-[10.5px] leading-snug text-ink">
          Cross-block correlation rises in stress, so the blocks diversify each
          other <i>less</i> exactly when that matters. A full-sample correlation
          averages this away.
        </p>
      )}

      {moved.length === 0 ? (
        <p className="text-[11px] text-muted">
          No block&rsquo;s share moves by more than half a point between the two
          regimes: the shape of this security&rsquo;s risk is stable even though
          its level is not.
        </p>
      ) : (
        <Chart
          height={Math.max(200, moved.length * 28 + 80)}
          data={[
            { type: "bar", orientation: "h", name: "calm",
              y: [...moved].reverse().map((r) => r.block),
              x: [...moved].reverse().map((r) => r.calm),
              marker: { color: "#B9C4D2" },
              hovertemplate: "%{y}<br>calm %{x:+.1%}<extra></extra>" },
            { type: "bar", orientation: "h", name: "stressed",
              y: [...moved].reverse().map((r) => r.block),
              x: [...moved].reverse().map((r) => r.stress),
              marker: { color: PALETTE[0] },
              hovertemplate: "%{y}<br>stressed %{x:+.1%}<extra></extra>" },
          ]}
          layout={{
            margin: { l: 110, r: 16, t: 8, b: 34 },
            xaxis: { tickformat: "+.0%", zeroline: true, title: "share of risk" },
            yaxis: { automargin: true },
            barmode: "group",
            legend: { orientation: "h", y: -0.3 },
          }}
        />
      )}
    </>
  );
}

function Stat({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div>
      <div className="text-2xs uppercase tracking-label text-muted">{label}</div>
      <div className="font-mono text-[12px] text-ink">
        {a} <span className="text-muted">&rarr;</span> {b}
      </div>
    </div>
  );
}

function BlockDetail({ block, sigma }: { block: any; sigma: number }) {
  const rows = block.factors;
  return (
    <>
      <div className="mb-2 flex flex-wrap items-baseline gap-x-5 gap-y-1
                      text-[11px] text-muted">
        <span>
          <b className="text-ink">{signed(block.share)}</b> of total risk across{" "}
          {block.n_factors} factor{block.n_factors === 1 ? "" : "s"}
        </span>
        <span>
          own variance {pct(block.own_variance_share)}, cross-block{" "}
          {signed(block.cross_variance_share)}
        </span>
        <span title="Share of the block's correlation trace taken by its first
                     principal component. Near 1 the block moves as one thing;
                     near 1/k its factors are close to unrelated.">
          PC1 explains {pct(block.pc1_share ?? 0)} of the block
        </span>
      </div>

      <Chart
        height={Math.max(160, rows.length * 24 + 70)}
        data={[{
          type: "bar", orientation: "h",
          y: [...rows].reverse().map((f: any) => f.factor_id),
          x: [...rows].reverse().map((f: any) => f.share_of_total),
          marker: {
            color: [...rows].reverse().map((f: any) =>
              f.ctr >= 0 ? PALETTE[0] : "#8C3A2E"),
          },
          hovertemplate: "%{y}<br>%{x:+.2%} of total risk<extra></extra>",
        }]}
        layout={{
          margin: { l: 130, r: 16, t: 6, b: 32 },
          xaxis: { tickformat: "+.1%", zeroline: true },
          yaxis: { automargin: true, tickfont: { size: 9 } },
          showlegend: false,
        }}
      />

      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="border-b border-line text-2xs uppercase
                           tracking-label text-muted">
              <th className="py-1 text-left">Factor</th>
              <th className="py-1 text-right">β</th>
              <th className="py-1 text-right">Vol</th>
              <th className="py-1 text-right">Contribution</th>
              <th className="py-1 text-right">Of block</th>
              <th className="py-1 text-right" title="Weight in the block's first
                                                     eigenvector">PC1 weight</th>
              <th className="py-1 text-right" title="Mean correlation with the
                                                     other factors in this block">
                Centrality</th>
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.map((f: any) => (
              <tr key={f.factor_id} className="border-b border-lineSoft">
                <td className="py-1 text-left">{f.factor_id}</td>
                <td className="py-1 text-right">{num(f.beta, 2)}</td>
                <td className="py-1 text-right text-muted">{pct(f.vol)}</td>
                <td className={`py-1 text-right ${
                  f.ctr < 0 ? "text-fail" : "text-ink"}`}>
                  {signed(f.share_of_total)}
                </td>
                <td className="py-1 text-right text-muted">
                  {signed(f.share_of_block)}
                </td>
                <td className="py-1 text-right text-muted">
                  {num(f.pc1_loading, 2)}
                </td>
                <td className="py-1 text-right text-muted">
                  {num(f.centrality, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1.5 text-[10.5px] leading-snug text-muted">
        Shares of the block are signed and can exceed 100%: where two factors
        inside a block offset each other the net is small and the parts are not
        fractions of it. Drawing that as a pie would be a lie about the
        arithmetic, so it is a bar.
      </p>
    </>
  );
}
