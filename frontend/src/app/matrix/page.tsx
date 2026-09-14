"use client";

/**
 * Covariance & PCA: the correlation structure of the factor panel, its eigen
 * spectrum, and how a chosen pair has co-moved through time.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, PALETTE, Panel } from "@/components/Chart";
import { api, Basis, MatrixResult, num, pct } from "@/lib/api";
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
  const [pca, setPCA] = useState<any>(null);
  const [pair, setPair] = useState<[string, string] | null>(null);
  const [rolling, setRolling] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = () => {
    setBusy(true); setError(null);
    const body = { start: start || null, method, order, basis };
    Promise.all([
      api.matrix(body),
      api.pca({ start: start || null, n_components: 10, basis }),
    ])
      .then(([m, p]) => {
        setData(m); setPCA(p);
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
    // which is how anyone expects to see a correlation matrix.
    return {
      z: [...data.correlation].reverse(),
      x: data.names,
      y: [...data.names].reverse(),
    };
  }, [data]);

  /**
   * PC1 sorted by loading, most positive to most negative.
   *
   * Registry order tells you nothing about a component — the eye has to scan the
   * whole column to find the poles. Sorted, the two ends of the axis read straight
   * off the top and bottom of the chart, which is the only thing PC1 is for.
   *
   * Plotly draws a horizontal bar chart from the bottom up, so the array is
   * reversed to put the largest loading at the top.
   */
  const pc1 = useMemo(() => {
    if (!pca?.loadings?.[0] || !pca?.names) return null;
    const order = pca.names
      .map((name: string, i: number) => ({ name, value: pca.loadings[0][i] }))
      .sort((a: any, b: any) => a.value - b.value);
    return {
      names: order.map((o: any) => o.name),
      values: order.map((o: any) => o.value),
    };
  }, [pca]);

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
          pca: pca
            ? { var_share: pca.var_share, cum_share: pca.cum_share }
            : undefined,
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
            ["Factors", data.names.length, "in the estimated matrix", "neutral",
             undefined],
            ["Observations", data.n_obs.toLocaleString(),
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

      <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
        <Panel
          title="Correlation matrix"
          caption="Click a cell to plot that pair's rolling correlation below. Blue is positive, red negative, white zero."
        >
          {heatmap && data && (
            <Chart
              height={Math.max(460, data.names.length * 12)}
              data={[{
                type: "heatmap", z: heatmap.z, x: heatmap.x, y: heatmap.y,
                colorscale: DIVERGING, zmid: 0, zmin: -1, zmax: 1,
                hovertemplate: "%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
                colorbar: { thickness: 10, len: 0.6, tickfont: { size: 9 } },
              }]}
              layout={{
                margin: { l: 130, r: 10, t: 10, b: 130 },
                xaxis: { tickfont: { size: 9 }, tickangle: -55, showgrid: false },
                yaxis: { tickfont: { size: 9 }, showgrid: false },
                hovermode: "closest", showlegend: false,
              }}
            />
          )}
        </Panel>

        <div className="space-y-4">
          <Panel
            title="Eigenvalue spectrum"
            caption={
              <>
                How much of the {basis === "excess" ? "raw" : "orthogonalised"} panel
                each component explains. Correlation-based, so a high-volatility
                factor cannot dominate the leading components through scale alone. A
                flat tail means the factors are genuinely distinct — which is what
                the orthogonalisation is for, and switching the panel above makes the
                difference visible in one bar.
              </>
            }
          >
            {pca && (
              <Chart
                height={240}
                numericX
                data={[
                  { x: pca.eigenvalues.map((_: number, i: number) => i + 1),
                    y: pca.var_share, type: "bar", name: "share",
                    marker: { color: PALETTE[0] } },
                  { x: pca.eigenvalues.map((_: number, i: number) => i + 1),
                    y: pca.cum_share, type: "scatter", mode: "lines+markers",
                    name: "cumulative", yaxis: "y2",
                    line: { color: PALETTE[1], width: 1.5 } },
                ]}
                layout={{
                  xaxis: { title: "component" },
                  yaxis: { title: "variance share", tickformat: ".0%" },
                  yaxis2: { overlaying: "y", side: "right", tickformat: ".0%",
                            range: [0, 1], showgrid: false },
                  hovermode: "closest",
                }}
              />
            )}
          </Panel>

          <Panel
            title="PC1 loadings"
            caption={
              pca ? (
                <>
                  <b>Where this comes from.</b> The {pca.names.length} factors listed
                  below, on the{" "}
                  <b>{(pca.basis ?? basis) === "excess" ? "raw" : "orthogonalised"}</b>{" "}
                  panel, over the {pca.n_obs?.toLocaleString()} days on which every
                  one of them has a value. Each column is standardised to zero mean
                  and unit variance first, so this is an eigenvector of the{" "}
                  <b>correlation</b> matrix, not the covariance: otherwise a 38%-vol
                  commodity factor would dominate the leading component through scale
                  alone. PC1 is the eigenvector with the largest eigenvalue, carrying{" "}
                  {pca.var_share ? pct(pca.var_share[0]) : "—"} of the total variance.
                  <div className="mt-1">
                    A loading is that factor&rsquo;s weight in the component. Sign is
                    arbitrary — an eigenvector times minus one is the same
                    eigenvector — so read it as grouping, not direction: factors on
                    the same side move together along this axis, and it usually comes
                    out as a global risk-on / risk-off direction. Sorted by loading so
                    the two poles read off the ends.
                  </div>
                  <div className="mt-1">
                    PDF section 5 keeps PCA as a control on the economic factors, not
                    a replacement: the question it answers is whether the named
                    factors already span the common structure.
                  </div>
                </>
              ) : (
                "Principal components of the factor panel, as a control on the "
                + "economic factors rather than a replacement for them."
              )
            }
          >
            {pca && pc1 && (
              <Chart
                height={Math.max(240, pc1.names.length * 11)}
                numericX
                data={[{
                  type: "bar", orientation: "h",
                  y: pc1.names, x: pc1.values,
                  marker: {
                    color: pc1.values.map((v: number) =>
                      v >= 0 ? PALETTE[0] : "#8C3A2E"),
                  },
                  hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
                }]}
                layout={{
                  margin: { l: 130, r: 10, t: 10, b: 36 },
                  xaxis: { title: "loading" },
                  yaxis: { tickfont: { size: 9 }, automargin: true },
                  showlegend: false, hovermode: "closest",
                }}
              />
            )}
          </Panel>
        </div>
      </div>

      <Panel
        title="Rolling pairwise correlation"
        caption="A stable full-sample correlation can hide a pair that was uncorrelated for a decade and went to 0.8 in a crisis — the Krisenkorrelation problem of PDF section 7.2."
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
    </div>
  );
}
