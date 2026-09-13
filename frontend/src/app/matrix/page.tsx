"use client";

/**
 * Covariance & PCA: the correlation structure of the factor panel, its eigen
 * spectrum, and how a chosen pair has co-moved through time.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, PALETTE, Panel, Stat } from "@/components/Chart";
import { api, MatrixResult, num, pct } from "@/lib/api";
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
  const [data, setData] = useState<MatrixResult | null>(null);
  const [pca, setPCA] = useState<any>(null);
  const [pair, setPair] = useState<[string, string] | null>(null);
  const [rolling, setRolling] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = () => {
    setBusy(true); setError(null);
    const body = { start: start || null, method, order };
    Promise.all([api.matrix(body), api.pca({ start: start || null, n_components: 10 })])
      .then(([m, p]) => {
        setData(m); setPCA(p);
        if (!pair && m.names.length >= 2) setPair([m.names[0], m.names[1]]);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  useEffect(load, [method, order, start]);

  useEffect(() => {
    if (!pair) return;
    api.rollingCorrelation({ factor_a: pair[0], factor_b: pair[1], window: 252 })
      .then(setRolling)
      .catch(() => setRolling(null));
  }, [pair]);

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

  const d = data?.diagnostics;

  usePublishSnapshot(
    data
      ? {
          method: data.method,
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
            <b>This page is the model&rsquo;s covariance, so it is computed on the
            orthogonalised factors</b> — not the raw series the Factor Explorer
            shows. It has to be: the loadings in the Loadings Lab are loadings on
            the orthogonalised set, and portfolio risk is β&prime;Σβ, so Σ must be
            the covariance of the same series the βs refer to. On the raw factors
            the correlations would be far higher — that overlap is exactly what the
            orthogonalisation removes.
            <br />
            Shrinkage matters here: with 39 factors on a rolling window the sample
            covariance is badly conditioned, and an optimiser handed an indefinite
            matrix will happily build a portfolio out of the negative eigenvalue.
          </>
        }
      >
        <div className="flex flex-wrap items-end gap-3">
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
        <div className="grid grid-cols-2 gap-4 rounded border border-line bg-panel px-4 py-3 md:grid-cols-4 lg:grid-cols-8">
          <Stat label="Factors" value={data.names.length} />
          <Stat label="Observations" value={data.n_obs.toLocaleString()} />
          <Stat label="Span" value={`${data.start} → ${data.end}`} />
          <Stat label="Shrinkage δ" value={num(d.shrink_intensity, 3)}
                hint="0 = sample, 1 = target" />
          <Stat label="Condition number" value={num(d.condition_number, 0)}
                tone={d.condition_number > 500 ? "bad" : "neutral"}
                hint="ratio of largest to smallest eigenvalue" />
          <Stat label="Min eigenvalue" value={d.min_eigenvalue.toExponential(2)}
                tone={d.min_eigenvalue < 0 ? "bad" : "good"} />
          <Stat label="PSD" value={d.is_psd ? "yes" : "no"}
                tone={d.is_psd ? "good" : "bad"}
                hint={d.psd_repaired ? "repaired by eigenvalue clipping" : "as estimated"} />
          <Stat label="PC1 share" value={pct(d.pc1_share)}
                hint="one dominant direction means one factor" />
        </div>
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
            caption="Principal components of the orthogonalised factor panel, correlation-based so a high-volatility factor cannot dominate the leading components through scale alone. A flat tail means the factors are genuinely distinct — which is what the orthogonalisation is for."
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
            caption="Usually a global risk-on / risk-off direction. PDF section 5 keeps PCA as a control on the economic factors, not a replacement."
          >
            {pca && (
              <Chart
                height={Math.max(240, pca.names.length * 11)}
                numericX
                data={[{
                  type: "bar", orientation: "h",
                  y: pca.names, x: pca.loadings[0],
                  marker: {
                    color: pca.loadings[0].map((v: number) =>
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
