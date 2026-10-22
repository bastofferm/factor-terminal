"use client";

/**
 * Perspectives: what each view of the future implies for a portfolio, and what
 * holding a portfolio would cost if that view were the right one.
 *
 * The weights between the vectors are the user's - a ten-year forecast leaves
 * no held-out span for anything else to choose them on. What the page can do is
 * make the choice's consequences visible before it is made: the frontier under
 * each vector, every portfolio scored under every vector, and the gap between a
 * portfolio and each vector's frontier at the same risk (Broadie 1993: the
 * frontier built on estimates is too optimistic, and portfolios chosen on it sit
 * below it).
 */

import { useMemo, useState } from "react";
import { Chart, Panel } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { ReturnForecast, num, pct } from "@/lib/api";
import { RETURN_METRICS as RM } from "@/lib/returnMetrics";
import { Toggle, classLabels, riskyKeys, vectorColour, vectorOrder } from "@/components/returns/shared";

type Weights = Record<string, number>;

export function Perspectives({
  result, overrides, setOverride, running, stale, onRun,
}: {
  result: ReturnForecast;
  overrides: Record<string, unknown>;
  setOverride: (key: string, value: unknown) => void;
  running: boolean;
  stale: boolean;
  onRun: () => void;
}) {
  const labels = classLabels(result);
  const keys = riskyKeys(result);
  const order = vectorOrder(result);
  const ports = result.portfolio_order?.length ? result.portfolio_order
    : result.portfolios.map((p) => p.id);
  const byId = Object.fromEntries(result.portfolios.map((p) => [p.id, p]));
  const vlabel = (k: string) => result.vectors[k]?.label ?? k;

  const conviction: Weights = (overrides.conviction as Weights) ?? result.assumptions.conviction;
  const concern: Weights = (overrides.concern as Weights) ?? result.assumptions.concern;
  const between: Weights = (overrides.between as Weights) ?? result.assumptions.between;
  const convictionKeys = order.filter((k) => result.vectors[k].group === "conviction");
  const concernKeys = order.filter((k) => result.vectors[k].group === "concern");

  const [under, setUnder] = useState<string>("blend");
  const [metric, setMetric] = useState<"gap" | "excess" | "sharpe">("gap");
  const [picked, setPicked] = useState<string>(ports[0]);

  const frontierData = useMemo(() => {
    const traces: Plotly.Data[] = order.map((k, i) => ({
      type: "scatter", mode: "lines", name: vlabel(k),
      x: result.frontiers[k]?.risk ?? [], y: result.frontiers[k]?.ret ?? [],
      line: { color: vectorColour(k, i), width: k === under ? 3 : 1.2, dash: k === "history" ? "dot" : "solid" },
      opacity: k === under ? 1 : 0.55,
      hovertemplate: `${vlabel(k)} frontier: σ %{x:.1%}, excess %{y:.2%}<extra></extra>`,
    }));
    const cells = result.matrix.filter((c) => c.vector === under);
    traces.push({
      // Names on hover only: a dozen portfolios cluster near the low-risk end
      // and their labels would print over one another.
      type: "scatter", mode: "markers", name: `Portfolios under ${vlabel(under)}`,
      x: cells.map((c) => c.risk), y: cells.map((c) => c.excess),
      text: cells.map((c) => byId[c.portfolio]?.name.replace("Max Sharpe: ", "★ ") ?? c.portfolio),
      marker: { size: 7, color: "#2A2F3A", symbol: cells.map((c) =>
        byId[c.portfolio]?.kind === "optimal" ? "star" : byId[c.portfolio]?.kind === "preset" ? "circle" : "square") },
      customdata: cells.map((c) => [c.gap, c.sharpe]),
      hovertemplate: "%{text}: σ %{x:.1%}, excess %{y:.2%}, gap %{customdata[0]:.2%}, Sharpe %{customdata[1]:.2f}<extra></extra>",
    });
    return traces;
  }, [order, result, under, byId]);

  const heat = useMemo(() => {
    const z = ports.map((p) => order.map((v) => {
      const c = result.matrix.find((m) => m.portfolio === p && m.vector === v);
      return c ? (c as any)[metric] as number | null : null;
    }));
    // Rounded before formatting, so a tiny negative prints as 0.0 rather than -0.0.
    const fmt = (x: number | null) => x == null ? ""
      : metric === "sharpe" ? (Math.round(x * 100) / 100 + 0).toFixed(2)
      : (Math.round(x * 1000) / 10 + 0).toFixed(1);
    return [{
      type: "heatmap",
      z, x: order.map(vlabel), y: ports.map((p) => byId[p]?.name ?? p),
      text: z.map((r) => r.map(fmt)), texttemplate: "%{text}", textfont: { size: 10 },
      colorscale: metric === "gap"
        ? [[0, "#FFFFFF"], [0.5, "#E9C9AE"], [1, "#A14E3A"]]
        : [[0, "#A14E3A"], [0.5, "#FFFFFF"], [1, "#2F4D73"]],
      zmid: metric === "gap" ? undefined : 0,
      showscale: false,
      hovertemplate: `%{y} under %{x}: %{text}${metric === "sharpe" ? "" : " pp"}<extra></extra>`,
    }] as Plotly.Data[];
  }, [ports, order, result, metric, byId]);

  const bandData = useMemo(() => {
    const ys = ports.map((p) => byId[p]?.name ?? p);
    const seg = (lo: (p: string) => number, hi: (p: string) => number) => {
      const x: (number | null)[] = [], y: (string | null)[] = [];
      ports.forEach((p, i) => { x.push(lo(p), hi(p), null); y.push(ys[i], ys[i], null); });
      return { x, y };
    };
    const outer = seg((p) => byId[p].band.p5, (p) => byId[p].band.p95);
    const inner = seg((p) => byId[p].band.p25, (p) => byId[p].band.p75);
    return [
      { type: "scatter", mode: "lines", name: "5-95%", ...outer, line: { color: "#9AA0AB", width: 2 },
        hoverinfo: "skip" },
      { type: "scatter", mode: "lines", name: "25-75%", ...inner, line: { color: "#476D99", width: 8 },
        hoverinfo: "skip" },
      { type: "scatter", mode: "markers", name: "median", y: ys, x: ports.map((p) => byId[p].band.p50),
        marker: { color: "#F5F4F0", size: 8, line: { color: "#2A2F3A", width: 1.5 } },
        customdata: ports.map((p) => [byId[p].band.p5, byId[p].band.p95]),
        hovertemplate: "%{y}: median %{x:.2%} (5%: %{customdata[0]:.2%}, 95%: %{customdata[1]:.2%})<extra></extra>" },
    ] as Plotly.Data[];
  }, [ports, byId]);

  const setWeight = (group: "conviction" | "concern" | "between", key: string, v: number,
                     current: Weights) => {
    setOverride(group, { ...current, [key]: Math.max(0, v) });
  };
  const convictionShare = (between.conviction ?? 0) / ((between.conviction ?? 0) + (between.concern ?? 0) || 1);

  return (
    <div className="space-y-4">
      <Panel
        title="Weights between the views"
        caption={<>Two stages, as in the paper: weights within each group, then between the groups.
          Conviction holds the forecasts you believe (the building blocks and any external capital
          market assumptions); concern holds the vectors that limit how much the allocation depends on
          them. Defaults: equal within groups, 50/50 between. Changing a weight needs a new run - the
          frontiers and the matrix are recomputed on the server.</>}
        actions={<button className="btn" onClick={onRun} disabled={running}>
          {running ? "Running…" : stale ? "Re-run with these weights" : "Re-run"}</button>}
      >
        <div className="grid gap-4 md:grid-cols-3">
          <WeightGroup title="Conviction" keys={convictionKeys} weights={conviction} labelOf={vlabel}
                       defaultWeight={1} disabled={running}
                       onChange={(k, v) => setWeight("conviction", k, v, conviction)} />
          <WeightGroup title="Concern" keys={concernKeys} weights={concern} labelOf={vlabel}
                       defaultWeight={0} disabled={running}
                       onChange={(k, v) => setWeight("concern", k, v, concern)} />
          <div>
            <div className="label">Between the groups</div>
            <input type="range" min={0} max={100} step={5} className="mt-3 w-full"
                   value={Math.round(convictionShare * 100)} disabled={running}
                   onChange={(e) => {
                     const c = Number(e.target.value) / 100;
                     setOverride("between", { conviction: c, concern: 1 - c });
                   }} />
            <div className="mt-1 flex justify-between text-[11px]">
              <span>Conviction <b>{pct(convictionShare, 0)}</b></span>
              <span>Concern <b>{pct(1 - convictionShare, 0)}</b></span>
            </div>
            {stale && <p className="mt-2 text-[11px] text-warn">The views below still show the last run.</p>}
          </div>
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel
          title="Frontiers under each view"
          caption={<>Long-only frontiers of arithmetic excess return against volatility, one per vector.
            The points are the portfolios scored under the highlighted vector; ★ marks each vector&apos;s
            maximum-Sharpe portfolio.</>}
          actions={
            <select className="field w-56" value={under} onChange={(e) => setUnder(e.target.value)}>
              {order.map((k) => <option key={k} value={k}>Score under: {vlabel(k)}</option>)}
            </select>}
        >
          <Chart data={frontierData} height={420} numericX
                 layout={{ hovermode: "closest", xaxis: { tickformat: ".0%", title: { text: "volatility" } },
                           yaxis: { tickformat: ".0%", title: { text: "expected excess return" } },
                           margin: { b: 80 } }} />
        </Panel>

        <Panel
          title="Perspectives matrix"
          caption={<>Every portfolio under every vector. The gap is how far the portfolio falls below
            that vector&apos;s frontier at the same risk, in percentage points a year
            <InfoTip metric={RM.gap} />: zero for the vector&apos;s own optimum, the price of being wrong
            for the rest.</>}
          actions={<Toggle label="Matrix metric" value={metric} onChange={setMetric}
                           options={[["gap", "Gap"], ["excess", "Excess"], ["sharpe", "Sharpe"]] as const} />}
        >
          <Chart data={heat} height={420}
                 layout={{ hovermode: "closest", margin: { l: 210, r: 10, t: 10, b: 110 },
                           xaxis: { side: "bottom", tickangle: -30 }, yaxis: { autorange: "reversed" },
                           showlegend: false }} />
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[3fr_2fr]">
        <Panel
          title="Ten-year band of each portfolio"
          caption={<>Compound annual return over ten years, simulated under the building-block draws
            and the strategic covariance; rebalanced yearly, so a diversified portfolio earns a little
            more than its weighted log returns<InfoTip metric={RM.band} />.</>}
        >
          <Chart data={bandData} height={Math.max(300, 34 * ports.length)} numericX
                 layout={{ hovermode: "closest", margin: { l: 210, r: 20, t: 10, b: 50 },
                           xaxis: { tickformat: ".0%" }, yaxis: { autorange: "reversed" } }} />
        </Panel>

        <Panel
          title="Weights"
          actions={
            <select className="field w-64" value={picked} onChange={(e) => setPicked(e.target.value)}>
              {ports.map((p) => <option key={p} value={p}>{byId[p]?.name ?? p}</option>)}
            </select>}
        >
          <table className="w-full border-collapse">
            <thead><tr><th className="th">Class</th><th className="th text-right">Weight</th></tr></thead>
            <tbody>
              {keys.filter((k) => (byId[picked]?.weights[k] ?? 0) > 0.0005).map((k) => (
                <tr key={k} className="border-t border-lineSoft">
                  <td className="cell font-sans text-[11.5px]">{labels[k]}</td>
                  <td className="cell text-right">{pct(byId[picked].weights[k], 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
            {order.filter((v) => result.matrix.some((m) => m.portfolio === picked && m.vector === v)).map((v) => {
              const c = result.matrix.find((m) => m.portfolio === picked && m.vector === v)!;
              return (
                <div key={v} className="rounded border border-lineSoft bg-white px-2 py-1">
                  <div className="truncate text-muted" title={vlabel(v)}>{vlabel(v)}</div>
                  <div className="font-mono">{pct(c.excess, 2)} · SR {num(c.sharpe, 2)}</div>
                  <div className="font-mono text-muted">gap {pct(c.gap, 2)}</div>
                </div>
              );
            })}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function WeightGroup({
  title, keys, weights, labelOf, onChange, defaultWeight, disabled,
}: {
  title: string;
  keys: string[];
  weights: Weights;
  labelOf: (k: string) => string;
  onChange: (key: string, v: number) => void;
  defaultWeight: number;
  disabled: boolean;
}) {
  const w = (k: string) => weights[k] ?? defaultWeight;
  const total = keys.reduce((s, k) => s + w(k), 0) || 1;
  return (
    <div>
      <div className="label">{title}</div>
      <div className="mt-1 space-y-1">
        {keys.map((k) => (
          <div key={k} className="flex items-center gap-2 text-[11.5px]">
            <span className="flex-1 truncate" title={labelOf(k)}>{labelOf(k)}</span>
            <input type="number" min={0} step={0.5} className="field w-16 py-0.5 text-right"
                   value={w(k)} disabled={disabled}
                   onChange={(e) => onChange(k, Number(e.target.value))} />
            <span className="w-10 text-right font-mono text-muted">{pct(w(k) / total, 0)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
