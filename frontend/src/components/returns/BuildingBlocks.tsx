"use client";

/**
 * Building blocks: each class's ten-year expected return, block by block, with
 * its band and the rear-view mirror beside it.
 *
 * The chart is the argument of the page in one picture: the bars are what is
 * observable today, the diamond is their sum, the whisker is how wide the next
 * decade can plausibly land, and the cross is what simply extrapolating the
 * fund's own history would have said. The table under it opens each row onto
 * the sources and the checks behind its numbers.
 */

import { Fragment, useMemo, useState } from "react";
import { Chart, Panel } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { ReturnForecast, pct } from "@/lib/api";
import { RETURN_METRICS as RM } from "@/lib/returnMetrics";
import { LicenceNotes, SourceList, classLabels, signedPct } from "@/components/returns/shared";

const BLOCKS = [
  ["income", "Income", "#2F4D73"],
  ["buyback", "Net buybacks", "#6B86A8"],
  ["real_growth", "Real growth", "#5B8C5A"],
  ["inflation", "Inflation", "#B5A642"],
  ["valuation", "Valuation", "#C2703D"],
  ["other", "Other", "#8B5E83"],
] as const;

export function BuildingBlocks({ result }: { result: ReturnForecast }) {
  const labels = classLabels(result);
  const etf = Object.fromEntries(result.classes.map((c) => [c.key, c.etf]));
  const [open, setOpen] = useState<string | null>(null);
  const rows = result.blocks;

  const data = useMemo(() => {
    const y = rows.map((b) => labels[b.key]);
    const traces: Plotly.Data[] = BLOCKS.map(([k, name, colour]) => ({
      type: "bar", orientation: "h", name, y,
      x: rows.map((b) => (b as any)[k] as number),
      marker: { color: colour },
      hovertemplate: `${name}: %{x:.2%}<extra>%{y}</extra>`,
    }));
    traces.push({
      type: "scatter", mode: "markers", name: "Band (5-95%, median)", y,
      x: rows.map((b) => b.band.p50),
      error_x: {
        type: "data", symmetric: false, thickness: 1.2, width: 4, color: "#2A2F3A",
        array: rows.map((b) => b.band.p95 - b.band.p50),
        arrayminus: rows.map((b) => b.band.p50 - b.band.p5),
      },
      marker: { size: 5, color: "#2A2F3A", symbol: "line-ns-open" },
      customdata: rows.map((b) => [b.band.p5, b.band.p95]),
      hovertemplate: "median %{x:.2%} (5%: %{customdata[0]:.2%}, 95%: %{customdata[1]:.2%})<extra>%{y}</extra>",
    });
    traces.push({
      type: "scatter", mode: "markers", name: "Expected return", y,
      x: rows.map((b) => b.total),
      marker: { symbol: "diamond", size: 10, color: "#F5F4F0", line: { color: "#2A2F3A", width: 1.6 } },
      hovertemplate: "expected %{x:.2%}<extra>%{y}</extra>",
    });
    traces.push({
      type: "scatter", mode: "markers", name: "Own history (rear-view)", y,
      x: rows.map((b) => b.history.full?.geo ?? null),
      marker: { symbol: "x-thin-open", size: 9, color: "#A14E3A", line: { width: 2 } },
      customdata: rows.map((b) => b.history.full?.start ?? ""),
      hovertemplate: "history %{x:.2%} since %{customdata}<extra>%{y}</extra>",
    });
    return traces;
  }, [rows, labels]);

  const cape = rows.filter((b) => b.check?.cape !== undefined);
  const usedSources = Array.from(new Set(cape.map((b) => b.check.licence as string)));

  return (
    <div className="space-y-4">
      <Panel
        title="Ten-year expected returns, block by block"
        caption={<>Compound annual returns in dollars. The bars are what is observable today; the
          diamond is their sum; the whisker is the 5-95% band of the next decade; the cross is the
          fund&apos;s own history extrapolated - the rear-view mirror. Cash is expected to earn{" "}
          <b>{pct(result.cash.total, 2)}</b> a year.</>}
      >
        <Chart
          data={data}
          height={420}
          numericX
          layout={{
            barmode: "relative",
            hovermode: "closest",
            margin: { l: 170, r: 20, t: 10, b: 60 },
            xaxis: { tickformat: ".0%", zeroline: true, title: { text: "compound annual return" } },
            yaxis: { autorange: "reversed" },
          }}
        />
      </Panel>

      <Panel
        title="The numbers and where they come from"
        caption="Click a row for its sources, the valuation norms it was compared with and the checks."
      >
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="th">Class</th>
                <th className="th">Fund</th>
                <th className="th text-right">Income<InfoTip metric={RM.income} /></th>
                <th className="th text-right">Buyback</th>
                <th className="th text-right">Growth<InfoTip metric={RM.real_growth} /></th>
                <th className="th text-right">Infl.<InfoTip metric={RM.inflation} /></th>
                <th className="th text-right">Valuation<InfoTip metric={RM.valuation} /></th>
                <th className="th text-right">Other<InfoTip metric={RM.other} /></th>
                <th className="th text-right">Expected<InfoTip metric={RM.total} /></th>
                <th className="th text-right">Arith.<InfoTip metric={RM.arithmetic} /></th>
                <th className="th text-right">Vol<InfoTip metric={RM.vol} /></th>
                <th className="th text-right">5%<InfoTip metric={RM.band} /></th>
                <th className="th text-right">95%</th>
                <th className="th text-right">History<InfoTip metric={RM.history} /></th>
                <th className="th text-right">Last 10y</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((b) => (
                <Fragment key={b.key}>
                  <tr onClick={() => setOpen(open === b.key ? null : b.key)}
                      className={"cursor-pointer border-t border-lineSoft hover:bg-white "
                        + (open === b.key ? "bg-white" : "")}>
                    <td className="cell font-sans text-[11.5px]">
                      <span className="mr-1 inline-block w-2 text-muted">{open === b.key ? "▾" : "▸"}</span>
                      {labels[b.key]}</td>
                    <td className="cell text-muted">{etf[b.key]}</td>
                    <td className="cell text-right">{pct(b.income, 2)}</td>
                    <td className="cell text-right">{pct(b.buyback, 2)}</td>
                    <td className="cell text-right">{pct(b.real_growth, 2)}</td>
                    <td className="cell text-right">{pct(b.inflation, 2)}</td>
                    <td className={"cell text-right " + (b.valuation < -0.005 ? "text-fail"
                        : b.valuation > 0.005 ? "text-pass" : "")}>{signedPct(b.valuation, 2)}</td>
                    <td className="cell text-right">{signedPct(b.other, 2)}</td>
                    <td className="cell text-right font-semibold text-navy">{pct(b.total, 2)}</td>
                    <td className="cell text-right">{pct(b.arithmetic, 2)}</td>
                    <td className="cell text-right">{pct(b.vol, 1)}</td>
                    <td className="cell text-right text-muted">{pct(b.band.p5, 1)}</td>
                    <td className="cell text-right text-muted">{pct(b.band.p95, 1)}</td>
                    <td className="cell text-right">{pct(b.history.full?.geo, 1)}</td>
                    <td className="cell text-right">{pct(b.history.trailing10?.geo, 1)}</td>
                  </tr>
                  {open === b.key && (
                    <tr className="bg-white">
                      <td colSpan={15} className="px-3 pb-3 pt-1">
                        <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
                          <SourceList sources={b.sources} />
                          <div className="space-y-1 text-[11px] text-muted">
                            {b.check?.targets && (
                              <div>
                                <span className="label">CAPE {b.check.cape?.toFixed(1)} against its norm since</span>
                                <div className="mt-0.5 font-mono">
                                  {Object.entries(b.check.targets as Record<string, number>).map(([y, v]) => (
                                    <span key={y} className="mr-3">{y}: {v.toFixed(1)}</span>))}
                                </div>
                              </div>
                            )}
                            {b.check?.dy_valuation !== undefined && (
                              <div>Dividend-yield method instead of CAPE: valuation{" "}
                                <b className="font-mono">{signedPct(b.check.dy_valuation, 2)}</b> a year.</div>
                            )}
                            {b.check?.dy_target !== undefined && (
                              <div>Dividend-yield norm <b className="font-mono">{pct(b.check.dy_target, 2)}</b>
                                {b.check.dy_valuation_weight !== undefined && b.check.dy_valuation_weight !== 1 &&
                                  <>, counted at {pct(b.check.dy_valuation_weight, 0)}</>}.</div>
                            )}
                            <div>Volatility: monthly {pct(b.vol, 1)}, quarterly {pct(b.vol_quarterly, 1)},
                              daily factor model {pct(b.vol_factor, 1)}.</div>
                            <div>History since {b.history.full?.start ?? "—"} ({b.history.full?.years ?? "—"} years).</div>
                            <div>Band median {pct(b.band.p50, 2)}, mean {pct(b.band.mean, 2)}.</div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
              <tr className="border-t border-line">
                <td className="cell font-sans text-[11.5px]">
                  <span className="mr-1 inline-block w-2" />Cash</td>
                <td className="cell text-muted">—</td>
                <td className="cell text-right">{pct(result.cash.total, 2)}</td>
                <td colSpan={5} />
                <td className="cell text-right font-semibold text-navy">{pct(result.cash.total, 2)}</td>
                <td colSpan={6} className="cell font-sans text-[10.5px] text-muted">
                  {result.cash.sources[0]?.source} · {result.cash.sources[0]?.as_of}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      <LicenceNotes licences={result.licences} only={usedSources} />
    </div>
  );
}
