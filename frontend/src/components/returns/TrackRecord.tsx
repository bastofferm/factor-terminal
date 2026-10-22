"use client";

/**
 * Track record: how the building-block method would have done, on the only
 * histories long enough to say.
 *
 * The terminal's own funds start between 2000 and 2013 - fewer than two
 * independent decades - so the forecasts are tested where the record is long:
 * Shiller's US data from 1871, the JST panel of sixteen countries from 1870, and
 * Barclays' country CAPE from 1982. Everything is scored against the historical
 * mean known at the time, and the non-overlapping rows carry the honest sample
 * size. Unflattering results are shown at the same size as flattering ones.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, ChartSkeleton, Panel } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { MetricCard, MetricStrip } from "@/components/MetricCard";
import { TrackRecord as TR, TrackScore, api, num, pct, pval } from "@/lib/api";
import { RETURN_METRICS as RM } from "@/lib/returnMetrics";
import { LicenceNotes, signedPct } from "@/components/returns/shared";

export function TrackRecord({ active }: { active: boolean }) {
  const [data, setData] = useState<TR | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!active || data) return;
    api.returnsTrackRecord().then(setData).catch((e) => setError(String(e.message ?? e)));
  }, [active, data]);

  const usChart = useMemo(() => {
    if (!data) return [];
    const s = data.us.series;
    const x = s.map((r) => r.date);
    const line = (name: string, y: (number | null)[], color: string, dash = "solid", width = 1.6) => ({
      type: "scatter", mode: "lines", name, x, y, line: { color, dash, width },
      hovertemplate: `${name}: %{y:.1%}<extra></extra>`, connectgaps: false,
    });
    return [
      line("Building blocks (forecast)", s.map((r) => r.forecast), "#2F4D73", "solid", 2),
      line("Realised next decade", s.map((r) => r.realised), "#C2703D", "solid", 2),
      line("Historical mean (benchmark)", s.map((r) => r.mean), "#9AA0AB", "dash"),
      line("Last decade (rear-view)", s.map((r) => r.rear), "#B9BEC7", "dot", 1.2),
    ] as Plotly.Data[];
  }, [data]);

  const bondChart = useMemo(() => {
    if (!data) return [];
    const s = data.bonds.shiller.series;
    const x = s.map((r) => r.date);
    return [
      { type: "scatter", mode: "lines", name: "10-year yield at the start", x, y: s.map((r) => r.yield),
        line: { color: "#2F4D73", width: 2 }, hovertemplate: "yield %{y:.1%}<extra></extra>" },
      { type: "scatter", mode: "lines", name: "Realised next decade", x, y: s.map((r) => r.realised),
        line: { color: "#C2703D", width: 2 }, hovertemplate: "realised %{y:.1%}<extra></extra>" },
    ] as Plotly.Data[];
  }, [data]);

  if (error) {
    return <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">{error}</div>;
  }
  if (!data) return <Panel title="Track record"><ChartSkeleton height={360} /></Panel>;

  const pr = data.us.predictors;
  const nov = data.us.nonoverlapping;
  const p = data.panel;

  return (
    <div className="space-y-4">
      <MetricStrip>
        <MetricCard label="CAPE yield vs next decade" value={num(pr.caey_corr, 2)}
                    sub={`US ${pr.from}-${pr.to}; Ilmanen (AQR, 2025): +0.52`} />
        <MetricCard label="Last decade vs next" value={num(pr.rear_corr, 2)} tone="bad"
                    sub="the rear-view mirror points the wrong way; Ilmanen: −0.37" />
        <MetricCard label="US OOS R², decades" value={num(nov.oos_r2, 2)} metric={RM.oosR2}
                    tone={(nov.oos_r2 ?? 0) > 0 ? "good" : "bad"}
                    sub={`${nov.n} non-overlapping decades, DM p ${pval(nov.dm_p)}`} />
        <MetricCard label="Bonds: yield vs next decade" value={num(data.bonds.shiller.corr, 2)}
                    tone="good"
                    sub={`GS10 ${data.bonds.shiller.from}-${data.bonds.shiller.to}, mean error ${signedPct(data.bonds.shiller.mean_error, 2)}`} />
        {p && <MetricCard label="JST panel OOS R², ex US" value={num(p.pooled_ex_us.oos_r2, 2)}
                          tone={(p.pooled_ex_us.oos_r2 ?? 0) > 0 ? "good" : "bad"} metric={RM.oosR2}
                          sub={`${p.pooled_ex_us.countries} countries; without the valuation block ${num(p.pooled_ex_us_no_valuation.oos_r2, 2)}`} />}
      </MetricStrip>

      <Panel
        title="US: the building-block forecast against the decade that followed"
        caption={<>Real compound annual returns, Shiller data. At each January from 1901 the forecast is
          rebuilt from what was known then: the dividend yield, a CAPE with earnings lagged{" "}
          {data.settings.report_lag_months} months, real earnings growth and the CAPE norm as expanding
          means. Plotted on the forecast date; the realised line is the following ten years.</>}
      >
        <Chart data={usChart} height={360} layout={{ yaxis: { tickformat: ".0%" } }} />
        <ScoreTable rows={[
          ...data.us.periods.map((r) => ({ label: `Monthly from ${String(r.from).slice(0, 4)}`, ...r })),
          { label: "Non-overlapping decades", ...nov },
          { label: "Rear-view mirror instead", ...data.us.rear_view },
        ]} />
        <p className="mt-2 text-[11px] text-muted">
          Overlapping monthly windows share most of their outcome, so the Diebold-Mariano test uses 119
          lags and the non-overlapping row is the honest sample size. Since 1983 the forecast has been
          too pessimistic: valuations stayed above their expanding norm for four decades, which is the
          paper&apos;s point that a CAPE norm implicitly assumes rates return to the level that shaped it.
        </p>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel
          title="Bonds: the starting yield is the forecast"
          caption={<>Shiller&apos;s 10-year yield against the bond&apos;s nominal return over the next
            decade, {data.bonds.shiller.from}-{data.bonds.shiller.to}. The terminal&apos;s own Treasury
            funds since 2002 cover 1.4 independent decades and are descriptive only.</>}
        >
          <Chart data={bondChart} height={300} layout={{ yaxis: { tickformat: ".0%" } }} />
          <table className="mt-2 w-full border-collapse">
            <thead><tr><th className="th">Series</th><th className="th text-right">Corr.</th>
              <th className="th text-right">Mean error</th><th className="th text-right">Span</th></tr></thead>
            <tbody>
              <tr className="border-t border-lineSoft">
                <td className="cell font-sans">GS10, monthly</td>
                <td className="cell text-right">{num(data.bonds.shiller.corr, 2)}</td>
                <td className="cell text-right">{signedPct(data.bonds.shiller.mean_error, 2)}</td>
                <td className="cell text-right">{data.bonds.shiller.n} months</td>
              </tr>
              <tr className="border-t border-lineSoft">
                <td className="cell font-sans">GS10, decades</td>
                <td className="cell text-right">{num(data.bonds.shiller.nonoverlapping.corr, 2)}</td>
                <td className="cell text-right">{signedPct(data.bonds.shiller.nonoverlapping.mean_error, 2)}</td>
                <td className="cell text-right">{data.bonds.shiller.nonoverlapping.n}</td>
              </tr>
              {Object.entries(data.bonds.etfs).map(([k, v]) => (
                <tr key={k} className="border-t border-lineSoft">
                  <td className="cell font-sans">{k} (descriptive)</td>
                  <td className="cell text-right">{num(v.corr, 2)}</td>
                  <td className="cell text-right">{signedPct(v.mean_error, 2)}</td>
                  <td className="cell text-right">{v.independent_decades} decades</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <Panel
          title="Country CAPE (Barclays) against the next decade"
          caption="Correlation of the CAPE yield (1/CAPE) at each year-end since 1982 with the following ten years' real return from the JST panel. Two to three independent decades per market: a description, not a test."
        >
          <table className="w-full border-collapse">
            <thead><tr><th className="th">Market</th><th className="th text-right">Corr.</th>
              <th className="th text-right">Years</th><th className="th text-right">Decades</th>
              <th className="th text-right">Span</th></tr></thead>
            <tbody>
              {data.cape_intl.map((c) => (
                <tr key={c.country} className="border-t border-lineSoft">
                  <td className="cell font-sans">{c.country}</td>
                  <td className={"cell text-right " + ((c.corr ?? 0) > 0 ? "text-pass" : "text-fail")}>{num(c.corr, 2)}</td>
                  <td className="cell text-right">{c.n}</td>
                  <td className="cell text-right">{c.independent_decades}</td>
                  <td className="cell text-right">{c.from}-{c.to}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      {p && (
        <Panel
          title="Sixteen countries since 1890: the dividend-yield form of the method"
          caption={<>JST panel, annual, non-overlapping decades: dividend yield + expanding-mean real
            dividend growth + reversion of the yield toward its expanding norm, against each
            country&apos;s expanding historical mean. Pooled, the losses are averaged across countries
            within each decade before testing, because one decade&apos;s surprise hits every market at
            once. The second R² column drops the valuation block (Brightman&apos;s form). This is the
            evidence behind the dividend-yield valuation used for EM and REITs.</>}
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead><tr>
                <th className="th">Country</th><th className="th text-right">Decades</th>
                <th className="th text-right">Corr.</th><th className="th text-right">Mean error</th>
                <th className="th text-right">OOS R²<InfoTip metric={RM.oosR2} /></th>
                <th className="th text-right">without valuation</th><th className="th text-right">Span</th>
              </tr></thead>
              <tbody>
                {p.countries.map((c) => (
                  <tr key={c.iso} className="border-t border-lineSoft">
                    <td className="cell font-sans">{c.country}</td>
                    <td className="cell text-right">{c.n}</td>
                    <td className="cell text-right">{num(c.corr, 2)}</td>
                    <td className="cell text-right">{signedPct(c.mean_error, 1)}</td>
                    <td className={"cell text-right " + ((c.oos_r2 ?? 0) > 0 ? "text-pass" : "text-fail")}>{num(c.oos_r2, 2)}</td>
                    <td className={"cell text-right " + ((c.oos_r2_no_valuation ?? 0) > 0 ? "text-pass" : "text-fail")}>{num(c.oos_r2_no_valuation, 2)}</td>
                    <td className="cell text-right text-muted">{c.from}-{c.to}</td>
                  </tr>
                ))}
                {([["Pooled, ex US", p.pooled_ex_us, p.pooled_ex_us_no_valuation],
                   ["Pooled, all", p.pooled_all, p.pooled_all_no_valuation]] as const).map(([l, a, b]) => (
                  <tr key={l} className="border-t border-line bg-white font-semibold">
                    <td className="cell font-sans">{l}</td>
                    <td className="cell text-right">{a.n}</td>
                    <td className="cell text-right text-muted">t {num(a.t_stat, 2)}</td>
                    <td className="cell text-right">{signedPct(a.mean_error, 1)}</td>
                    <td className={"cell text-right " + ((a.oos_r2 ?? 0) > 0 ? "text-pass" : "text-fail")}>{num(a.oos_r2, 2)}</td>
                    <td className={"cell text-right " + ((b.oos_r2 ?? 0) > 0 ? "text-pass" : "text-fail")}>{num(b.oos_r2, 2)}
                      <span className="ml-1 font-normal text-muted">t {num(b.t_stat, 2)}</span></td>
                    <td className="cell text-right text-muted">{a.decades} decades</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <LicenceNotes licences={data.licences} />
    </div>
  );
}

function ScoreTable({ rows }: { rows: (TrackScore & { label: string })[] }) {
  return (
    <div className="mt-2 overflow-x-auto">
      <table className="w-full border-collapse">
        <thead><tr>
          <th className="th">Sample</th><th className="th text-right">n</th>
          <th className="th text-right">Corr.</th><th className="th text-right">Mean error</th>
          <th className="th text-right">RMSE</th><th className="th text-right">RMSE of mean</th>
          <th className="th text-right">OOS R²<InfoTip metric={RM.oosR2} /></th>
          <th className="th text-right">DM t<InfoTip metric={RM.dmT} /></th>
          <th className="th text-right">p</th>
        </tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-lineSoft">
              <td className="cell font-sans">{r.label}</td>
              <td className="cell text-right">{r.n}</td>
              <td className="cell text-right">{num(r.corr, 2)}</td>
              <td className="cell text-right">{signedPct(r.mean_error, 1)}</td>
              <td className="cell text-right">{pct(r.rmse, 1)}</td>
              <td className="cell text-right">{pct(r.rmse_benchmark, 1)}</td>
              <td className={"cell text-right " + ((r.oos_r2 ?? 0) > 0 ? "text-pass" : "text-fail")}>{num(r.oos_r2, 2)}</td>
              <td className="cell text-right">{num(r.dm_t, 2)}</td>
              <td className={"cell text-right " + ((r.dm_p ?? 1) < 0.05 ? "font-semibold" : "")}>{pval(r.dm_p)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
