"use client";

/**
 * The factor model run on a portfolio of stocks and funds.
 *
 * Start from a pre-defined portfolio, a sector portfolio or a fund portfolio, or
 * build one: stocks from the catalogue search, funds from a filtered list that
 * marks the broad market representatives and the funds a factor is built from.
 * The run estimates each holding once (stored for later), forecasts the
 * portfolio's risk day by day, backtests it on the portfolio's own returns,
 * decomposes it by holding and by factor block, and - against a benchmark fund -
 * gives the tracking error. See backend/pipeline/portfolio_risk.py.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, PALETTE, Panel } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { MetricCard, MetricStrip, alignFor } from "@/components/MetricCard";
import { SecuritySearch } from "@/components/SecuritySearch";
import {
  api, Fund, num, pct, PortfolioHolding, PortfolioJob, PortfolioPreset, pval,
} from "@/lib/api";
import { blockStyle } from "@/lib/blocks";
import { Metric } from "@/lib/metrics";
import { PORTFOLIO_METRICS as PM } from "@/lib/portfolioMetrics";
import { backtest, Tone } from "@/lib/verdict";

const GROUP_LABEL: Record<string, string> = {
  predefined: "Pre-defined portfolios",
  sector: "Sector portfolios (five large caps each)",
  funds: "Fund portfolios",
};

type Row = PortfolioHolding & { kind?: "stock" | "fund" };
type Tile = { label: string; value: string; sub: string; tone: Tone; metric: Metric };

export function PortfolioRisk() {
  const [presets, setPresets] = useState<PortfolioPreset[]>([]);
  const [fundData, setFundData] = useState<{ groups: { key: string; label: string }[];
                                              funds: Fund[] } | null>(null);
  const [presetId, setPresetId] = useState("mega_cap_tech");
  const [rows, setRows] = useState<Row[]>([]);
  const [benchmark, setBenchmark] = useState<string>("");
  const [horizon, setHorizon] = useState(21);
  const [covHalflife, setCovHalflife] = useState(42);

  // fund filters
  const [fundGroup, setFundGroup] = useState("all");
  const [broadOnly, setBroadOnly] = useState(true);
  const [hideInputs, setHideInputs] = useState(false);
  const [fundQuery, setFundQuery] = useState("");
  const [stockPick, setStockPick] = useState("");

  const [job, setJob] = useState<PortfolioJob | null>(null);
  const [ranWith, setRanWith] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.portfolioPresets().then(setPresets).catch((e) => setError(String(e.message ?? e)));
    api.portfolioFunds().then(setFundData).catch(() => setFundData(null));
  }, []);

  const fundById = useMemo(
    () => Object.fromEntries((fundData?.funds ?? []).map((f) => [f.instrument_id, f])),
    [fundData]);

  // Loading a preset replaces the holdings and the benchmark.
  useEffect(() => {
    const p = presets.find((x) => x.id === presetId);
    if (!p) return;
    setRows(p.holdings.map((h) => ({ ...h, kind: h.instrument_id.includes(":") ? "stock" : "fund" })));
    setBenchmark(p.benchmark ?? "");
  }, [presetId, presets]);

  const running = job?.status === "running";
  useEffect(() => {
    if (!running || !job) return;
    const id = window.setInterval(() => {
      api.portfolioGet(job.job_id).then(setJob).catch(() => {});
    }, 1200);
    return () => window.clearInterval(id);
  }, [running, job?.job_id]);

  const total = rows.reduce((s, r) => s + (r.weight || 0), 0);

  const addHolding = (id: string, name?: string | null) => {
    if (!id || rows.some((r) => r.instrument_id === id)) return;
    const n = rows.length;
    // The newcomer gets an equal share; the others shrink in proportion.
    const scale = n ? n / (n + 1) : 1;
    setRows([...rows.map((r) => ({ ...r, weight: r.weight * scale })),
             { instrument_id: id, name: name ?? fundById[id]?.name ?? null,
               weight: 1 / (n + 1), kind: id.includes(":") ? "stock" : "fund" }]);
    setPresetId("custom");
  };
  const setWeight = (i: number, pctValue: number) => {
    setRows(rows.map((r, k) => (k === i ? { ...r, weight: pctValue / 100 } : r)));
    setPresetId("custom");
  };
  const remove = (i: number) => { setRows(rows.filter((_, k) => k !== i)); setPresetId("custom"); };
  const equalWeight = () => setRows(rows.map((r) => ({ ...r, weight: 1 / rows.length })));
  const normalise = () => total > 0 && setRows(rows.map((r) => ({ ...r, weight: r.weight / total })));

  const funds = useMemo(() => {
    const q = fundQuery.trim().toLowerCase();
    return (fundData?.funds ?? []).filter((f) =>
      (fundGroup === "all" || f.group === fundGroup)
      && (!broadOnly || f.broad)
      && (!hideInputs || f.factor_inputs.length === 0)
      && (!q || f.instrument_id.toLowerCase().includes(q) || f.name.toLowerCase().includes(q)));
  }, [fundData, fundGroup, broadOnly, hideInputs, fundQuery]);

  const benchmarkFunds = useMemo(
    () => (fundData?.funds ?? []).filter((f) => f.broad || f.group === "equity_sector"
                                               || f.instrument_id === benchmark),
    [fundData, benchmark]);

  // What a run was asked for, so a result can say when the inputs have moved on.
  const inputs = JSON.stringify([rows.map((r) => [r.instrument_id, r.weight]), benchmark,
                                 horizon, covHalflife]);

  const start = () => {
    setError(null);
    const holdings = rows.filter((r) => r.weight).map((r) => ({
      instrument_id: r.instrument_id, weight: r.weight }));
    if (!holdings.length) { setError("Give at least one holding a weight."); return; }
    setRanWith(inputs);
    api.portfolioRun({ holdings, benchmark: benchmark || null, horizon_days: horizon,
                       cov_halflife: covHalflife })
      .then((r) => api.portfolioGet(r.job_id))
      .then(setJob)
      .catch((e) => setError(String(e.message ?? e)));
  };

  const res = job?.result ?? null;
  const stale = !!res && !running && ranWith !== inputs;

  return (
    <div className="space-y-4">
      <Panel
        title="Portfolio"
        caption={<>Runs the risk model on a portfolio of stocks and funds: each holding is
          estimated on its asset-class factor template (a stock with its own sector
          factor), the portfolio&apos;s risk is forecast day by day from the factor
          covariance and the holdings&apos; specific risks, and the forecast is
          backtested on the portfolio&apos;s own returns. Holdings are estimated once and
          stored, so a second run is quick.</>}
      >
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="label">Start from</label>
            <select className="field mt-1 w-72" value={presetId}
                    onChange={(e) => setPresetId(e.target.value)} disabled={running}>
              {(["predefined", "sector", "funds"] as const).map((g) => (
                <optgroup key={g} label={GROUP_LABEL[g]}>
                  {presets.filter((p) => p.group === g).map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </optgroup>
              ))}
              <option value="custom">Custom (as edited below)</option>
            </select>
          </div>
          <div>
            <label className="label inline-flex items-center">
              Benchmark<InfoTip metric={PM.trackingError} label="Benchmark" />
            </label>
            <select className="field mt-1 w-64" value={benchmark} disabled={running}
                    onChange={(e) => setBenchmark(e.target.value)}>
              <option value="">None</option>
              {benchmarkFunds.map((f) => (
                <option key={f.instrument_id} value={f.instrument_id}>
                  {f.instrument_id} · {f.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label">Horizon</label>
            <select className="field mt-1 w-28" value={horizon} disabled={running}
                    onChange={(e) => setHorizon(Number(e.target.value))}>
              {[5, 21, 63].map((h) => <option key={h} value={h}>{h} days</option>)}
            </select>
          </div>
          <div>
            <label className="label">Covariance half-life</label>
            <select className="field mt-1 w-48" value={covHalflife} disabled={running}
                    onChange={(e) => setCovHalflife(Number(e.target.value))}>
              <option value={42}>42 days (calibrated)</option>
              <option value={21}>21 days (fast)</option>
              <option value={63}>63 days (slow)</option>
            </select>
          </div>
          <button className="btn ml-auto min-w-[150px]" onClick={start} disabled={running}>
            {running ? `Running… ${Math.round(job!.elapsed)}s` : "Run portfolio risk"}
          </button>
        </div>
        {presets.find((p) => p.id === presetId)?.note && (
          <p className="mt-1.5 text-[11px] text-muted">
            {presets.find((p) => p.id === presetId)!.note}
          </p>
        )}

        {/* --- holdings ------------------------------------------------------ */}
        <div className="mt-3 max-h-[260px] overflow-auto rounded border border-line bg-white">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-panel">
              <tr>
                <th className="th text-left">Holding</th>
                <th className="th text-left">Name</th>
                <th className="th text-left">Type</th>
                <th className="th">Weight %</th>
                <th className="th" />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const f = fundById[r.instrument_id];
                return (
                  <tr key={r.instrument_id} className="border-t border-lineSoft">
                    <td className="cell text-left font-mono">{r.instrument_id}</td>
                    <td className="cell max-w-[260px] truncate text-left font-sans">
                      {r.name ?? f?.name ?? ""}</td>
                    <td className="cell text-left font-sans text-[11px]">
                      {f ? <>fund · {f.group_label}</> : r.kind === "stock" ? "stock" : "fund"}
                      {f?.broad && <Chip tone="navy">broad market</Chip>}
                      {f && f.factor_inputs.length > 0 && (
                        <Chip tone="warn" title={`Input to ${f.factor_inputs.join(", ")}`}>
                          factor input</Chip>)}
                    </td>
                    <td className="cell">
                      <input type="number" step="0.5" className="field w-20 py-0.5 text-right"
                             value={Math.round(r.weight * 1000) / 10} disabled={running}
                             onChange={(e) => setWeight(i, Number(e.target.value))} />
                    </td>
                    <td className="cell">
                      <button className="btn-ghost px-1.5 py-0" onClick={() => remove(i)}
                              disabled={running} title="Remove">✕</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11.5px]">
          <span>{rows.length} holdings · weights sum to{" "}
            <b className={Math.abs(total - 1) > 0.001 ? "text-warn" : ""}>{pct(total, 1)}</b>
          </span>
          <button className="btn-ghost px-2 py-0.5" onClick={equalWeight} disabled={running || !rows.length}>
            Equal weight</button>
          <button className="btn-ghost px-2 py-0.5" onClick={normalise}
                  disabled={running || !rows.length || Math.abs(total - 1) < 1e-9}>
            Rescale to 100%</button>
          <button className="btn-ghost px-2 py-0.5" onClick={() => { setRows([]); setPresetId("custom"); }}
                  disabled={running}>Clear</button>
        </div>

        {/* --- adding holdings --------------------------------------------------- */}
        <div className="mt-3 grid gap-4 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
          <div>
            <label className="label">Add a stock</label>
            <div className="mt-1 flex gap-2">
              <div className="flex-1">
                <SecuritySearch value={stockPick} onChange={setStockPick} types="equity" />
              </div>
              <button className="btn" disabled={running || !stockPick}
                      onClick={() => { addHolding(stockPick); setStockPick(""); }}>Add</button>
            </div>
          </div>
          <div>
            <label className="label">Add a fund</label>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <select className="field w-56" value={fundGroup}
                      onChange={(e) => setFundGroup(e.target.value)}>
                <option value="all">All fund groups</option>
                {(fundData?.groups ?? []).map((g) => (
                  <option key={g.key} value={g.key}>{g.label}</option>
                ))}
              </select>
              <input className="field w-36" placeholder="Filter…" value={fundQuery}
                     onChange={(e) => setFundQuery(e.target.value)} />
              <label className="flex items-center gap-1 text-[11.5px]">
                <input type="checkbox" checked={broadOnly}
                       onChange={(e) => setBroadOnly(e.target.checked)} />
                Broad market representatives only
              </label>
              <label className="flex items-center gap-1 text-[11.5px]"
                     title="A fund a factor is built from is regressed on a factor made of itself">
                <input type="checkbox" checked={hideInputs}
                       onChange={(e) => setHideInputs(e.target.checked)} />
                Hide factor inputs
              </label>
            </div>
            <div className="mt-1.5 max-h-[170px] overflow-auto rounded border border-line bg-white">
              {funds.length === 0
                ? <p className="p-2 text-[11.5px] text-muted">No fund matches the filters.</p>
                : funds.map((f) => (
                  <div key={f.instrument_id}
                       className="flex items-center gap-2 border-t border-lineSoft px-2 py-1 text-[11.5px] first:border-t-0">
                    <span className="w-14 font-mono font-semibold text-navy">{f.instrument_id}</span>
                    <span className="min-w-0 flex-1 truncate">{f.name}</span>
                    <span className="hidden text-muted sm:inline">{f.group_label}</span>
                    {f.factor_inputs.length > 0 && (
                      <Chip tone="warn" title={`Input to ${f.factor_inputs.join(", ")}`}>input</Chip>)}
                    <button className="btn-ghost px-1.5 py-0" disabled={running}
                            onClick={() => addHolding(f.instrument_id, f.name)}>+ hold</button>
                    <button className="btn-ghost px-1.5 py-0" disabled={running}
                            onClick={() => setBenchmark(f.instrument_id)}>benchmark</button>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </Panel>

      {(error || job?.status === "failed") && (
        <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
          {error ?? job?.error}
        </div>
      )}
      {running && (
        <div className="rounded border border-line bg-white p-3 text-[12px] text-muted">
          {job!.messages[job!.messages.length - 1] ?? "Starting"}…
        </div>
      )}

      {stale && (
        <div className="rounded border border-warn/30 bg-warn/5 p-2.5 text-[12px] text-warn">
          The portfolio or the settings have changed since this run; the results below
          are for the portfolio as it was. Run again to update them.
        </div>
      )}
      {res && <div className={"space-y-4" + (stale ? " opacity-60" : "")}><Results res={res} /></div>}
    </div>
  );
}

function Chip({ children, tone, title }: { children: React.ReactNode; tone: "navy" | "warn";
                                          title?: string }) {
  return (
    <span title={title}
          className={`ml-1.5 rounded px-1 py-[1px] text-[9.5px] uppercase tracking-label ${
            tone === "warn" ? "bg-warn/10 text-warn" : "bg-navy/10 text-navy"}`}>
      {children}
    </span>
  );
}

/**
 * The bias statistic is a standard deviation, so one crash block the forecast
 * could not anticipate can carry it (the global multi-asset fund portfolio: the
 * block from 21 February 2020 at z = -17 takes it from 0.91 to 1.69). Name such a
 * block and the statistic without it, beside the formal figure rather than in
 * its place.
 */
function outlier(b: Record<string, any>): string {
  const z = b.bias_worst_z as number | null;
  if (z == null || Math.abs(z) < 4 || b.bias_stat_ex_worst == null) return "";
  return ` · ${num(b.bias_stat_ex_worst, 2)} without the block from ${b.bias_worst_date}`
       + ` (z ${num(z, 1)})`;
}

function Results({ res }: { res: NonNullable<PortfolioJob["result"]> }) {
  const f = res.forecast;
  const bt = res.backtest.portfolio as Record<string, any>;
  const act = res.active;
  const abt = res.backtest.active as Record<string, any> | undefined;
  const band = bt.bias_band as number | null;
  const dm = backtest.benchmark(bt.dm_t_own_ewma, bt.dm_p_own_ewma);
  const t = (label: string, value: string, sub: string, tone: Tone, metric: Metric): Tile =>
    ({ label, value, sub, tone, metric });
  const tiles: Tile[] = [
    t("Predicted volatility", pct(f.vol), `as of ${res.as_of}`, "neutral", PM.predictedVol),
    t("Systematic share", pct(f.systematic_share ?? null, 0),
      `systematic ${pct(f.systematic_vol)} · specific ${pct(f.specific_vol)}`, "neutral",
      PM.systematicShare),
    t("1-day VaR 99%", pct(f.var_99_t ?? f.var_99, 2),
      f.var_99_t != null ? `t, ν = ${num(f.t_df, 1)} · normal ${pct(f.var_99, 2)}` : "normal",
      "neutral", PM.var99),
    ...(act ? [t("Tracking error", pct(act.tracking_error),
                 `vs ${act.benchmark} (vol ${pct(act.benchmark_vol)})`, "neutral",
                 PM.trackingError)] : []),
    t("Bias statistic", num(bt.bias_stat, 3),
      (band != null ? `${res.horizon}-day blocks, band 1 ± ${num(band, 3)}` : "")
        + outlier(bt),
      bt.bias_in_band == null ? "neutral" : bt.bias_in_band ? "good" : "bad",
      PM.biasStatistic),
    t("Daily dispersion", num(bt.z_std, 3), "sd of standardised returns, target 1",
      backtest.zStd(bt.z_std).tone, PM.zStd),
    t("VaR 99% breaches",
      `${bt.exceptions_99_t ?? bt.exceptions_99} / ${num(bt.expected_99_t ?? bt.expected_99, 0)}`,
      `${bt.exceptions_99_t != null ? "fitted t" : "normal"} · Kupiec p ${pval(bt.kupiec_p_99_t ?? bt.kupiec_p_99)}`,
      backtest.kupiec(bt.kupiec_p_99_t ?? bt.kupiec_p_99).tone, PM.kupiec99),
    t("vs own EWMA", num((bt.loss ?? NaN) - (bt.loss_own_ewma ?? NaN), 4),
      `daily loss difference · DM p ${pval(bt.dm_p_own_ewma)}`, dm.tone, PM.portfolioVsEwma),
  ];

  const s = res.series.portfolio;
  const blocks = Object.entries(f.by_block);
  return (
    <>
      <Panel title="Forecast"
             caption={`As of ${res.as_of}. Backtested on ${res.start} to ${res.as_of}: every
               forecast used only data up to its own day. Model: ${res.model.window}-day OLS
               refitted every ${res.model.step} days on the orthogonalised panel, factor
               covariance half-life ${res.cov_halflife} days, specific risk from
               out-of-sample residuals, beta noise removed.`}>
        <MetricStrip>
          {tiles.map((x, i) => (
            <MetricCard key={x.label} label={x.label} value={x.value} sub={x.sub}
                        tone={x.tone} metric={x.metric} align={alignFor(i)} />
          ))}
        </MetricStrip>
        {res.warnings.length > 0 && (
          <ul className="mt-3 space-y-1">
            {res.warnings.map((w, i) => (
              <li key={i} className="text-[11.5px] leading-snug text-warn">— {w}</li>
            ))}
          </ul>
        )}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Predicted versus realised volatility"
               caption={`Predicted is the forecast for the next day; realised is the
                 volatility of the following ${res.horizon} days. They should track.`}>
          <Chart height={300} data={[
            { x: s.dates, y: s.predicted, type: "scatter", mode: "lines", name: "predicted",
              line: { color: PALETTE[0], width: 1.4 } },
            { x: s.dates, y: s.realised, type: "scatter", mode: "lines",
              name: `realised (next ${res.horizon} days)`,
              line: { color: PALETTE[1], width: 1.1 } },
          ]} layout={{ yaxis: { title: "annualised volatility", tickformat: ".0%" } }} />
        </Panel>

        {act && res.series.active && (
          <Panel title={`Tracking error against ${act.benchmark}`}
                 caption={`The same forecast on the active weights: the portfolio less one
                   unit of ${act.benchmark_name}. Backtest bias statistic
                   ${num(abt?.bias_stat, 3)} (band 1 ± ${num(abt?.bias_band, 3)})${
                   abt ? outlier(abt) : ""}.`}>
            <Chart height={300} data={[
              { x: res.series.active.dates, y: res.series.active.predicted, type: "scatter",
                mode: "lines", name: "predicted TE", line: { color: PALETTE[3], width: 1.4 } },
              { x: res.series.active.dates, y: res.series.active.realised, type: "scatter",
                mode: "lines", name: `realised (next ${res.horizon} days)`,
                line: { color: PALETTE[4], width: 1.1 } },
            ]} layout={{ yaxis: { title: "annualised tracking error", tickformat: ".0%" } }} />
          </Panel>
        )}

        <Panel title="Risk by factor block"
               caption="Each block's share of the portfolio's variance; the rest is specific risk. Negative is a block the portfolio is hedged against.">
          <div className="mb-1 flex items-center text-[11px] text-muted">
            Definition<InfoTip metric={PM.blockShare} label="Risk by factor block" />
          </div>
          <Chart height={260} numericX data={[{
            type: "bar", orientation: "h",
            y: [...blocks.map(([b]) => blockStyle(b).label), "Specific"].reverse(),
            x: [...blocks.map(([, v]) => v),
                1 - blocks.reduce((t, [, v]) => t + v, 0)].reverse(),
            marker: { color: [...blocks.map(([b]) => blockStyle(b).colour), "#9AA0AE"].reverse() },
            hovertemplate: "%{y}: %{x:.1%}<extra></extra>",
          }]} layout={{ xaxis: { tickformat: ".0%", title: "share of variance" },
                        margin: { l: 110 }, showlegend: false }} />
        </Panel>

        <Panel title="Largest factor exposures"
               caption="The portfolio's betas to individual factors and each one's share of its variance.">
          <table className="w-full border-collapse">
            <thead><tr>
              <th className="th text-left">Factor</th>
              <th className="th"><span className="inline-flex items-center">Exposure
                <InfoTip metric={PM.exposure} /></span></th>
              <th className="th"><span className="inline-flex items-center">Share of variance
                <InfoTip metric={PM.blockShare} /></span></th>
            </tr></thead>
            <tbody>
              {f.factors.map((x) => (
                <tr key={x.factor_id} className="border-t border-lineSoft">
                  <td className="cell text-left font-sans">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-[2px]"
                          style={{ background: blockStyle(x.block_id).colour }} />
                    <span className="font-mono">{x.factor_id}</span>{" "}
                    <span className="text-muted">{x.name}</span>
                  </td>
                  <td className="cell">{num(x.exposure, 2)}</td>
                  <td className="cell">{pct(x.share, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      </div>

      <Panel title="Risk by holding"
             caption="Where the portfolio's variance comes from. Contributions sum to 100%; a negative one hedges the rest.">
        <div className="max-h-[360px] overflow-auto">
          <table className="w-full border-collapse">
            <thead className="sticky top-0 bg-panel"><tr>
              <th className="th text-left">Holding</th>
              <th className="th text-left">Name</th>
              <th className="th text-left">Type</th>
              <th className="th">Weight</th>
              <th className="th"><span className="inline-flex items-center">Standalone vol
                <InfoTip metric={PM.standaloneVol} /></span></th>
              <th className="th"><span className="inline-flex items-center">Contribution
                <InfoTip metric={PM.contribution} /></span></th>
              <th className="th"><span className="inline-flex items-center">Beta to portfolio
                <InfoTip metric={PM.betaToPortfolio} /></span></th>
              <th className="th">Factors</th>
            </tr></thead>
            <tbody>
              {res.holdings.map((h) => (
                <tr key={h.instrument_id} className="border-t border-lineSoft">
                  <td className="cell text-left font-mono">{h.instrument_id}</td>
                  <td className="cell max-w-[240px] truncate text-left font-sans">{h.name ?? ""}</td>
                  <td className="cell text-left font-sans text-[11px] text-muted">
                    {h.kind}{h.group ? ` · ${h.group}` : ""}</td>
                  <td className="cell">{pct(h.weight, 1)}</td>
                  <td className="cell">{pct(h.standalone_vol, 1)}</td>
                  <td className="cell">
                    <span className="inline-flex items-center gap-2">
                      <span className="inline-block h-[6px] rounded-sm bg-navy/60"
                            style={{ width: `${Math.min(80, Math.abs(h.contribution ?? 0) * 160)}px` }} />
                      {pct(h.contribution, 1)}
                    </span>
                  </td>
                  <td className="cell">{num(h.beta_to_portfolio, 2)}</td>
                  <td className="cell text-muted">{h.n_factors}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
