"use client";

/**
 * Raw Explorer: the data before the model touches it.
 *
 * The Factor Explorer describes the model, so it shows orthogonalised factors —
 * the series the regression, the covariance and the risk forecast all consume.
 * This page answers the other question, and keeps it on its own screen so the two
 * can never be confused: what does the underlying series actually do?
 *
 * Two kinds sit in one list. A factor here is its own excess return before the
 * block hierarchy is removed; an instrument is a single input exactly as ingested,
 * with no construction applied at all.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, ChartSkeleton, PALETTE, Panel, Stat, VerdictBadge }
  from "@/components/Chart";
import { num, pct, pval } from "@/lib/api";
import { blockStyle } from "@/lib/blocks";
import { episodeLayout } from "@/lib/episodes";
import { assess, bySign } from "@/lib/verdict";
import { usePublishSnapshot } from "@/lib/chat-context";

type Kind = "factor" | "instrument";

interface CatalogEntry {
  id: string;
  name: string | null;
  group_id: string;
  group_name: string;
  n_obs: number | null;
  sd_ann: number | null;
  last_date?: string | null;
  n_orth?: number | null;
  is_live?: boolean;
  is_total_return?: boolean;
  currency?: string;
  status?: "live" | "discontinued" | "not_refreshed";
  days_behind?: number | null;
}

export default function RawPage() {
  const [catalog, setCatalog] = useState<{ factors: CatalogEntry[]; instruments: CatalogEntry[] } | null>(null);
  const [kind, setKind] = useState<Kind>("factor");
  const [selected, setSelected] = useState<string>("");
  const [filter, setFilter] = useState("");
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/raw/catalog")
      .then((r) => r.json())
      .then((c) => {
        setCatalog(c);
        if (c.factors?.length) setSelected(c.factors[0].id);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setError(null);
    setData(null);
    fetch(`/api/raw/${kind}/${encodeURIComponent(selected)}`)
      .then((r) => (r.ok ? r.json() : r.json().then((b) => Promise.reject(new Error(b.detail)))))
      .then(setData)
      .catch((e) => setError(String(e.message ?? e)));
  }, [kind, selected]);

  const list = useMemo(() => {
    const all = (kind === "factor" ? catalog?.factors : catalog?.instruments) ?? [];
    const q = filter.trim().toLowerCase();
    const matched = q
      ? all.filter((e) => e.id.toLowerCase().includes(q) ||
                          (e.name ?? "").toLowerCase().includes(q) ||
                          e.group_name.toLowerCase().includes(q))
      : all;
    const grouped: Record<string, CatalogEntry[]> = {};
    matched.forEach((e) => (grouped[e.group_name] ??= []).push(e));
    return grouped;
  }, [catalog, kind, filter]);

  const bands = useMemo(
    () => episodeLayout(data?.dates?.[0], data?.dates?.[data.dates.length - 1]),
    [data]
  );

  usePublishSnapshot(
    data
      ? {
          view: "raw explorer",
          kind: data.kind, id: data.id, meta: data.meta,
          stats: data.stats,
          diagnostics: {
            verdict: data.diagnostics?.verdict,
            reason: data.diagnostics?.verdict_reason,
            adf_p: data.diagnostics?.adf_p, kpss_p: data.diagnostics?.kpss_p,
            vr5: data.diagnostics?.vr5, ac1: data.diagnostics?.ac1,
            zero_return_share: data.diagnostics?.zero_return_share,
          },
          note: "These figures are the raw series before orthogonalisation. The "
              + "Factor Explorer shows the orthogonalised factor instead.",
        }
      : null
  );

  const d = data?.diagnostics;
  const s = data?.stats;
  const accent = kind === "factor"
    ? blockStyle(catalog?.factors.find((f) => f.id === selected)?.group_id).colour
    : PALETTE[1];

  return (
    <div className="grid gap-4 lg:grid-cols-[250px_1fr]">
      <aside className="space-y-3">
        <Panel title="Series">
          <div className="mb-2 flex overflow-hidden rounded border border-line">
            {(["factor", "instrument"] as const).map((k) => (
              <button
                key={k}
                onClick={() => {
                  setKind(k);
                  const first = (k === "factor" ? catalog?.factors : catalog?.instruments)?.[0];
                  if (first) setSelected(first.id);
                }}
                className={`flex-1 px-2 py-1 text-[11px] capitalize transition ${
                  kind === k ? "bg-navy text-white" : "bg-white text-muted hover:text-navy"
                }`}
              >
                {k === "factor" ? "Factors" : "Instruments"}
              </button>
            ))}
          </div>
          <input
            className="field"
            placeholder="Filter…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <p className="mt-2 text-[10px] leading-tight text-muted">
            The figure beside each name is its annualised volatility.
          </p>
          {/*
            stale and ended are liveness markers on an ingested series, and a factor
            has no such state: it is rebuilt each night from whatever inputs exist.
            Explaining them above a list that can never show them sent a reader
            hunting for markers that were not missing, they were inapplicable.
          */}
          {kind === "instrument" ? (
            <p className="mt-1.5 text-[10px] leading-tight text-muted">
              Single inputs exactly as ingested — no construction, no
              orthogonalisation. <b className="text-warn">stale</b> means it is
              still trading but outside the nightly ingest;{" "}
              <b className="text-fail">ended</b> means the provider stopped
              publishing it.
            </p>
          ) : (
            <p className="mt-1.5 text-[10px] leading-tight text-muted">
              Each factor&rsquo;s own excess return, before the block hierarchy is
              removed. Names here drop any &ldquo;ex-&rdquo; qualifier, which
              describes the residual rather than this series.
            </p>
          )}
        </Panel>

        <div className="max-h-[66vh] overflow-auto rounded border border-line bg-panel">
          {Object.entries(list).map(([group, entries]) => (
            <div key={group}>
              <div className="sticky top-0 z-10 bg-lineSoft px-2 py-1 text-2xs
                              font-semibold uppercase tracking-label text-muted">
                {group}
              </div>
              {entries.map((e) => (
                <button
                  key={e.id}
                  onClick={() => setSelected(e.id)}
                  title={e.name ?? e.id}
                  className={`flex w-full items-center justify-between gap-2 px-2 py-1
                              text-left text-[12px] transition
                    ${e.id === selected ? "bg-navy text-white" : "hover:bg-lineSoft"}`}
                >
                  <span className="truncate font-mono">{e.id}</span>
                  <span
                    title="Annualised volatility over the full history"
                    className={`shrink-0 font-mono text-[10px] tabular-nums
                      ${e.id === selected ? "text-white/70" : "text-muted"}`}
                  >
                    {e.sd_ann ? pct(e.sd_ann, 0) : "—"}
                  </span>
                  {e.status === "discontinued" && (
                    <span
                      className="shrink-0 text-2xs text-fail"
                      title={`The provider stopped publishing this: last observation ${e.last_date ?? "?"}, ${e.days_behind} days behind the panel. It is excluded from factor construction.`}
                    >
                      ended
                    </span>
                  )}
                  {e.status === "not_refreshed" && (
                    <span
                      className="shrink-0 text-2xs text-warn"
                      title={`Still trading, but outside the nightly ingest, which covers the factor universe only. Last mirrored from the warehouse on ${e.last_date ?? "?"}, ${e.days_behind} days behind.`}
                    >
                      stale
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      </aside>

      <div className="space-y-4">
        {error && (
          <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
            {error}
          </div>
        )}

        <div className="rounded border border-warn/30 bg-warn/5 px-4 py-2 text-[12px]">
          <b>This is the raw series.</b> Factors here are shown before block-hierarchy
          orthogonalisation, and instruments before any construction. The model does
          not use these numbers — the Factor Explorer shows what it does use, and the
          two will differ, often by a lot.
        </div>

        {data && (
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel index={0} title="Return series"
                   caption="The series as stored, with no transformation beyond the log return itself.">
              <Chart
                height={188}
                episodes={bands}
                data={[
                  { x: data.dates, y: data.returns, type: "scatter", mode: "lines",
                    name: "daily return", line: { color: accent, width: 0.7 },
                    hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>" },
                  ...[1, -1].map((sign) => ({
                    x: [data.dates[0], data.dates[data.dates.length - 1]],
                    y: Array(2).fill(sign * 2 * s.vol_ann / Math.sqrt(252)),
                    type: "scatter" as const, mode: "lines" as const,
                    name: "±2 sd", showlegend: sign === 1,
                    line: { color: "#B0553F", width: 1, dash: "dot" as const },
                    hoverinfo: "skip" as const,
                  })),
                ]}
                layout={{ yaxis: { title: "daily return", tickformat: ".1%" },
                          margin: { l: 56, r: 14, t: 8, b: 34 }, legend: { y: -0.3 } }}
              />
            </Panel>

            <Panel
              index={1}
              title={`${data.id} — ${data.meta?.name ?? ""}`}
              caption={
                <>
                  {kind === "factor" ? (
                    <>
                      {/*
                        No mention of what the hierarchy removes. This page plots the
                        series before any of that, and naming the residualisation
                        target beside it invited exactly the reading the header used
                        to make explicit — that this was already ex-global when it
                        is not. The Factor Explorer is where that belongs.
                      */}
                      Log excess return over cash, as constructed.
                      {data.meta?.block_name && <> {data.meta.block_name} block.</>}
                    </>
                  ) : (
                    <>
                      {data.meta?.asset_class} · {data.meta?.currency}
                      {data.meta?.is_total_return === false && (
                        <b className="text-warn"> · price return, no dividends</b>
                      )}
                      {data.meta?.is_live === false && (
                        <b className="text-fail"> · no longer updating</b>
                      )}
                      {data.meta?.notes && (
                        <div className="mt-1 italic">{data.meta.notes}</div>
                      )}
                    </>
                  )}
                </>
              }
            >
              <div className="mb-3 flex flex-wrap items-end gap-x-8 gap-y-2
                              border-b border-lineSoft pb-3">
                {/*
                  Only the figures whose sign means something to an investor are
                  coloured. Volatility, skew, kurtosis, drawdown and the tail
                  measures below are shape, not verdict: there is no good or bad
                  volatility for a factor, and a drawdown is negative by
                  definition, so a permanent red would carry no information.
                */}
                <Stat label="Ann. volatility" size="hero"
                      animate={s.vol_ann} format={(v) => pct(v)} />
                <Stat label="Ann. return" size="hero"
                      animate={s.mean_ann} format={(v) => pct(v)}
                      tone={bySign(s.mean_ann)} />
                <Stat label="Sharpe" size="hero"
                      animate={s.sharpe} format={(v) => num(v)}
                      tone={bySign(s.sharpe)} />
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                <Stat label="Skew" value={num(s.skew)} />
                <Stat label="Excess kurt." value={num(s.excess_kurtosis)} />
                <Stat label="Max drawdown" value={pct(s.max_drawdown)} />
                <Stat label="Total (compounded)" value={pct(s.total_compounded, 0)}
                      tone={bySign(s.total_compounded)} />
                <Stat label="Hit rate" value={pct(s.hit_rate)}
                      hint="share of up days" />
                <Stat label="VaR 95% (1d)" value={pct(s.var95_daily)} />
                <Stat label="ES 95% (1d)" value={pct(s.es95_daily)} />
                <Stat label="Observations" value={s.n_obs?.toLocaleString()} />
              </div>
            </Panel>

            {/*
              Where the numbers came from. An instrument id does not say whether it
              was fetched from Yahoo this morning or mirrored out of a warehouse
              table refreshed in June, and neither does a factor id say which
              instruments it reads. Both matter before trusting a figure, and the
              history span matters most: a factor is only as long as its shortest
              input, which is how liq_funding starts in 2018 while every other
              liquidity series goes back to 2000.
            */}
            <Panel index={2} title="Where this comes from">
              {kind === "instrument" ? (
                <div className="grid gap-3 sm:grid-cols-2">
                  <Fact label="Provider" value={data.meta?.source} />
                  <Fact label="Series at the provider"
                        value={data.meta?.label ?? data.id} mono />
                  <Fact label="History held"
                        value={span(data.dates)} />
                  <Fact label="Observations"
                        value={data.stats?.n_obs?.toLocaleString()} />
                  <Fact label="Asset class" value={data.meta?.asset_class} />
                  <Fact label="Currency" value={data.meta?.currency} />
                </div>
              ) : (
                <>
                  <div className="mb-3 grid gap-3 border-b border-lineSoft pb-3 sm:grid-cols-3">
                    <Fact label="Construction"
                          value={data.meta?.construction?.method} mono />
                    <Fact label="History held" value={span(data.dates)} />
                    <Fact label="Observations"
                          value={data.stats?.n_obs?.toLocaleString()} />
                  </div>
                  {(data.meta?.sources ?? []).length > 0 ? (
                    <div className="overflow-auto" style={{ maxHeight: 260 }}>
                      <table className="w-full border-collapse">
                        <thead>
                          <tr>
                            <th className="th">Input</th>
                            <th className="th">At the provider</th>
                            <th className="th">Source</th>
                            <th className="th">History</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.meta.sources.map((src: any) => (
                            <tr key={src.id} className="border-t border-lineSoft">
                              <td className="cell">{src.id}</td>
                              <td className="cell" title={src.name ?? ""}>
                                {src.label ?? src.id}
                              </td>
                              <td className="cell font-sans text-muted">
                                {src.source}
                              </td>
                              <td className="cell text-muted">
                                {src.first_date
                                  ? src.first_date + " to " + src.last_date
                                  : "no data"}
                                {src.n_obs
                                  ? " \u00b7 " + Number(src.n_obs).toLocaleString()
                                  : ""}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="py-4 text-center text-[12px] text-muted">
                      No named inputs recorded for this construction.
                    </p>
                  )}
                  {data.meta?.construction?.note && (
                    <p className="mt-2 text-[11px] italic text-muted">
                      {data.meta.construction.note}
                    </p>
                  )}
                </>
              )}
            </Panel>

            <Panel index={3} title="Compounded return"
                   caption="exp(Σ log r) − 1: what holding this series would actually have returned.">
              <Chart
                height={280} episodes={bands}
                data={[{ x: data.dates, y: data.compounded, type: "scatter",
                         mode: "lines", line: { color: accent, width: 1.5 },
                         hovertemplate: "%{x|%Y-%m-%d}<br>%{y:+.1%}<extra></extra>" }]}
                layout={{ yaxis: { title: "compounded", tickformat: ".0%" },
                          showlegend: false }}
              />
            </Panel>

            <Panel index={4} title="Rolling volatility"
                   caption="Annualised, at 21, 63 and 252 days. Shaded bands mark well-known market episodes — editorial context, not a model output.">
              <Chart
                height={280} episodes={bands}
                data={data.rolling_windows.map((w: number, i: number) => ({
                  x: data.dates, y: data.rolling[String(w)], type: "scatter",
                  mode: "lines", name: `${w}d`,
                  line: { color: PALETTE[i], width: 1.2 },
                }))}
                layout={{ yaxis: { title: "annualised vol", tickformat: ".0%" } }}
              />
            </Panel>

            <Panel index={5} title="Return distribution"
                   caption={`${data.distribution.n_bins} bins by the Freedman-Diaconis rule; Student-t fit has ${num(data.distribution.t_df, 1)} degrees of freedom. ${data.distribution.n_outside} observations lie outside the drawn range.`}>
              <Chart
                height={280} numericX
                data={[
                  { x: data.distribution.bin_centres, y: data.distribution.density,
                    type: "bar", name: "empirical", width: data.distribution.bin_width,
                    marker: { color: accent, opacity: 0.35, line: { width: 0 } } },
                  { x: data.distribution.grid, y: data.distribution.kde,
                    type: "scatter", mode: "lines", name: "empirical (kernel)",
                    line: { color: accent, width: 2, shape: "spline" } },
                  { x: data.distribution.grid, y: data.distribution.normal_pdf,
                    type: "scatter", mode: "lines", name: "normal",
                    line: { color: PALETTE[1], width: 1.6, shape: "spline" } },
                  { x: data.distribution.grid, y: data.distribution.t_pdf,
                    type: "scatter", mode: "lines",
                    name: `student-t (df ${num(data.distribution.t_df, 1)})`,
                    line: { color: PALETTE[2], width: 1.6, dash: "dot", shape: "spline" } },
                ]}
                layout={{ bargap: 0.02,
                          xaxis: { title: "daily return", tickformat: ".1%",
                                   range: [data.distribution.lo, data.distribution.hi] },
                          yaxis: { title: "density", rangemode: "tozero" } }}
              />
            </Panel>

            <Panel index={6} title="Autocorrelation"
                   caption="Returns versus squared returns. Autocorrelation in returns is a stale-pricing warning; in squared returns it is volatility clustering, which is expected.">
              <Chart
                height={280} numericX
                data={[
                  { x: data.acf.lags, y: data.acf.returns, type: "bar",
                    name: "returns", marker: { color: PALETTE[0] } },
                  { x: data.acf.lags, y: data.acf.squared, type: "bar",
                    name: "squared returns", marker: { color: PALETTE[1], opacity: 0.7 } },
                  ...[1, -1].map((sign) => ({
                    x: [0, 30], y: Array(2).fill(sign * data.acf.confidence_band),
                    type: "scatter" as const, mode: "lines" as const,
                    name: "95% band", showlegend: sign === 1,
                    line: { color: "#9AA0AE", width: 1, dash: "dash" as const },
                  })),
                ]}
                layout={{ xaxis: { title: "lag" }, yaxis: { title: "autocorrelation" },
                          hovermode: "closest" }}
              />
            </Panel>
          </div>
        )}

        {!data && !error && (
          <div className="grid gap-4 xl:grid-cols-2">
            {[0, 1, 2, 3].map((i) => (
              <Panel key={i} index={i} title="Loading…">
                <ChartSkeleton height={240} />
              </Panel>
            ))}
          </div>
        )}

        {d && (
          <Panel
            index={7}
            title="Stationarity battery"
            caption="Computed live on the series above, not read from the stored factor diagnostics — those are for the orthogonalised factors, and a verdict for a different series would be worse than none."
            actions={<VerdictBadge verdict={d.verdict} showLabel title={d.verdict_reason} />}
          >
            {d.verdict_reason && (
              <p className="mb-3 text-[12px]">{d.verdict_reason}</p>
            )}
            {/*
              Colour is a claim about the number, so it comes from lib/verdict.ts
              rather than from a rule written here — the Factor Explorer runs the
              same battery on the orthogonalised series, and the two must not
              disagree about which p-value is the good one.
            */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
              {([
                ["ADF", `p = ${pval(d.adf_p)}`, assess.adf(d.adf_p), "H0: unit root"],
                ["KPSS", `p = ${pval(d.kpss_p)}`, assess.kpss(d.kpss_p), "H0: stationary"],
                ["Phillips-Perron", `p = ${pval(d.pp_p)}`, assess.pp(d.pp_p),
                 "HAC-robust ADF"],
                ["Ljung-Box (10)", `p = ${pval(d.lb10_p)}`, assess.ljungBox(d.lb10_p),
                 "H0: no autocorrelation"],
                ["ARCH-LM", `p = ${pval(d.arch_lm_p)}`, assess.archLm(d.arch_lm_p),
                 "recorded, never gated"],
                ["Jarque-Bera", `p = ${pval(d.jb_p)}`, assess.jarqueBera(d.jb_p),
                 "informational"],
                ["VR (2)", num(d.vr2), assess.varianceRatio(d.vr2), "1 = random walk"],
                ["VR (5)", num(d.vr5), assess.varianceRatio(d.vr5), "1 = random walk"],
                ["VR (10)", num(d.vr10), assess.varianceRatio(d.vr10), "1 = random walk"],
                ["AC(1)", num(d.ac1, 3), assess.autocorrelation(d.ac1),
                 "first-order autocorrelation"],
                ["Zero returns", pct(d.zero_return_share, 1),
                 assess.zeroReturns(d.zero_return_share), "illiquidity check"],
              ] as const).map(([label, value, a, what]) => (
                <Stat key={label} label={label} value={value} tone={a.tone}
                      hint={`${what} — ${a.reason}`} />
              ))}
              <Stat label="Observations" value={d.n_obs?.toLocaleString()} />
            </div>
            {d.za_break_date && (
              <div className="mt-3 rounded border border-warn/30 bg-warn/5 px-3 py-2
                              text-[12px] text-warn">
                Zivot-Andrews locates a structural break at <b>{d.za_break_date}</b>.
              </div>
            )}
          </Panel>
        )}
      </div>
    </div>
  );
}


/** A labelled fact in the provenance panel. */
function Fact({ label, value, mono }: {
  label: string;
  value?: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="text-2xs uppercase tracking-label text-muted">{label}</div>
      <div className={"text-[12px] text-navy " + (mono ? "font-mono" : "")}>
        {value ?? "—"}
      </div>
    </div>
  );
}

/**
 * The span actually held, taken from the returned dates rather than the registry.
 *
 * A registry first_date is when the provider's series begins; this is the first day
 * on which a *return* exists here, which is what the charts above are drawn from.
 * Where the two differ, the one on screen is the honest one.
 */
function span(dates?: string[]): string {
  if (!dates?.length) return "—";
  const years = (new Date(dates[dates.length - 1]).getTime()
                 - new Date(dates[0]).getTime()) / (365.25 * 24 * 3600 * 1000);
  return dates[0] + " to " + dates[dates.length - 1]
       + " (" + years.toFixed(1) + " years)";
}
