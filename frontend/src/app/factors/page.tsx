"use client";

/**
 * Factor Explorer: one factor at a time — return path, risk through time,
 * distribution, autocorrelation, and the full stationarity battery.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, ChartSkeleton, PALETTE, Panel, Skeleton, Stat, VerdictBadge }
  from "@/components/Chart";
import { Sparkline } from "@/components/Sparkline";
import { api, FactorMeta, num, pct, pval } from "@/lib/api";
import { blockColour, blockStyle } from "@/lib/blocks";
import { episodeLayout } from "@/lib/episodes";
import { usePublishSnapshot } from "@/lib/chat-context";

export default function FactorsPage() {
  const [factors, setFactors] = useState<FactorMeta[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [start, setStart] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const [series, setSeries] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [risk, setRisk] = useState<any>(null);
  const [hist, setHist] = useState<any>(null);
  const [qq, setQQ] = useState<any>(null);
  const [acf, setACF] = useState<any>(null);
  const [diag, setDiag] = useState<any>(null);
  const [ref, setRef] = useState<any>(null);
  const [logY, setLogY] = useState(false);
  const [sparks, setSparks] = useState<Record<string, number[]>>({});

  useEffect(() => {
    api.factors()
      .then((f) => {
        setFactors(f);
        if (f.length) setSelected(f[0].factor_id);
      })
      .catch((e) => setError(String(e.message ?? e)));
    // One request for all forty sparklines; the sidebar would otherwise fire forty.
    api.factorSparklines().then((d) => setSparks(d.series)).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selected) return;
    const p = start ? { start } : undefined;
    setError(null);
    Promise.all([
      api.factorSeries(selected, p),
      api.factorStats(selected, p),
      api.factorRollingRisk(selected, "21,63,252", p),
      api.factorHistogram(selected),
      api.factorQQ(selected),
      api.factorACF(selected),
      api.factorDiagnostics(selected).catch(() => null),
      api.factorReference(selected).catch(() => null),
    ])
      .then(([s, st, r, h, q, a, d, rf]) => {
        setSeries(s); setStats(st); setRisk(r); setHist(h);
        setQQ(q); setACF(a); setDiag(d); setRef(rf);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [selected, start]);

  const meta = useMemo(
    () => factors.find((f) => f.factor_id === selected),
    [factors, selected]
  );
  const byBlock = useMemo(() => {
    const m: Record<string, FactorMeta[]> = {};
    factors.forEach((f) => (m[f.block_id] ??= []).push(f));
    return m;
  }, [factors]);

  const window252 = diag?.windows?.find((w: any) => w.window_days === 252)
    ?? diag?.windows?.[0];

  // What the assistant is allowed to quote: exactly the figures rendered below.
  usePublishSnapshot(
    selected && stats
      ? {
          factor_id: selected,
          name: meta?.name,
          block: meta?.block_id,
          construction: meta?.construction,
          orthogonalised_against: meta?.orthogonalize_against,
          stats,
          diagnostics: window252
            ? {
                window_days: window252.window_days,
                verdict: window252.verdict,
                verdict_reason: window252.verdict_reason,
                flags: window252.flags,
                adf_p: window252.adf_p,
                kpss_p: window252.kpss_p,
                vr5: window252.vr5,
                ac1: window252.ac1,
                zero_return_share: window252.zero_return_share,
              }
            : undefined,
          distribution: hist
            ? { n_bins: hist.n_bins, t_df: hist.t_df, n_outside: hist.n_outside,
                jarque_bera_p: hist.jarque_bera_p, sd: hist.sd }
            : undefined,
        }
      : null
  );

  // Crisis bands, derived from whatever span the selected factor actually covers.
  const bands = useMemo(
    () => episodeLayout(series?.dates?.[0], series?.dates?.[series.dates.length - 1]),
    [series]
  );
  const accent = blockColour(meta?.block_id);

  return (
    <div className="grid gap-4 lg:grid-cols-[240px_1fr]">
      {/* factor picker */}
      <aside className="space-y-3">
        <Panel title="Sample">
          <label className="label">Start date</label>
          <input
            type="date"
            className="field mt-1"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
          {start && (
            <button className="btn-ghost mt-2 w-full" onClick={() => setStart("")}>
              Full history
            </button>
          )}
        </Panel>

        <div className="max-h-[74vh] overflow-auto rounded border border-line bg-panel">
          {Object.entries(byBlock).map(([block, list]) => {
            const bs = blockStyle(block);
            return (
              <div key={block}>
                <div
                  className="sticky top-0 z-10 flex items-center gap-1.5 px-2 py-1
                             text-2xs font-semibold uppercase tracking-label text-muted"
                  style={{ background: bs.tint, backdropFilter: "blur(2px)" }}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-[1px]"
                    style={{ background: bs.colour }}
                  />
                  {bs.label}
                </div>
                {list.map((f) => {
                  const active = f.factor_id === selected;
                  return (
                    <button
                      key={f.factor_id}
                      onClick={() => setSelected(f.factor_id)}
                      title={f.name}
                      className={`flex w-full items-center gap-2 border-l-[3px] py-1 pl-2 pr-2
                                  text-left text-[12px] transition
                        ${active ? "bg-navy text-white" : "hover:bg-lineSoft"}`}
                      style={{ borderLeftColor: active ? bs.colour : "transparent" }}
                    >
                      <span className="flex-1 truncate font-mono">{f.factor_id}</span>
                      {sparks[f.factor_id] ? (
                        <Sparkline
                          values={sparks[f.factor_id]}
                          colour={active ? "#FFFFFF" : bs.colour}
                          width={46}
                          height={14}
                        />
                      ) : (
                        <Skeleton width={46} height={10} />
                      )}
                      <VerdictBadge
                        verdict={f.verdict}
                        title={f.verdict_reason ?? undefined}
                      />
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      </aside>

      <div className="space-y-4">
        {error && (
          <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
            {error}
          </div>
        )}

        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Panel
            title="Raw return series"
            caption={
              <>
                The daily series the model actually consumes — everything else on
                this page is derived from it. Clustered spikes are volatility
                clustering, and a visible shift in the width of the band is the kind
                of structural break the stationarity battery tests for.
              </>
            }
          >
            {!series && <ChartSkeleton height={188} />}
            {series && stats && (
              <Chart
                height={188}
                episodes={bands}
                data={[
                  {
                    x: series.dates, y: series.returns, type: "scatter",
                    mode: "lines", name: "daily return",
                    line: { color: accent, width: 0.7 },
                    hovertemplate: "%{x|%Y-%m-%d}<br>%{y:.2%}<extra></extra>",
                  },
                  // A two-standard-deviation reference, so the eye can judge how
                  // often the series leaves the band. For a normal series that is
                  // about one day in twenty; a fat-tailed one leaves it far more
                  // dramatically than it leaves it often.
                  ...[1, -1].map((sign) => ({
                    x: [series.dates[0], series.dates[series.dates.length - 1]],
                    y: [sign * 2 * stats.vol_ann / Math.sqrt(252),
                        sign * 2 * stats.vol_ann / Math.sqrt(252)],
                    type: "scatter" as const, mode: "lines" as const,
                    name: "±2 sd", showlegend: sign === 1,
                    line: { color: "#B0553F", width: 1, dash: "dot" as const },
                    hoverinfo: "skip" as const,
                  })),
                ]}
                layout={{
                  yaxis: { title: "daily return", tickformat: ".1%", zeroline: true },
                  margin: { l: 56, r: 14, t: 8, b: 34 },
                  legend: { y: -0.3 },
                }}
              />
            )}
          </Panel>

          {meta && (
            <Panel
              title={`${meta.factor_id} — ${meta.name}`}
              caption={
                <>
                  <span className="font-mono">{meta.construction?.method}</span>
                  {meta.orthogonalize_against?.length > 0 && (
                    <>
                      {" · orthogonalised against "}
                      <span className="font-mono">
                        {meta.orthogonalize_against.join(", ")}
                      </span>
                    </>
                  )}
                  {meta.construction?.note && (
                    <div className="mt-1 italic">{meta.construction.note}</div>
                  )}
                </>
              }
            >
              {stats && (
                <>
                  <div className="mb-3 flex flex-wrap items-end gap-x-8 gap-y-2
                                  border-b border-lineSoft pb-3">
                    <Stat
                      label="Ann. volatility" size="hero"
                      animate={stats.vol_ann} format={(v) => pct(v)}
                    />
                    <Stat
                      label="Ann. return" size="hero"
                      animate={stats.mean_ann} format={(v) => pct(v)}
                      tone={stats.mean_ann >= 0 ? "neutral" : "bad"}
                    />
                    <Stat
                      label="Sharpe" size="hero"
                      animate={stats.sharpe} format={(v) => num(v)}
                    />
                  </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                  <Stat label="Ann. return" value={pct(stats.mean_ann)} />
                  <Stat label="Ann. vol" value={pct(stats.vol_ann)} />
                  <Stat label="Sharpe" value={num(stats.sharpe)} />
                  <Stat label="Skew" value={num(stats.skew)} />
                  <Stat label="Excess kurt." value={num(stats.excess_kurtosis)} />
                  <Stat label="Max drawdown" value={pct(stats.max_drawdown)} tone="bad" />
                  <Stat label="Hit rate" value={pct(stats.hit_rate)} />
                  <Stat label="VaR 95% (1d)" value={pct(stats.var95_daily)} />
                  <Stat label="ES 95% (1d)" value={pct(stats.es95_daily)} />
                  <Stat label="Observations" value={stats.n_obs?.toLocaleString()} />
                  <Stat label="From" value={stats.first_date} />
                  <Stat label="To" value={stats.last_date} />
                </div>
                </>
              )}
            </Panel>
          )}
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <Panel
            title="Cumulative return"
            caption="Sum of daily log excess returns, not compounded simple returns — the factors are log series and the two diverge materially over twenty years."
          >
            {!series && <ChartSkeleton height={280} />}
            {series && (
              <Chart
                height={280}
                episodes={bands}
                data={[{
                  x: series.dates, y: series.cumulative, type: "scatter", mode: "lines",
                  name: selected, line: { color: accent, width: 1.5 },
                }]}
                layout={{ yaxis: { title: "cumulative log return", tickformat: ".0%" },
                          showlegend: false }}
              />
            )}
          </Panel>

          <Panel
            title="Rolling volatility"
            caption="Annualised, at 21, 63 and 252 days. The short window shows the regime, the long one the level; a gap between them is a regime change in progress. Shaded bands mark well-known market episodes — editorial context, not a model output."
          >
            {!risk && <ChartSkeleton height={280} />}
            {risk && (
              <Chart
                height={280}
                episodes={bands}
                data={Object.entries(risk.series).map(([w, v], i) => ({
                  x: risk.dates, y: v as number[], type: "scatter", mode: "lines",
                  name: `${w}d`, line: { color: PALETTE[i], width: 1.2 },
                }))}
                layout={{ yaxis: { title: "annualised vol", tickformat: ".0%" } }}
              />
            )}
          </Panel>

          <DistributionPanel hist={hist} logY={logY} setLogY={setLogY} />

          <Panel
            title="Quantile-quantile"
            caption="Points on the 45-degree line match the reference distribution. Departure at the ends is tail risk the normal assumption misses."
          >
            {qq && (
              <Chart
                height={280}
                numericX
                data={[
                  { x: qq.normal_quantiles, y: qq.sample_quantiles, type: "scatter",
                    mode: "markers", name: "vs normal",
                    marker: { color: PALETTE[0], size: 4, opacity: 0.6 } },
                  { x: qq.t_quantiles, y: qq.sample_quantiles, type: "scatter",
                    mode: "markers", name: "vs student-t",
                    marker: { color: PALETTE[2], size: 4, opacity: 0.6 } },
                  { x: qq.normal_quantiles, y: qq.normal_quantiles, type: "scatter",
                    mode: "lines", name: "45°",
                    line: { color: "#9AA0AE", width: 1, dash: "dash" } },
                ]}
                layout={{ xaxis: { title: "theoretical quantile", tickformat: ".1%" },
                          yaxis: { title: "sample quantile", tickformat: ".1%" },
                          hovermode: "closest" }}
              />
            )}
          </Panel>

          <Panel
            title="Autocorrelation"
            caption="Returns versus squared returns. Autocorrelation in returns is a stale-pricing warning; in squared returns it is volatility clustering, which is normal and argues for HAC errors rather than against the factor."
          >
            {acf && (
              <Chart
                height={280}
                numericX
                data={[
                  { x: acf.lags, y: acf.acf, type: "bar", name: "returns",
                    marker: { color: PALETTE[0] } },
                  { x: acf.lags, y: acf.acf_squared, type: "bar", name: "squared returns",
                    marker: { color: PALETTE[1], opacity: 0.7 } },
                  { x: [0, acf.lags.length], y: [acf.confidence_band, acf.confidence_band],
                    type: "scatter", mode: "lines", name: "95% band",
                    line: { color: "#9AA0AE", width: 1, dash: "dash" } },
                  { x: [0, acf.lags.length], y: [-acf.confidence_band, -acf.confidence_band],
                    type: "scatter", mode: "lines", showlegend: false,
                    line: { color: "#9AA0AE", width: 1, dash: "dash" } },
                ]}
                layout={{ xaxis: { title: "lag" }, yaxis: { title: "autocorrelation" },
                          hovermode: "closest" }}
              />
            )}
          </Panel>

          <Panel
            title="Published-factor comparison"
            caption="Correlation with Fama-French and AQR over the overlapping sample. These lag by one to two months and never drive the daily model; they exist to confirm an in-house construction measures what its name claims."
          >
            {ref?.comparisons?.length ? (
              <div className="max-h-[280px] overflow-auto">
                <table className="w-full border-collapse">
                  <thead><tr><th className="th">Dataset</th><th className="th">Factor</th>
                    <th className="th">Overlap</th><th className="th">Correlation</th></tr></thead>
                  <tbody>
                    {ref.comparisons.map((c: any, i: number) => (
                      <tr key={i} className="border-t border-lineSoft">
                        <td className="cell max-w-[220px] truncate" title={c.dataset}>{c.dataset}</td>
                        <td className="cell">{c.factor}</td>
                        <td className="cell">{c.n_overlap?.toLocaleString()}</td>
                        <td className={`cell font-semibold ${
                          Math.abs(c.correlation) > 0.4 ? "text-pass" : "text-muted"}`}>
                          {num(c.correlation, 3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-8 text-center text-[12px] text-muted">
                No overlapping published series.
              </div>
            )}
          </Panel>
        </div>

        {window252 && <DiagnosticsPanel d={window252} all={diag?.windows ?? []} />}
      </div>
    </div>
  );
}

function DistributionPanel({
  hist, logY, setLogY,
}: {
  hist: any;
  logY: boolean;
  setLogY: (v: boolean) => void;
}) {
  if (!hist) return <Panel title="Return distribution"><div className="h-[300px]" /></Panel>;

  const outsideNote =
    hist.n_outside > 0
      ? ` ${hist.n_outside} of ${hist.n.toLocaleString()} observations lie outside the drawn range and are not shown.`
      : "";

  return (
    <Panel
      title="Return distribution"
      caption={
        <>
          {hist.n_bins} bins by the Freedman-Diaconis rule over the 0.1st to 99.9th
          percentile — an equal-width rule across the full range would spend almost
          every bin on empty tail. Student-t fit has {num(hist.t_df, 1)} degrees of
          freedom against Jarque-Bera p = {pval(hist.jarque_bera_p)}; the gap between
          the two curves in the tail is why a normal VaR under-counts breaches.
          {outsideNote}
        </>
      }
      actions={
        <label className="flex items-center gap-1.5 text-[11px] text-muted">
          <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} />
          log density
        </label>
      }
    >
      <Chart
        height={300}
        numericX
        data={[
          {
            x: hist.bin_centres, y: hist.density, type: "bar", name: "empirical",
            width: hist.bin_width,
            marker: { color: PALETTE[0], opacity: 0.35, line: { width: 0 } },
            hovertemplate: "%{x:.2%}<br>density %{y:.1f}<extra></extra>",
          },
          {
            x: hist.grid, y: hist.kde, type: "scatter", mode: "lines",
            name: "empirical (kernel)",
            line: { color: PALETTE[0], width: 2, shape: "spline" },
          },
          {
            x: hist.grid, y: hist.normal_pdf, type: "scatter", mode: "lines",
            name: "normal",
            line: { color: PALETTE[1], width: 1.6, shape: "spline" },
          },
          {
            x: hist.grid, y: hist.t_pdf, type: "scatter", mode: "lines",
            name: `student-t (df ${num(hist.t_df, 1)})`,
            line: { color: PALETTE[2], width: 1.6, dash: "dot", shape: "spline" },
          },
        ]}
        layout={{
          bargap: 0.02,
          xaxis: { title: "daily return", tickformat: ".1%", range: [hist.lo, hist.hi] },
          // Log density makes the tails legible: on a linear scale a peak near 50
          // flattens everything beyond two standard deviations into the axis, which
          // is precisely the region the normal-versus-t comparison is about.
          yaxis: logY
            ? { title: "density (log)", type: "log", exponentformat: "power" }
            : { title: "density", rangemode: "tozero" },
          hovermode: "x unified",
        }}
      />
    </Panel>
  );
}


function DiagnosticsPanel({ d, all }: { d: any; all: any[] }) {
  const [win, setWin] = useState<number>(d.window_days);
  const row = all.find((w) => w.window_days === win) ?? d;

  const adfRejects = row.adf_p !== null && row.adf_p < 0.05;
  const kpssRejects = row.kpss_p !== null && row.kpss_p < 0.05;
  const cell = adfRejects && !kpssRejects ? "stationary"
    : !adfRejects && kpssRejects ? "unit root"
    : adfRejects && kpssRejects ? "break / heteroskedasticity"
    : "inconclusive";

  return (
    <Panel
      title="Stationarity battery"
      caption="ADF and KPSS have opposite nulls; reading them jointly is the point. ARCH effects are recorded, never gated — a GARCH process is strictly stationary."
      actions={
        <select className="field w-auto" value={win}
                onChange={(e) => setWin(Number(e.target.value))}>
          {all.map((w) => (
            <option key={w.window_days} value={w.window_days}>
              {w.window_days === 0 ? "Full history" : `Trailing ${w.window_days}d`}
            </option>
          ))}
        </select>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-3 rounded border border-line bg-white px-3 py-2">
        <VerdictBadge verdict={row.verdict} />
        <span className="text-[12px]">
          <b className="font-semibold">{cell}</b>
          {row.verdict_reason ? ` — ${row.verdict_reason}` : ""}
        </span>
        {(row.flags ?? []).map((f: string) => (
          <span key={f} className="rounded bg-lineSoft px-1.5 py-0.5 font-mono text-2xs text-muted">
            {f}
          </span>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
        <Test name="ADF" stat={row.adf_stat} p={row.adf_p}
              hint="H0: unit root — reject is good" good={adfRejects} />
        <Test name="KPSS" stat={row.kpss_stat} p={row.kpss_p}
              hint="H0: stationary — reject is bad" good={!kpssRejects} />
        <Test name="Phillips-Perron" stat={row.pp_stat} p={row.pp_p}
              hint="HAC-robust ADF" good={row.pp_p !== null && row.pp_p < 0.05} />
        <Test name="Ljung-Box (10)" stat={row.lb10_stat} p={row.lb10_p}
              hint="autocorrelation" good={row.lb10_p !== null && row.lb10_p > 0.05} />
        <Test name="ARCH-LM" stat={row.arch_lm_stat} p={row.arch_lm_p}
              hint="clustering — expected" good={null} />
        <Test name="Jarque-Bera" stat={row.jb_stat} p={row.jb_p}
              hint="normality — usually rejected" good={null} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
        <Stat label="VR (2)" value={num(row.vr2)} hint="1 = random walk" />
        <Stat label="VR (5)" value={num(row.vr5)} hint=">1 stale, <1 bouncing" />
        <Stat label="VR (10)" value={num(row.vr10)} />
        <Stat label="AC(1)" value={num(row.ac1, 3)} />
        <Stat label="Zero returns" value={pct(row.zero_return_share, 1)} />
        <Stat label="Observations" value={row.n_obs?.toLocaleString()} />
      </div>

      {row.za_break_date && (
        <div className="mt-3 rounded border border-warn/30 bg-warn/5 px-3 py-2 text-[12px] text-warn">
          Zivot-Andrews locates a structural break at <b>{row.za_break_date}</b>
          {row.za_p !== null && <> (p = {pval(row.za_p)})</>}.
        </div>
      )}
    </Panel>
  );
}

function Test({ name, stat, p, hint, good }: {
  name: string; stat: number | null; p: number | null; hint: string; good: boolean | null;
}) {
  const tone = good === null ? "text-navy" : good ? "text-pass" : "text-fail";
  return (
    <div className="rounded border border-line bg-white px-2 py-1.5">
      <div className="text-2xs uppercase tracking-label text-muted">{name}</div>
      <div className={`font-mono text-sm font-semibold ${tone}`}>p = {pval(p)}</div>
      <div className="font-mono text-[10px] text-muted">stat {num(stat)}</div>
      <div className="text-[10px] leading-tight text-muted">{hint}</div>
    </div>
  );
}
