"use client";

/**
 * Factor Explorer: one factor at a time — return path, risk through time,
 * distribution, autocorrelation, and the full stationarity battery.
 */

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Chart, ChartSkeleton, PALETTE, Panel, Skeleton, Stat, VerdictBadge }
  from "@/components/Chart";
import { Sparkline } from "@/components/Sparkline";
import { api, Basis, FactorComparison, FactorMeta, num, pct, pval } from "@/lib/api";
import { blockColour, blockStyle } from "@/lib/blocks";
import { episodeLayout } from "@/lib/episodes";
import { StationarityBattery, useWindowPicker }
  from "@/components/StationarityBattery";
import { bySign } from "@/lib/verdict";
import { usePublishSnapshot } from "@/lib/chat-context";

// KaTeX, its stylesheet and its fonts are a few hundred kilobytes that matter
// only once someone opens a profile, so the dialog and everything it pulls in
// arrive on that click rather than with the page.
const FactorProfile = dynamic(
  () => import("@/components/FactorProfile").then((m) => m.FactorProfile),
  { ssr: false }
);

export default function FactorsPage() {
  const [factors, setFactors] = useState<FactorMeta[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [start, setStart] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const [series, setSeries] = useState<any>(null);
  const [inputs, setInputs] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [risk, setRisk] = useState<any>(null);
  const [hist, setHist] = useState<any>(null);
  const [qq, setQQ] = useState<any>(null);
  const [acf, setACF] = useState<any>(null);
  const [diag, setDiag] = useState<any>(null);
  const [ref, setRef] = useState<any>(null);
  const [logY, setLogY] = useState(false);
  const [sparks, setSparks] = useState<Record<string, number[]>>({});
  const [basis, setBasis] = useState<Basis>("excess");
  const [profileOf, setProfileOf] = useState<string | null>(null);
  const [comparison, setComparison] = useState<FactorComparison | null>(null);

  useEffect(() => {
    api.factors()
      .then((f) => {
        setFactors(f);
        if (f.length) setSelected(f[0].factor_id);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  // One request for all forty sparklines; the sidebar would otherwise fire forty.
  // Refetched with the basis so the mini-plots always show the same series as the
  // charts they link to.
  useEffect(() => {
    api.factorSparklines(basis).then((d) => setSparks(d.series)).catch(() => {});
  }, [basis]);

  useEffect(() => {
    if (!selected) return;
    // One query object for every panel, so no two can end up describing different
    // samples or a different series.
    const p = { basis, ...(start ? { start } : {}) };
    setError(null);
    setInputs(null);
    Promise.all([
      api.factorSeries(selected, p),
      api.factorStats(selected, p),
      api.factorRollingRisk(selected, "21,63,252", p),
      api.factorHistogram(selected, p),
      api.factorQQ(selected, p),
      api.factorACF(selected, p),
      api.factorDiagnostics(selected).catch(() => null),
      api.factorReference(selected).catch(() => null),
      // The stored inputs do not depend on the basis — they are what the factor
      // is built from, before any of it happens — but they do depend on `start`,
      // so they ride along here rather than in their own effect.
      api.factorInputs(selected, start ? { start } : undefined).catch(() => null),
    ])
      .then(([s, st, r, h, q, a, d, rf, inp]) => {
        setSeries(s); setStats(st); setRisk(r); setHist(h);
        setQQ(q); setACF(a); setDiag(d); setRef(rf); setInputs(inp);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [selected, start, basis]);

  // The raw-against-orthogonalised comparison carries both panels in one response,
  // so it does not belong in the basis-dependent fetch above — refetching it when
  // the toggle moves would reload the same bytes.
  useEffect(() => {
    if (!selected) return;
    setComparison(null);
    api.factorComparison(selected, start ? { start } : undefined)
      .then(setComparison)
      .catch(() => setComparison(null));
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
                      <span
                        role="button"
                        tabIndex={0}
                        aria-label={`What is ${f.factor_id}?`}
                        title="What is this, and where does it come from?"
                        onClick={(e) => { e.stopPropagation(); setProfileOf(f.factor_id); }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault(); e.stopPropagation();
                            setProfileOf(f.factor_id);
                          }
                        }}
                        className={`flex h-3.5 w-3.5 shrink-0 items-center justify-center
                                    rounded-full border text-[9px] font-semibold leading-none
                                    transition
                          ${active
                            ? "border-white/40 text-white/70 hover:border-white hover:text-white"
                            : "border-line text-muted hover:border-navy2 hover:text-navy"}`}
                      >
                        i
                      </span>
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
                  <div className="mt-1">
                    {basis === "excess" ? (
                      <>
                        Figures describe the <b>factor itself</b> — its log excess
                        return over cash, which is what it earned.
                      </>
                    ) : (
                      <>
                        Figures describe only what this factor adds <b>beyond{" "}
                        <span className="font-mono">
                          {meta.orthogonalize_against?.join(", ") || "nothing"}
                        </span></b>, once their overlap is removed. This is the series
                        the regression uses, and it is not what the factor earned.
                      </>
                    )}{" "}
                    <button
                      className="underline decoration-dotted underline-offset-2
                                 hover:text-navy"
                      onClick={() => setProfileOf(selected)}
                    >
                      What is this factor?
                    </button>
                  </div>
                  {meta.construction?.note && (
                    <div className="mt-1 italic">{meta.construction.note}</div>
                  )}
                </>
              }
              actions={
                meta.orthogonalize_against?.length > 0 ? (
                  <div className="flex overflow-hidden rounded border border-line">
                    {([
                      ["excess", "Factor",
                       "The series itself - what this factor earned."],
                      ["orth", "Incremental",
                       "What it adds beyond the factors above it, once their overlap "
                       + "is removed. This is the series the regression uses."],
                    ] as const).map(([v, label, tip]) => (
                      <button
                        key={v}
                        title={tip}
                        onClick={() => setBasis(v)}
                        className={`px-2.5 py-1 text-[11px] transition ${
                          basis === v
                            ? "bg-navy text-white"
                            : "bg-white text-muted hover:text-navy"
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                ) : undefined
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
                      tone={bySign(stats.mean_ann)}
                    />
                    <Stat
                      label="Sharpe" size="hero"
                      animate={stats.sharpe} format={(v) => num(v)}
                      tone={bySign(stats.sharpe)}
                    />
                  </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
                  <Stat label="Ann. return" value={pct(stats.mean_ann)}
                        tone={bySign(stats.mean_ann)} />
                  <Stat label="Ann. vol" value={pct(stats.vol_ann)} />
                  <Stat label="Sharpe" value={num(stats.sharpe)}
                        tone={bySign(stats.sharpe)} />
                  <Stat label="Skew" value={num(stats.skew)} />
                  <Stat label="Excess kurt." value={num(stats.excess_kurtosis)} />
                  <Stat label="Max drawdown" value={pct(stats.max_drawdown)} />
                  <Stat label="Hit rate" value={pct(stats.hit_rate)}
                        hint="share of up days" />
                  <Stat label="VaR 95% (1d)" value={pct(stats.var95_daily)} />
                  <Stat label="ES 95% (1d)" value={pct(stats.es95_daily)} />
                  <Stat label="Observations" value={stats.n_obs?.toLocaleString("en-US")} />
                  <Stat label="From" value={stats.first_date} />
                  <Stat label="To" value={stats.last_date} />
                </div>
                </>
              )}
            </Panel>
          )}
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <InputsPanel inputs={inputs} bands={bands} />

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
                        <td className="cell">{c.n_overlap?.toLocaleString("en-US")}</td>
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

        <ComparisonPanel data={comparison} accent={accent} episodes={bands} />

        {window252 && <FactorBattery latest={window252} all={diag?.windows ?? []} />}
      </div>

      {profileOf && (
        <FactorProfile factorId={profileOf} onClose={() => setProfileOf(null)} />
      )}
    </div>
  );
}

/**
 * The same factor on both panels, on the same days.
 *
 * The model can be estimated either way, so the question "what does the hierarchy
 * actually take out of this factor" is a real one rather than a footnote. Both
 * paths are drawn on one axis and the arithmetic is stated underneath: how much
 * variance came out, what the residual still correlates with, and what the average
 * loading was that produced it.
 *
 * Everything is measured on the intersection. The orthogonalised series starts 252
 * observations later — the burn-in before the first refit — and comparing a full
 * raw history with a shorter residual would book that difference in dates as an
 * effect of the orthogonalisation.
 */
function ComparisonPanel({
  data, accent, episodes,
}: {
  data: FactorComparison | null;
  accent: string;
  episodes: any;
}) {
  if (!data) {
    return (
      <Panel title="Raw against orthogonalised">
        <ChartSkeleton height={220} />
      </Panel>
    );
  }

  if (data.identical) {
    return (
      <Panel
        title="Raw against orthogonalised"
        caption="This factor sits at the top of its block hierarchy."
      >
        <p className="py-6 text-center text-[12px] text-muted">
          <span className="font-mono text-navy">{data.factor_id}</span> is
          residualised against nothing, so the two panels hold the same series,
          value for value. Estimating the model on raw factors rather than
          orthogonalised ones changes nothing about this one.
        </p>
      </Panel>
    );
  }

  const a = data.alignment;
  return (
    <Panel
      title="Raw against orthogonalised"
      caption={
        <>
          The model can be estimated on either panel, and this is the difference
          between them for this factor. <b>Raw</b> is what the factor earned;{" "}
          <b>orthogonalised</b> is what is left once{" "}
          <span className="font-mono">
            {data.targets.map((t) => t.factor_id).join(", ")}
          </span>{" "}
          have been regressed out; <b>removed</b> is the difference, which the model
          books against those factors instead. All three on the{" "}
          {a.n_obs.toLocaleString("en-US")} days where both series exist.
        </>
      }
    >
      <Chart
        height={220}
        episodes={episodes}
        data={[
          { x: data.dates, y: data.cumulative.raw, type: "scatter", mode: "lines",
            name: "raw", line: { color: accent, width: 1.4 },
            hovertemplate: "%{x|%Y-%m-%d}<br>raw %{y:.3f}<extra></extra>" },
          { x: data.dates, y: data.cumulative.orth, type: "scatter", mode: "lines",
            name: "orthogonalised", line: { color: "#2F3B52", width: 1.4 },
            hovertemplate: "%{x|%Y-%m-%d}<br>orth %{y:.3f}<extra></extra>" },
          { x: data.dates, y: data.cumulative.removed, type: "scatter", mode: "lines",
            name: "removed", line: { color: "#9AA0AE", width: 1, dash: "dot" },
            hovertemplate: "%{x|%Y-%m-%d}<br>removed %{y:.3f}<extra></extra>" },
        ]}
        layout={{
          yaxis: { title: "cumulative log return", zeroline: true },
          margin: { l: 60, r: 14, t: 8, b: 34 },
          legend: { orientation: "h", y: -0.22 },
        }}
      />

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="Variance removed"
          value={pct(a.variance_removed, 1)}
          tone={(a.variance_removed ?? 0) < 0 ? "warn" : "neutral"}
          hint={
            (a.variance_removed ?? 0) < 0
              ? "Negative: the residual is noisier than the factor it came from. "
                + "The fitted loading is applied a month after it was estimated, so "
                + "where the true loading is near zero and moves, subtracting it adds "
                + "variance instead of removing it."
              : "Share of the raw factor's variance the hierarchy takes out"
          }
        />
        <Stat
          label="Correlation"
          value={num(a.correlation, 3)}
          hint="Raw against orthogonalised, on the common days"
        />
        <Stat
          label="Removed vol"
          value={pct(a.tracking_vol_ann, 1)}
          hint="Annualised volatility of the part attributed to the blocks above"
        />
        <Stat
          label="Common sample"
          value={a.n_obs.toLocaleString("en-US")}
          hint={`${a.first_date} to ${a.last_date}`}
        />
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <div>
          <div className="label mb-1">Side by side</div>
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="th"> </th>
                <th className="th">Raw</th>
                <th className="th">Orthogonalised</th>
              </tr>
            </thead>
            <tbody>
              {([
                ["Return p.a.", (s: any) => pct(s.mean_ann, 1)],
                ["Volatility p.a.", (s: any) => pct(s.vol_ann, 1)],
                ["Sharpe", (s: any) => num(s.sharpe, 2)],
                ["Skew", (s: any) => num(s.skew, 2)],
                ["Excess kurtosis", (s: any) => num(s.excess_kurtosis, 1)],
                ["Max drawdown", (s: any) => pct(s.max_drawdown, 1)],
              ] as const).map(([label, fmt]) => (
                <tr key={label} className="border-t border-lineSoft">
                  <td className="cell font-sans text-muted">{label}</td>
                  <td className="cell">{fmt(data.stats.raw)}</td>
                  <td className="cell">{fmt(data.stats.orth)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <div className="label mb-1">What the residualisation achieved</div>
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="th">Against</th>
                <th className="th" title="Correlation of the raw factor with this target">
                  Before
                </th>
                <th className="th" title="Correlation of the orthogonalised factor with the same target">
                  After
                </th>
                <th className="th" title="Full-sample loading of the raw factor on its targets. The rolling fit refits every 21 days, so this is an average rather than a coefficient the model used.">
                  Avg β
                </th>
              </tr>
            </thead>
            <tbody>
              {data.targets.map((t) => {
                const beta = data.implied_betas.find(
                  (b) => b.factor_id === t.factor_id);
                return (
                  <tr key={t.factor_id} className="border-t border-lineSoft">
                    <td className="cell" title={t.name}>{t.factor_id}</td>
                    <td className="cell">{num(t.corr_raw, 3)}</td>
                    <td
                      className={`cell font-semibold ${
                        Math.abs(t.corr_orth ?? 0) < 0.15 ? "text-pass" : "text-warn"
                      }`}
                    >
                      {num(t.corr_orth, 3)}
                    </td>
                    <td className="cell text-muted">{num(beta?.average_beta, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-1.5 text-[11px] text-muted">
            A residual correlation near zero is the objective; a residue of a few
            hundredths is what a causal, rolling fit leaves when the true loading
            moves, and an <i>after</i> figure larger than the <i>before</i> one
            simply means there was nothing much to remove in the first place.
            Whatever remains is not lost — the factor covariance matrix estimates it
            explicitly.
          </p>
        </div>
      </div>
    </Panel>
  );
}

/**
 * The series the factor is built from, drawn as they are stored.
 *
 * This panel replaced the compounded and cumulative return paths. Both of those
 * were constructions, and a construction cannot be the evidence when the
 * construction is what is in doubt: one bad first observation moved the whole
 * compounded path down by 78% and left a perfectly plausible shape behind it.
 * A price is checkable against the outside world. A cumulated return is not.
 *
 * Scale is the one judgement here. Raw prices in a carry basket span two orders
 * of magnitude — USDJPY near 110, EURUSD near 1.1 — so the axis goes logarithmic
 * when the series are far enough apart that a linear one would draw the small
 * ones as flat lines on the floor. Judged on medians, so a yield that touches
 * zero does not by itself force the switch. Nothing is rescaled: a log axis is a
 * way of reading the numbers, not a change to them.
 */
function InputsPanel({ inputs, bands }: { inputs: any; bands: any }) {
  const drawn = (inputs?.series ?? []).filter((s: any) => s.values?.length);
  const absent = (inputs?.series ?? []).filter((s: any) => !s.values?.length);

  const median = (v: number[]) => {
    const s = [...v].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  };
  const medians = drawn.map((s: any) => Math.abs(median(s.values))).filter(Boolean);
  const spread = medians.length > 1
    ? Math.max(...medians) / Math.min(...medians) : 1;
  const allPositive = drawn.every((s: any) => s.values.every((v: number) => v > 0));
  const useLog = allPositive && spread > 20;

  const units = Array.from(new Set(drawn.map((s: any) => s.unit).filter(Boolean)));

  return (
    <Panel
      index={2}
      className="xl:col-span-2"
      title="Underlying data"
      caption={
        <>
          The stored series this factor is built from, as stored — an adjusted
          close for an instrument, a level for a macro series, a rate for a
          currency. Nothing here is derived, which is the point: a price can be
          checked against the outside world, and a cumulated return path cannot.
          {units.length > 0 && <> Units: {units.join(", ")}.</>}
          {useLog && <> Drawn on a logarithmic axis because the series are {Math.round(spread)}× apart; the values are untouched.</>}
        </>
      }
    >
      {!inputs && <ChartSkeleton height={300} />}
      {inputs && drawn.length === 0 && (
        <div className="flex h-[300px] items-center justify-center px-8 text-center
                        text-[11px] leading-relaxed text-muted">
          Nothing is stored under the identifiers this factor&rsquo;s construction
          names{absent.length > 0 && <> — {absent.map((s: any) => s.id).join(", ")}</>}.
          That is a gap in the data rather than a factor without inputs.
        </div>
      )}
      {inputs && drawn.length > 0 && (
        <>
          <Chart
            height={300}
            episodes={bands}
            data={drawn.map((s: any, i: number) => ({
              x: s.dates, y: s.values, type: "scatter", mode: "lines",
              name: s.label, line: { color: PALETTE[i % PALETTE.length], width: 1.4 },
              hovertemplate: `%{x|%Y-%m-%d}<br>%{y:,.4~f} ${s.unit ?? ""}<extra>${s.label}</extra>`,
            }))}
            layout={{
              yaxis: { title: useLog ? "level (log scale)" : "level",
                       type: useLog ? "log" : "linear" },
              showlegend: drawn.length > 1,
              legend: { orientation: "h", y: -0.18 },
            }}
          />
          {absent.length > 0 && (
            <p className="mt-2 text-[10.5px] leading-snug text-fail">
              Named in the construction but not stored:{" "}
              {absent.map((s: any) => s.id).join(", ")}. This factor is being built
              from less than it claims.
            </p>
          )}
        </>
      )}
    </Panel>
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
      ? ` ${hist.n_outside} of ${hist.n.toLocaleString("en-US")} observations lie outside the drawn range and are not shown.`
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


/**
 * The battery with this page's window picker attached.
 *
 * The diagnostics job stores a verdict per window, so a reader can ask whether a
 * series that passes on full history still passes on the trailing year. The Raw
 * Explorer computes one window live and has nothing to pick between, which is the
 * only difference between the two callers.
 */
function FactorBattery({ latest, all }: { latest: any; all: any[] }) {
  const { row, picker } = useWindowPicker(all, latest);
  return <StationarityBattery row={row} actions={picker} />;
}
