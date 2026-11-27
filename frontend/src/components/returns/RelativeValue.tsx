"use client";

/**
 * The pieces of the Relative Value & Timing page: the scorecard, the class
 * table, the credit and macro panels, and the DeepSeek summary.
 *
 * Colour on a z-score is direction, not verdict. A negative value z says the
 * long leg is dear, which is not a failure of anything, so the scorecard does
 * not borrow verdict.ts's green and red (those claim "the model wants this").
 * Positive is the navy of the charts, negative their copper, and the tint
 * deepens with |z| up to 2.
 */

import { Fragment, useState } from "react";
import { InfoTip } from "@/components/InfoTip";
import { Panel } from "@/components/Chart";
import { Sparkline } from "@/components/Sparkline";
import {
  RelativeValueRun, RvClassRow, RvCreditLeg, RvSignal, RvSignalKey, RvSummary, RvView, pct,
} from "@/lib/api";
import { RV_METRICS as M } from "@/lib/relativeValueMetrics";
import { signedPct } from "@/components/returns/shared";

const NAVY = "47,77,115";
const COPPER = "194,112,61";

/** Background tint for a signed score: navy for the long leg, copper against it. */
export function zTint(z: number | null | undefined): React.CSSProperties {
  if (z === null || z === undefined || !isFinite(z)) return {};
  const a = Math.min(Math.abs(z) / 2, 1) * 0.3;
  return { backgroundColor: `rgba(${z > 0 ? NAVY : COPPER},${a.toFixed(3)})` };
}

/** A share of history as an ordinal percentile ("83rd"), floored at the 1st. */
export function ordinal(p: number | null | undefined): string {
  if (p === null || p === undefined || !isFinite(p)) return "—";
  const n = Math.max(1, Math.min(100, Math.round(p * 100)));
  const suffix = n % 100 >= 10 && n % 100 <= 20 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${suffix}`;
}

/** "+1.25" or "−0.62"; a dash when there is no value. */
export function fz(z: number | null | undefined, digits = 2): string {
  if (z === null || z === undefined || !isFinite(z)) return "—";
  const s = Math.abs(z).toFixed(digits);
  return z > 0 ? `+${s}` : z < 0 ? `−${s}` : s;
}

const SIGNALS: readonly (readonly [RvSignalKey, string])[] = [
  ["value", "Value"], ["carry", "Carry"], ["trend", "Trend"], ["macro", "Macro"],
];

function stanceClass(s: string): string {
  if (s === "overweight") return "font-semibold text-navy";
  if (s === "underweight") return "font-semibold text-[#A14E3A]";
  if (s.startsWith("lean over")) return "text-navy";
  if (s.startsWith("lean under")) return "text-[#A14E3A]";
  return "text-muted";
}

function legs(v: RvView): string {
  const long = v.hedge ? `${v.long} (hedged)` : v.long;
  return v.short ? `${long} / ${v.short}` : `${long} / cash`;
}

// ---------------------------------------------------------------------------
// scorecard
// ---------------------------------------------------------------------------

export function Scorecard({ run, reads }: { run: RelativeValueRun; reads: Record<string, string> }) {
  const [open, setOpen] = useState<string | null>(null);
  const cols = 15;
  return (
    <Panel
      title="Scorecard: four signals per view, the score, and what it implies"
      caption={<>Every signal is a z-score against the pair&apos;s own history, clipped at ±3 and
        positive when it favours the long leg; navy favours the long leg, copper the short. The
        score averages the signals a view has. Tilts are on the long leg, scaled into a{" "}
        {pct(run.sizing.te_budget, 1)} tracking-error budget. Click a row for the inputs.</>}
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="th">View</th>
              <th className="th">Legs</th>
              {SIGNALS.map(([k, label]) => (
                <th key={k} className="th text-right">{label}<InfoTip metric={M[k]} /></th>
              ))}
              <th className="th text-right">Score<InfoTip metric={M.score} /></th>
              <th className="th text-right">Δ 1M<InfoTip metric={M.change} /></th>
              <th className="th">Band</th>
              <th className="th">Stance</th>
              <th className="th text-right">Tilt<InfoTip metric={M.tilt} /></th>
              <th className="th text-right">TE</th>
              <th className="th text-right">10y excess<InfoTip metric={M.anchor} /></th>
              <th className="th text-right">/ risk</th>
              <th className="th">36 months</th>
            </tr>
          </thead>
          <tbody>
            {run.views.map((v) => {
              const last = v.history.length ? v.history[v.history.length - 1].score : 0;
              return (
                <Fragment key={v.id}>
                  <tr onClick={() => setOpen(open === v.id ? null : v.id)}
                      className={"cursor-pointer border-t border-lineSoft hover:bg-white "
                        + (open === v.id ? "bg-white" : "")}>
                    <td className="cell whitespace-nowrap font-sans text-[11.5px]">
                      <span className="mr-1 inline-block w-2 text-muted">{open === v.id ? "▾" : "▸"}</span>
                      {v.label}</td>
                    <td className="cell whitespace-nowrap text-muted">{legs(v)}</td>
                    {SIGNALS.map(([k]) => {
                      const z = v.signals[k]?.z ?? null;
                      return <td key={k} className="cell text-right" style={zTint(z)}>{fz(z)}</td>;
                    })}
                    <td className="cell text-right font-semibold" style={zTint(v.score)}>{fz(v.score)}</td>
                    <td className="cell text-right text-muted">{fz(v.change)}</td>
                    <td className={"cell font-sans text-[11px] " + (v.act ? "text-navy" : "text-muted")}>
                      {v.score === null ? "—" : v.act ? "act" : "hold"}</td>
                    <td className={"cell whitespace-nowrap font-sans text-[11px] " + stanceClass(v.stance)}>
                      {v.stance}</td>
                    <td className="cell text-right">{signedPct(v.tilt, 1)}</td>
                    <td className="cell text-right text-muted">{pct(v.te, 2)}</td>
                    <td className="cell text-right">{signedPct(v.anchor?.pair_excess, 1)}</td>
                    <td className="cell text-right" style={zTint(v.anchor?.per_risk)}>
                      {fz(v.anchor?.per_risk)}</td>
                    <td className="cell">
                      <Sparkline values={v.history.map((h) => h.score)} width={64}
                                 colour={last >= 0 ? "#2F4D73" : "#C2703D"} />
                    </td>
                  </tr>
                  {reads[v.id] && (
                    <tr className="border-t border-dashed border-lineSoft">
                      <td colSpan={cols} className="px-2 pb-1.5 pt-0.5 pl-[22px] text-[10.5px] leading-snug text-muted">
                        <span className="mr-1.5 rounded-sm bg-lineSoft px-1 text-[9px] font-semibold uppercase tracking-wide text-navy3">
                          DeepSeek</span>{reads[v.id]}</td>
                    </tr>
                  )}
                  {open === v.id && (
                    <tr className="bg-white">
                      <td colSpan={cols} className="px-3 pb-3 pt-1"><ViewDetail v={v} /></td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function rawText(s: RvSignal): string {
  return s.unit === "pct" ? pct(s.raw, 2) : fz(s.raw, 3);
}

function ViewDetail({ v }: { v: RvView }) {
  const d = v.detail ?? {};
  return (
    <div className="space-y-2 text-[11px] leading-snug">
      <div className="text-muted">{v.note}. Pair: {legs(v)}
        {v.short && <>, short leg at {v.weight_short.toFixed(2)}× for equal volatility</>}
        ; pair volatility {pct(v.pair_vol, 1)}, history from {v.pair_from ?? "—"}.
        Weights used: {Object.entries(v.weights_used).map(([k, w]) => `${k} ${(w * 100).toFixed(0)}%`)
          .join(", ") || "none"}.</div>
      <div className="grid gap-3 md:grid-cols-4">
        {SIGNALS.map(([k, label]) => {
          const s = v.signals[k];
          return (
            <div key={k} className="rounded border border-lineSoft bg-panel px-2.5 py-2">
              <div className="label mb-0.5">{label} <span className="font-mono normal-case tracking-normal">
                {fz(s?.z ?? null)}</span></div>
              {!s ? <div className="text-muted">No {k} signal for this view.</div> : (
                <>
                  {s.label && <div className="text-navy">{s.label}</div>}
                  {s.raw !== undefined && s.raw !== null && (
                    <div className="font-mono">now {rawText(s)} · mean {s.unit === "pct" ? pct(s.mean, 2)
                      : fz(s.mean, 3)} · {ordinal(s.pct)} percentile</div>
                  )}
                  {s.n ? <div className="text-muted">{s.n} months from {s.from}, to {s.as_of}</div> : null}
                  {k === "trend" && (
                    <div className="font-mono">12-1M {signedPct(s.ret, 1)} · vol {pct(s.vol, 1)}
                      {s.above_sma !== null && s.above_sma !== undefined &&
                        <> · {s.above_sma ? "above" : "below"} 10M avg</>}</div>
                  )}
                  {k === "macro" && (
                    <div className="font-mono">growth sign {fz(s.s_growth ?? 0, 0)}, inflation sign{" "}
                      {fz(s.s_inflation ?? 0, 0)}</div>
                  )}
                  {s.how && <div className="mt-0.5 text-muted">{s.how}</div>}
                </>
              )}
            </div>
          );
        })}
      </div>
      {v.id === "us_eq_vs_tips" && d.cape !== undefined && (
        <div className="font-mono text-muted">Shiller CAPE {Number(d.cape).toFixed(1)} ({d.cape_as_of}) ·
          TIPS real {pct(d.tips_real as number, 2)} · excess CAPE yield on TIPS{" "}
          {pct(d.ecy_tips as number, 2)} · on Shiller&apos;s real rate {pct(d.ecy_shiller as number, 2)}</div>
      )}
      {v.id === "tips_vs_nominal" && d.breakeven !== undefined && (
        <div className="font-mono text-muted">Breakeven {pct(d.breakeven as number, 2)} ({d.breakeven_as_of}) ·
          CPI, 3 months annualised {pct(d.cpi_3m as number, 1)} ({d.cpi_as_of}) · TIPS real{" "}
          {pct(d.tips_real as number, 2)}. TIPS accrue CPI, which runs above the Fed&apos;s PCE target.</div>
      )}
      {v.id === "duration" && d.term_premium !== undefined && (
        <div className="font-mono text-muted">ACM term premium {pct(d.term_premium as number, 2)} ·
          10y {pct(d.dgs10 as number, 2)} · fed funds {pct(d.cash as number, 2)}</div>
      )}
      {v.anchor && (
        <div className="font-mono text-muted">Ten-year anchor: long {signedPct(v.anchor.long_excess, 2)}
          {v.anchor.short_excess !== null && <>, short {signedPct(v.anchor.short_excess, 2)}</>} over cash
          → pair {signedPct(v.anchor.pair_excess, 2)}, {fz(v.anchor.per_risk)} per unit of risk.</div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// classes
// ---------------------------------------------------------------------------

export function ClassesTable({ rows }: { rows: RvClassRow[] }) {
  const max = Math.max(0.01, ...rows.map((r) => Math.abs(r.per_risk ?? 0)));
  return (
    <Panel title="Ten-year attractiveness beside each class's own trend"
           caption="Ranked by expected excess return over cash per unit of volatility, from the latest
             Ten-Year Forecasts run; TIPS, high yield and investment grade from their yields. The trend
             columns are each fund's own excess return.">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="th">#</th>
              <th className="th">Class</th>
              <th className="th">Fund</th>
              <th className="th text-right">10y exp.</th>
              <th className="th text-right">Excess</th>
              <th className="th text-right">Vol</th>
              <th className="th text-right">/ risk<InfoTip metric={M.per_risk} /></th>
              <th className="th w-28" />
              <th className="th text-right">Trend<InfoTip metric={M.trend} /></th>
              <th className="th text-right">12-1M</th>
              <th className="th">10M avg</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const w = r.per_risk === null ? 0 : (Math.abs(r.per_risk) / max) * 50;
              return (
                <tr key={r.fund} className="border-t border-lineSoft" title={r.source}>
                  <td className="cell text-muted">{r.rank}</td>
                  <td className="cell whitespace-nowrap font-sans text-[11.5px]">{r.label}</td>
                  <td className="cell text-muted">{r.fund}</td>
                  <td className="cell text-right">{pct(r.expected, 1)}</td>
                  <td className="cell text-right">{signedPct(r.excess, 1)}</td>
                  <td className="cell text-right text-muted">{pct(r.vol, 1)}</td>
                  <td className="cell text-right font-semibold">{fz(r.per_risk)}</td>
                  <td className="cell">
                    <div className="relative h-2.5 w-28">
                      <div className="absolute left-1/2 top-0 h-full w-px bg-line" />
                      {r.per_risk !== null && (
                        <div className="absolute top-0 h-full rounded-sm"
                             style={{ width: `${w}%`, left: r.per_risk >= 0 ? "50%" : `${50 - w}%`,
                                      background: r.per_risk >= 0 ? "#2F4D73" : "#C2703D" }} />
                      )}
                    </div>
                  </td>
                  <td className="cell text-right" style={zTint(r.trend_z)}>{fz(r.trend_z)}</td>
                  <td className="cell text-right">{signedPct(r.ret_12_1, 1)}</td>
                  <td className="cell font-sans text-[11px] text-muted">
                    {r.above_sma === null ? "—" : r.above_sma ? "above" : "below"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// credit and macro
// ---------------------------------------------------------------------------

export function CreditPanel({ credit }: { credit: NonNullable<RelativeValueRun["credit"]> }) {
  const bp = (x: number | null | undefined) =>
    x === null || x === undefined || !isFinite(x) ? "—" : `${Math.round(x * 1e4)}`;
  const row = (name: string, l: RvCreditLeg) => (
    <tr className="border-t border-lineSoft">
      <td className="cell whitespace-nowrap font-sans text-[11.5px]">{name}</td>
      <td className="cell text-right">{bp(l.spread)}</td>
      <td className="cell text-right">{pct(l.pd, 1)}</td>
      <td className="cell text-right">{bp(l.el)}</td>
      <td className="cell text-right font-semibold">{bp(l.excess)}</td>
      <td className="cell text-right">{pct(l.pd_star, 1)}</td>
      <td className="cell text-right">{l.cover === null ? "—" : `${l.cover.toFixed(1)}×`}</td>
      <td className="cell text-right">{bp(l.cushion)}</td>
    </tr>
  );
  return (
    <Panel title="Credit after expected losses"
           caption={<>ICE BofA OAS on {credit.as_of}; LGD {pct(credit.hy.lgd, 0)}, spread duration
             HY {credit.hy.spread_duration}, IG {credit.ig.spread_duration}. HY OAS{" "}
             {signedPct(credit.hy_oas_change_3m, 0)} in three months; CCC {Math.round((credit.ccc_oas ?? 0) * 1e4)} bp,
             BBB {Math.round((credit.bbb_oas ?? 0) * 1e4)} bp. The OAS history starts on {credit.oas_from}.</>}
           actions={<InfoTip metric={M.credit} />}>
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="th" />
            <th className="th text-right">OAS bp</th>
            <th className="th text-right">PD</th>
            <th className="th text-right">EL bp</th>
            <th className="th text-right">Excess bp</th>
            <th className="th text-right">PD*</th>
            <th className="th text-right">Cover</th>
            <th className="th text-right">Cushion bp</th>
          </tr>
        </thead>
        <tbody>
          {row("High yield", credit.hy)}
          {row("High yield, lower PD", credit.hy_low_pd)}
          {row("Investment grade", credit.ig)}
        </tbody>
      </table>
    </Panel>
  );
}

const QUADRANTS: readonly (readonly [string, string, string])[] = [
  ["stagflation", "Stagflation", "cash, TIPS, gold"],
  ["reflation", "Reflation", "commodities, value"],
  ["recession", "Recession", "government bonds"],
  ["goldilocks", "Goldilocks", "equities, credit"],
];

export function MacroPanel({ macro }: { macro: RelativeValueRun["macro"] }) {
  return (
    <Panel title="Macro regime" actions={<InfoTip metric={M.regime} />}
           caption={<>Growth {fz(macro.growth.z)}: {macro.growth.how}. Inflation {fz(macro.inflation.z)}:{" "}
             {macro.inflation.how}.{macro.vix && <> VIX {macro.vix.level.toFixed(1)}, z {fz(macro.vix.z)},{" "}
               {ordinal(macro.vix.pct)} percentile since 2000.</>}</>}>
      <div className="grid grid-cols-2 gap-1 text-[11px]">
        {QUADRANTS.map(([k, name, what]) => (
          <div key={k} className={"rounded border px-2 py-1.5 " + (macro.regime === k
            ? "border-navy bg-navy text-white" : "border-lineSoft bg-white text-muted")}>
            <div className="font-medium">{name}</div>
            <div className={macro.regime === k ? "text-white/80" : ""}>{what}</div>
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-muted">
        <span>↑ inflation, top row</span><span>growth →, right column</span>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// the summary
// ---------------------------------------------------------------------------

export function SummaryPanel({
  summary, configured, model, busy, onSummarise, error, unverified, runId,
}: {
  summary: RvSummary | null;
  configured: boolean;
  model: string;
  busy: { elapsed: number } | null;
  onSummarise: (force: boolean) => void;
  error: string | null;
  unverified: string[];
  runId: string | undefined;
}) {
  const action = !configured ? null : (
    <button className="btn-ghost" disabled={!!busy || !runId} onClick={() => onSummarise(!!summary)}>
      {busy ? `Writing… ${Math.round(busy.elapsed)}s` : summary ? "Regenerate" : "Summarise with DeepSeek"}
    </button>
  );
  const c = summary?.content;
  return (
    <Panel title="Summary" actions={action}
           caption={c ? <>DeepSeek {summary!.model}, {summary!.created_at.slice(0, 16).replace("T", " ")},{" "}
             {summary!.attempts === 1 ? "first attempt" : `${summary!.attempts} attempts`}. It reads the
             scorecard; every number in it was checked against this run&apos;s data, and the stances
             are the pipeline&apos;s.</>
             : configured ? <>DeepSeek ({model}) reads this run&apos;s scorecard. Every number it writes
               is checked against the data; a summary that quotes a figure the run does not contain is
               refused.</>
             : <>No DeepSeek key is configured on the server (DEEPSEEK_API_KEY); the scores below do not
               depend on it.</>}>
      {error && (
        <div className="mb-2 rounded border border-fail/30 bg-fail/5 p-2 text-[11.5px] text-fail">
          {error}{unverified.length > 0 && <> Numbers not in the data: {unverified.join(", ")}.</>}</div>
      )}
      {c ? (
        <div className="space-y-2">
          <p className="text-[13px] font-medium leading-snug text-navy">{c.headline}</p>
          <div className="grid gap-3 text-[11.5px] leading-snug md:grid-cols-2">
            <div><div className="label mb-0.5">Timing</div>{c.timing}</div>
            <div><div className="label mb-0.5">Ten-year anchor</div>{c.classes}</div>
            {c.risks.length > 0 && (
              <div><div className="label mb-0.5">What would change it</div>
                <ul className="list-disc space-y-0.5 pl-4">{c.risks.map((r, i) => <li key={i}>{r}</li>)}</ul></div>
            )}
            {c.caveats.length > 0 && (
              <div><div className="label mb-0.5">Read with care</div>
                <ul className="list-disc space-y-0.5 pl-4 text-muted">
                  {c.caveats.map((r, i) => <li key={i}>{r}</li>)}</ul></div>
            )}
          </div>
        </div>
      ) : !error && configured && (
        <p className="text-[11.5px] text-muted">No summary of this run yet.</p>
      )}
    </Panel>
  );
}
