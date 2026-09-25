"use client";

/**
 * The stationarity battery, rendered once for both pages that run it.
 *
 * The Factor Explorer tests the orthogonalised factor and the Raw Explorer tests
 * the series before construction — different numbers, the same battery, and the
 * whole point of having both pages is that the same tests are applied to each. Two
 * layouts made that harder to see than it should be: one page showed six bordered
 * test cards with their verdicts spelled out, the other eleven undifferentiated
 * figures, and comparing them meant translating between two presentations first.
 *
 * Colour comes from `lib/verdict.ts`, so what counts as a good outcome is decided
 * in one place and cannot differ between the two callers either.
 */

import { ReactNode, useState } from "react";
import { Panel, Stat, VerdictBadge } from "@/components/Chart";
import { num, pct, pval } from "@/lib/api";
import { Assessment, assess } from "@/lib/verdict";

export interface DiagnosticsRow {
  window_days?: number;
  verdict?: string | null;
  verdict_reason?: string | null;
  flags?: string[] | null;
  adf_stat?: number | null; adf_p?: number | null;
  kpss_stat?: number | null; kpss_p?: number | null;
  pp_stat?: number | null; pp_p?: number | null;
  lb10_stat?: number | null; lb10_p?: number | null;
  arch_lm_stat?: number | null; arch_lm_p?: number | null;
  jb_stat?: number | null; jb_p?: number | null;
  vr2?: number | null; vr5?: number | null; vr10?: number | null;
  ac1?: number | null;
  zero_return_share?: number | null;
  n_obs?: number | null;
  za_break_date?: string | null; za_p?: number | null;
}

/**
 * The joint ADF x KPSS reading, which is the verdict the pipeline gates on.
 *
 * Opposite nulls, so the cross gives four answers rather than one, and only the
 * second blocks a series. A single ADF is useless on daily returns — it rejects
 * almost always.
 */
export function jointVerdict(row: DiagnosticsRow): string {
  const adfRejects = row.adf_p != null && row.adf_p < 0.05;
  const kpssRejects = row.kpss_p != null && row.kpss_p < 0.05;
  if (adfRejects && !kpssRejects) return "stationary";
  if (!adfRejects && kpssRejects) return "unit root";
  if (adfRejects && kpssRejects) return "break / heteroskedasticity";
  return "inconclusive";
}

export function StationarityBattery({
  row, caption, actions, index,
}: {
  row: DiagnosticsRow;
  /** Replaces the default explanation where a page needs to say something more. */
  caption?: ReactNode;
  actions?: ReactNode;
  index?: number;
}) {
  return (
    <Panel
      index={index}
      title="Stationarity battery"
      caption={caption ?? DEFAULT_CAPTION}
      actions={actions}
    >
      <div className="mb-3 flex flex-wrap items-center gap-3 rounded border
                      border-line bg-white px-3 py-2">
        <VerdictBadge verdict={row.verdict as any} />
        <span className="text-[12px]">
          <b className="font-semibold">{jointVerdict(row)}</b>
          {row.verdict_reason ? ` — ${row.verdict_reason}` : ""}
        </span>
        {(row.flags ?? []).map((f) => (
          <span key={f}
                className="rounded bg-lineSoft px-1.5 py-0.5 font-mono text-2xs text-muted">
            {f}
          </span>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
        <Test name="ADF" stat={row.adf_stat} p={row.adf_p}
              what="H0: unit root" a={assess.adf(row.adf_p)} />
        <Test name="KPSS" stat={row.kpss_stat} p={row.kpss_p}
              what="H0: stationary" a={assess.kpss(row.kpss_p)} />
        <Test name="Phillips-Perron" stat={row.pp_stat} p={row.pp_p}
              what="HAC-robust ADF" a={assess.pp(row.pp_p)} />
        <Test name="Ljung-Box (10)" stat={row.lb10_stat} p={row.lb10_p}
              what="H0: no autocorrelation" a={assess.ljungBox(row.lb10_p)} />
        <Test name="ARCH-LM" stat={row.arch_lm_stat} p={row.arch_lm_p}
              what="recorded, never gated" a={assess.archLm(row.arch_lm_p)} />
        <Test name="Jarque-Bera" stat={row.jb_stat} p={row.jb_p}
              what="informational" a={assess.jarqueBera(row.jb_p)} />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
        {([
          ["VR (2)", num(row.vr2), assess.varianceRatio(row.vr2), "1 = random walk"],
          ["VR (5)", num(row.vr5), assess.varianceRatio(row.vr5), "1 = random walk"],
          ["VR (10)", num(row.vr10), assess.varianceRatio(row.vr10), "1 = random walk"],
          ["AC(1)", num(row.ac1, 3), assess.autocorrelation(row.ac1),
           "first-order autocorrelation"],
          ["Zero returns", pct(row.zero_return_share, 1),
           assess.zeroReturns(row.zero_return_share), "illiquidity check"],
        ] as const).map(([label, value, a, what]) => (
          <Stat key={label} label={label} value={value} tone={a.tone}
                hint={`${what} — ${a.reason}`} />
        ))}
        <Stat label="Observations" value={row.n_obs?.toLocaleString("en-US")} />
      </div>

      {row.za_break_date && (
        <div className="mt-3 rounded border border-warn/30 bg-warn/5 px-3 py-2
                        text-[12px] text-warn">
          Zivot-Andrews locates a structural break at <b>{row.za_break_date}</b>
          {row.za_p != null && <> (p = {pval(row.za_p)})</>}.
        </div>
      )}
    </Panel>
  );
}

const DEFAULT_CAPTION =
  "ADF and KPSS have opposite nulls; reading them jointly is the point. ARCH " +
  "effects are recorded, never gated — a GARCH process is strictly stationary.";

/**
 * A window picker for the pages that store several windows per series.
 *
 * Returned as a hook rather than built in, because the Raw Explorer computes its
 * battery live on one window and has nothing to pick between.
 */
export function useWindowPicker(all: any[], fallback: any) {
  const [win, setWin] = useState<number>(fallback?.window_days ?? 0);
  const row = all.find((w) => w.window_days === win) ?? fallback;
  const picker = all.length > 1 ? (
    <select className="field w-auto" value={win}
            onChange={(e) => setWin(Number(e.target.value))}>
      {all.map((w) => (
        <option key={w.window_days} value={w.window_days}>
          {w.window_days === 0 ? "Full history" : `Trailing ${w.window_days}d`}
        </option>
      ))}
    </select>
  ) : undefined;
  return { row, picker };
}

/**
 * One test. The colour is the assessment, not the p-value: what counts as a good
 * outcome differs test by test — ADF wants a rejection, KPSS wants the opposite,
 * ARCH-LM wants nothing at all — so it comes from lib/verdict.ts.
 */
function Test({ name, stat, p, what, a }: {
  name: string;
  stat?: number | null;
  p?: number | null;
  what: string;
  a: Assessment;
}) {
  const tone = a.tone === "good" ? "text-pass"
    : a.tone === "warn" ? "text-warn"
    : a.tone === "bad" ? "text-fail"
    : "text-navy";
  return (
    <div className="rounded border border-line bg-white px-2 py-1.5" title={a.reason}>
      <div className="text-2xs uppercase tracking-label text-muted">{name}</div>
      <div className={`font-mono text-sm font-semibold ${tone}`}>p = {pval(p)}</div>
      <div className="font-mono text-[10px] text-muted">stat {num(stat)}</div>
      <div className="text-[10px] leading-tight text-muted">{what}</div>
      <div className={`text-[10px] leading-tight ${
        a.tone === "neutral" ? "text-muted" : tone}`}>{a.reason}</div>
    </div>
  );
}
