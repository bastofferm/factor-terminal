"use client";

/**
 * Pieces every Return Forecasts view uses: class labels, vector colours, the
 * source list behind a number and the licence notes the external data carries.
 */

import { DataLicence, ReturnForecast, ReturnSource } from "@/lib/api";
import { PALETTE } from "@/components/Chart";

/** Class key -> label, in the forecast's own order. */
export function classLabels(r: ReturnForecast): Record<string, string> {
  return Object.fromEntries(r.classes.map((c) => [c.key, c.label]));
}

/** The risky classes in the order the vectors and the covariance use. */
export function riskyKeys(r: ReturnForecast): string[] {
  return r.blocks.map((b) => b.key);
}

const VECTOR_COLOUR: Record<string, string> = {
  building_blocks: PALETTE[0],
  history: "#9AA0AB",
  market: PALETTE[1],
  equal: PALETTE[2],
  min_variance: PALETTE[3],
  blend: "#2A2F3A",
};

/** A fixed colour per vector, so a vector reads the same in every chart. */
export function vectorColour(key: string, i = 0): string {
  return VECTOR_COLOUR[key] ?? PALETTE[(4 + i) % PALETTE.length];
}

/** Vector keys in the order the run produced them (JSONB does not keep it). */
export function vectorOrder(r: ReturnForecast): string[] {
  return r.vector_order?.length ? r.vector_order : Object.keys(r.vectors);
}

/** Signed percentage with one decimal, e.g. "+1.2%" or "−0.6%". */
export function signedPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const s = (v * 100).toFixed(digits);
  return v > 0 ? `+${s}%` : v < 0 ? `−${s.replace("-", "")}%` : `${s}%`;
}

export function SourceList({ sources }: { sources: ReturnSource[] }) {
  return (
    <ul className="space-y-0.5 text-[11px] leading-snug">
      {sources.map((s, i) => (
        <li key={i}>
          <span className="label mr-1.5 inline-block w-20 normal-case tracking-normal">
            {s.block.replace("_", " ")}</span>
          <span className="text-navy">{s.source}</span>
          {s.as_of && <span className="text-muted"> · {s.as_of}</span>}
          {s.note && <span className="text-muted"> · {s.note}</span>}
        </li>
      ))}
    </ul>
  );
}

/**
 * The terms of the external data a view shows. Barclays' CAPE and the JST
 * panel are licensed for internal and non-commercial use; the page says so
 * beside the numbers rather than in a document nobody opens.
 */
export function LicenceNotes({ licences, only }: { licences: DataLicence[]; only?: string[] }) {
  const shown = licences.filter((l) => !only || only.includes(l.source));
  if (!shown.length) return null;
  return (
    <div className="rounded border border-lineSoft bg-white px-3 py-2 text-[10.5px] leading-snug text-muted">
      <div className="label mb-1">Data sources and terms</div>
      <ul className="space-y-1">
        {shown.map((l) => (
          <li key={l.source}>
            <span className="font-medium text-navy">{l.name}.</span> {l.attribution}.{" "}
            {l.licence}
            {l.restriction && <> <span className="text-warn">{l.restriction}</span></>}
            {l.accessed_on && <> Accessed {l.accessed_on}.</>}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A small two-way or n-way toggle, styled like the page tab bar but quieter. */
export function Toggle<T extends string>({
  value, options, onChange, label,
}: {
  value: T;
  options: readonly (readonly [T, string])[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label}
         className="inline-flex rounded border border-line bg-white p-0.5">
      {options.map(([v, text]) => (
        <button key={v} role="radio" aria-checked={value === v} onClick={() => onChange(v)}
                className={"rounded px-2 py-0.5 text-[11px] transition " + (value === v
                  ? "bg-navy2 text-white" : "text-navy hover:bg-lineSoft/60")}>
          {text}
        </button>
      ))}
    </div>
  );
}
