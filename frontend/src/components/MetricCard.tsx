"use client";

/**
 * One headline figure, large enough to read across a desk, with its definition
 * one hover away.
 *
 * The strips these replace were nine small mono numbers in a row: legible only to
 * someone who already knew what all nine were, and silent about the two things
 * that actually matter for most of them — how the number is computed, and where it
 * misleads. "Max VIF 88" is not information until you know it comes from
 * regressing one factor on the other thirty-nine, that the pipeline gates on it at
 * 100, and that on the raw panel it is a property of the factor set rather than a
 * fault.
 *
 * So the number gets the room it deserves and the explanation gets a popover. The
 * text lives in `lib/metrics.ts`, not here, so the condition number reads the same
 * on the Loadings Lab as it does on Covariance & PCA.
 */

import { ReactNode, useId, useState } from "react";
import { Metric } from "@/lib/metrics";
import { Tone } from "@/lib/verdict";
import { useCountUp } from "@/hooks/useCountUp";

/**
 * Which edge the popover hangs from.
 *
 * A 320px panel anchored left on the last card of a row runs off the screen. The
 * page passes the card's position rather than measuring: the grid wraps at known
 * breakpoints, so an index is enough and costs no layout pass.
 */
export type Align = "left" | "right";

export function MetricCard({
  label,
  value,
  tone = "neutral",
  metric,
  sub,
  align = "left",
  animate,
  format,
}: {
  label: string;
  value?: ReactNode;
  tone?: Tone;
  /** The definition shown on hover. Without one the card renders as a plain tile. */
  metric?: Metric;
  /** A second line under the number: a t-statistic, a p-value, a denominator. */
  sub?: ReactNode;
  align?: Align;
  /** Raw number to count up to on first render; `format` renders it. */
  animate?: number | null;
  format?: (v: number) => string;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const counted = useCountUp(animate ?? null);

  const colour =
    tone === "good" ? "text-pass"
      : tone === "warn" ? "text-warn"
      : tone === "bad" ? "text-fail"
      : "text-navy";

  const shown =
    animate !== undefined && format
      ? counted === null ? "—" : format(counted)
      : value;

  return (
    <div
      className="relative"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <div
        // Focusable so the definition is reachable without a mouse, which is the
        // only way a keyboard user gets at it at all.
        tabIndex={metric ? 0 : undefined}
        aria-describedby={metric && open ? id : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
        className={
          "h-full rounded border bg-white px-3 py-2.5 outline-none transition " +
          (metric
            ? "cursor-help border-line hover:border-navy2 hover:shadow-sm " +
              "focus-visible:border-navy2 focus-visible:shadow-sm"
            : "border-line")
        }
      >
        <div className="flex items-start justify-between gap-1">
          <span className="text-2xs uppercase tracking-label text-muted">
            {label}
          </span>
          {metric && (
            <span
              aria-hidden
              className="mt-[1px] shrink-0 font-mono text-[9px] leading-none text-navy3"
            >
              ?
            </span>
          )}
        </div>
        <div
          className={
            "hero-number mt-0.5 font-sans text-[22px] font-semibold leading-tight " +
            "tabular-nums " + colour
          }
        >
          {shown}
        </div>
        {sub && (
          <div className="mt-0.5 font-mono text-[10px] leading-tight text-muted">
            {sub}
          </div>
        )}
      </div>

      {metric && open && (
        <div
          id={id}
          role="tooltip"
          className={
            "absolute top-[calc(100%+6px)] z-40 w-[320px] rounded border " +
            "border-line bg-panel p-3 text-[11px] leading-snug shadow-lg " +
            (align === "right" ? "right-0" : "left-0")
          }
        >
          <div className="mb-1.5 text-[12px] font-semibold text-navy">
            {metric.title ?? label}
          </div>
          <Section body={metric.what} />
          <Section head="How it is computed" body={metric.how} />
          <Section head="How to read it" body={metric.read} />
        </div>
      )}
    </div>
  );
}

function Section({ head, body }: { head?: string; body: string }) {
  return (
    <div className="mt-1.5 first:mt-0">
      {head && (
        <div className="text-2xs uppercase tracking-label text-muted">{head}</div>
      )}
      <p className="text-muted">{body}</p>
    </div>
  );
}

/**
 * The strip these cards sit in.
 *
 * Five to a row rather than nine: a hero number needs the width, and two readable
 * rows beat one unreadable one. `overflow-visible` is load-bearing — the popover
 * is absolutely positioned inside a grid cell, and any clipping ancestor would cut
 * it off at the card edge.
 */
export function MetricStrip({ children }: { children: ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-2 overflow-visible sm:grid-cols-3 lg:grid-cols-5">
      {children}
    </div>
  );
}

/** Cards in the last two columns of a five-wide grid hang their popover right. */
export function alignFor(index: number, perRow = 5): Align {
  return index % perRow >= perRow - 2 ? "right" : "left";
}
