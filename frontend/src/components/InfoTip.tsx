"use client";

/**
 * A "?" beside a label that opens the label's definition.
 *
 * MetricCard carries its definition on the card; a table header or a line in a
 * list has no card to hang it on, so this is the same content behind a small
 * mark. The popover is rendered into document.body at the mark's screen position:
 * the leaderboard scrolls inside a fixed-height box, and an absolutely positioned
 * popover inside it would be cut off at the box's edge.
 */

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FormulaSection, Section } from "@/components/MetricCard";
import { Metric } from "@/lib/metrics";

const WIDTH = 440;

export function InfoTip({ metric, label }: { metric?: Metric; label?: string }) {
  const [pos, setPos] = useState<{ x: number; y: number; above: boolean } | null>(null);
  const [pinned, setPinned] = useState(false);
  const ref = useRef<HTMLButtonElement>(null);
  const id = useId();

  const show = () => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const x = Math.max(8, Math.min(r.left - 12, window.innerWidth - WIDTH - 8));
    const above = r.bottom > window.innerHeight * 0.6;
    setPos({ x, y: above ? r.top - 6 : r.bottom + 6, above });
  };
  const hide = () => { if (!pinned) setPos(null); };

  useEffect(() => {
    if (!pinned) return;
    const close = () => { setPinned(false); setPos(null); };
    window.addEventListener("scroll", close, true);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("keydown", close);
    };
  }, [pinned]);

  if (!metric) return null;
  return (
    <>
      <button
        ref={ref}
        type="button"
        aria-label={`What is ${metric.title ?? label ?? "this"}?`}
        aria-describedby={pos ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={() => { setPinned(false); setPos(null); }}
        onClick={(e) => { e.stopPropagation(); setPinned(!pinned); show(); }}
        className="ml-1 inline-flex h-[13px] w-[13px] shrink-0 cursor-help items-center
                   justify-center rounded-full border border-navy3/60 align-middle
                   font-mono text-[8.5px] leading-none text-navy3 hover:border-navy2
                   hover:text-navy2"
      >
        ?
      </button>
      {pos && typeof document !== "undefined" && createPortal(
        <div
          id={id}
          role="tooltip"
          style={{ left: pos.x, top: pos.above ? undefined : pos.y,
                   bottom: pos.above ? window.innerHeight - pos.y : undefined,
                   width: WIDTH }}
          className="fixed z-[60] max-h-[70vh] overflow-y-auto rounded border border-line
                     bg-panel p-3 text-left text-[11px] font-normal normal-case leading-snug
                     tracking-normal shadow-lg"
          onMouseEnter={() => setPinned(true)}
          onMouseLeave={() => { setPinned(false); setPos(null); }}
        >
          <div className="mb-1.5 text-[12px] font-semibold text-navy">
            {metric.title ?? label}
          </div>
          <Section body={metric.what} />
          <Section head="How it is computed" body={metric.how} />
          <FormulaSection latex={metric.formula} />
          {metric.h0 && <Section head="Null hypothesis" body={metric.h0} />}
          <Section head="How to read it" body={metric.read} />
          {metric.ref && <Section head="Source" body={metric.ref} />}
        </div>,
        document.body,
      )}
    </>
  );
}
