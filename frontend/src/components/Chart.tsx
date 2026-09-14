"use client";

/**
 * Plotly wrapper with a house layout.
 *
 * Loaded client-side only: plotly.js touches `document` at import time and would
 * break server rendering. The cartesian bundle is used rather than the full one —
 * it carries scatter, bar, histogram, heatmap and contour, which is everything
 * here, at roughly a third of the weight.
 */

import dynamic from "next/dynamic";
import type { ReactNode } from "react";
import { useCountUp } from "@/hooks/useCountUp";

// react-plotly.js is untyped, and next/dynamic erases what props survive the
// wrapper, so the component is given an explicit prop shape here rather than
// pulling in @types/plotly.js for a handful of fields.
type PlotProps = {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  style?: React.CSSProperties;
  useResizeHandler?: boolean;
};

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse bg-lineSoft" />,
}) as React.ComponentType<PlotProps>;

export const PALETTE = [
  "#2F4D73", "#C2703D", "#5B8C5A", "#8B5E83", "#B5A642",
  "#476D99", "#A14E3A", "#6B86A8", "#7D9B76", "#9C6B9E",
];

/**
 * A *fresh* base layout on every render.
 *
 * This must not be a shared module-level constant. Plotly mutates the layout
 * object it is handed — among other things it writes back the axis `type` it
 * auto-detected. A chart that passes no `xaxis` override would then hand Plotly
 * the shared object itself, and a date-based chart on the page would stamp
 * `type: 'date'` onto it for every chart rendered afterwards. Numeric axes then
 * read their values as milliseconds since the epoch: a return of -0.0356 becomes
 * 1970-01-01 00:59:59.9999, and the whole distribution collapses into one pixel.
 */
function baseLayout(): Partial<Plotly.Layout> {
  const axis = () => ({
    gridcolor: "#EEECE5",
    zerolinecolor: "#DDD8CD",
    linecolor: "#DDD8CD",
    automargin: true,
  });
  return {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Inter, system-ui, sans-serif", size: 11, color: "#3A4152" },
    margin: { l: 56, r: 18, t: 28, b: 42 },
    hovermode: "x unified",
    showlegend: true,
    legend: {
      orientation: "h",
      y: -0.18,
      x: 0,
      font: { size: 10 },
      bgcolor: "transparent",
    },
    xaxis: axis(),
    yaxis: axis(),
  };
}

const CONFIG: Partial<Plotly.Config> = {
  displaylogo: false,
  responsive: true,
  // Keep zoom, pan, box-select and PNG export; drop the lasso and the autoscale
  // duplicates that just crowd the bar.
  modeBarButtonsToRemove: ["lasso2d", "select2d", "toggleSpikelines"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

function merge(a: any, b: any): any {
  const out = { ...a };
  for (const [k, v] of Object.entries(b ?? {})) {
    out[k] =
      v && typeof v === "object" && !Array.isArray(v) && a[k] && typeof a[k] === "object"
        ? merge(a[k], v)
        : v;
  }
  return out;
}

export function Chart({
  data,
  layout,
  height = 320,
  numericX = false,
  episodes,
  className = "",
}: {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height?: number;
  /** Crisis bands for a date axis: pass the result of episodeLayout(). */
  episodes?: { shapes: any[]; annotations: any[] };
  /** Pin the x-axis to linear. Set it wherever x carries returns, quantiles or
   *  z-scores, so Plotly cannot mistake small numbers for epoch milliseconds. */
  numericX?: boolean;
  className?: string;
}) {
  const merged = merge(baseLayout(), { ...layout, height });
  if (numericX) merged.xaxis = { ...merged.xaxis, type: "linear" };
  if (episodes) {
    // Prepended, so a caller's own shapes still draw on top of the bands.
    merged.shapes = [...episodes.shapes, ...(merged.shapes ?? [])];
    merged.annotations = [...episodes.annotations, ...(merged.annotations ?? [])];
    if (episodes.annotations.length) {
      merged.margin = { ...merged.margin, t: Math.max(merged.margin?.t ?? 0, 20) };
    }
  }

  return (
    <div className={className} style={{ height }}>
      <Plot
        data={data}
        layout={merged}
        config={CONFIG}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

/** Panel with a title, an optional caption, and room for controls on the right. */
export function Panel({
  title,
  caption,
  actions,
  children,
  index = 0,
  className = "",
}: {
  title: string;
  caption?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  /** Position in a group, used only to stagger the entrance animation. */
  index?: number;
  className?: string;
}) {
  return (
    <section
      className={`enter rounded border border-line bg-panel px-4 py-3.5
                  shadow-[0_1px_2px_rgba(42,47,58,0.04)] ${className}`}
      style={{ animationDelay: `${Math.min(index, 8) * 45}ms` }}
    >
      <header className="mb-2 flex items-start justify-between gap-4">
        <div>
          <h2 className="text-2xs font-semibold uppercase tracking-label text-muted">
            {title}
          </h2>
          {caption && <p className="mt-1 text-[11px] leading-snug text-muted">{caption}</p>}
        </div>
        {actions && <div className="shrink-0">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

/**
 * Stationarity verdict as a dot.
 *
 * Was a PASS/WARN chip, which meant forty saturated green rectangles shouting over
 * the factor names they were annotating — the loudest element on the page carrying
 * the least information. The dot keeps the signal (a warn or a fail still catches
 * the eye against a field of green) and returns the emphasis to the names. The
 * reason moves into the tooltip, where it was already going unread on the chip.
 */
export function VerdictBadge({
  verdict,
  title,
  showLabel = false,
}: {
  verdict: string | null | undefined;
  title?: string;
  showLabel?: boolean;
}) {
  const tone =
    verdict === "pass" ? "bg-pass"
    : verdict === "warn" ? "bg-warn"
    : verdict === "fail" ? "bg-fail"
    : "bg-line";
  const text =
    verdict === "pass" ? "text-pass"
    : verdict === "warn" ? "text-warn"
    : verdict === "fail" ? "text-fail"
    : "text-muted";

  return (
    <span
      title={title ?? verdict ?? "no verdict"}
      className="inline-flex shrink-0 items-center gap-1"
      aria-label={`verdict: ${verdict ?? "none"}`}
    >
      <span className={`inline-block h-[7px] w-[7px] rounded-full ${tone}`} />
      {showLabel && (
        <span className={`text-2xs font-semibold uppercase tracking-wide ${text}`}>
          {verdict ?? "n/a"}
        </span>
      )}
    </span>
  );
}

/** Placeholder with the shape of the content it is standing in for. */
export function Skeleton({
  height = 16,
  width = "100%",
  className = "",
}: {
  height?: number | string;
  width?: number | string;
  className?: string;
}) {
  return <div className={`skeleton ${className}`} style={{ height, width }} />;
}

export function ChartSkeleton({ height = 280 }: { height?: number }) {
  return (
    <div style={{ height }} className="flex flex-col justify-end gap-1.5 py-3">
      {[38, 62, 47, 80, 55, 70].map((h, i) => (
        <Skeleton key={i} height={7} width={`${h}%`} />
      ))}
    </div>
  );
}

/**
 * A labelled statistic.
 *
 * `size="hero"` is for the two or three figures a page is actually about; the rest
 * stay at the default so the hierarchy means something. Pass `animate` with a raw
 * number and a formatter to have it count up on first appearance.
 */
export function Stat({
  label,
  value,
  hint,
  tone,
  size = "default",
  animate,
  format,
}: {
  label: string;
  value?: ReactNode;
  hint?: string;
  /**
   * What the figure says about the model, not what it looks like. `warn` is the
   * amber middle the battery needs: a flagged diagnostic nothing is gated on is
   * neither a pass nor a failure, and colouring it red would claim the series is
   * unusable when it is not. See lib/verdict.ts for which outcome earns which.
   */
  tone?: "good" | "warn" | "bad" | "neutral";
  size?: "default" | "hero";
  /** Raw number to animate toward. When given, `format` renders it. */
  animate?: number | null;
  format?: (v: number) => string;
}) {
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
    <div className="min-w-0">
      <div className="text-2xs uppercase tracking-label text-muted">{label}</div>
      <div
        className={
          size === "hero"
            ? `hero-number font-sans text-[26px] font-semibold ${colour}`
            : `font-mono text-sm font-semibold tabular-nums ${colour}`
        }
      >
        {shown}
      </div>
      {hint && <div className="mt-0.5 text-[10px] leading-tight text-muted">{hint}</div>}
    </div>
  );
}
