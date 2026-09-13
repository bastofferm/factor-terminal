/**
 * A tiny inline-SVG sparkline.
 *
 * Deliberately not Plotly. Forty Plotly instances in a scrolling sidebar would cost
 * more than the rest of the page put together, and a sparkline needs nothing Plotly
 * provides — no axes, no hover, no legend. This is a single path element.
 */

export function Sparkline({
  values,
  colour = "#2F4D73",
  width = 54,
  height = 16,
  strokeWidth = 1,
  showZero = true,
  className = "",
}: {
  values: number[] | undefined;
  colour?: string;
  width?: number;
  height?: number;
  strokeWidth?: number;
  /** Draw a hairline at zero, so "ended below where it started" is visible. */
  showZero?: boolean;
  className?: string;
}) {
  if (!values || values.length < 2) {
    return <span className={className} style={{ width, height, display: "inline-block" }} />;
  }

  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const span = max - min || 1;
  const pad = strokeWidth / 2;
  const usable = height - strokeWidth;

  const x = (i: number) => (i / (values.length - 1)) * width;
  const y = (v: number) => pad + (1 - (v - min) / span) * usable;

  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const zeroY = y(0);
  const last = values[values.length - 1];

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
      aria-hidden="true"
      style={{ display: "block", overflow: "visible" }}
    >
      {showZero && zeroY > pad && zeroY < height - pad && (
        <line
          x1={0} x2={width} y1={zeroY} y2={zeroY}
          stroke="#DDD8CD" strokeWidth={0.5} strokeDasharray="2 2"
        />
      )}
      <path
        d={d}
        fill="none"
        stroke={colour}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      {/* A dot on the final value; the end of the line is what the eye looks for. */}
      <circle cx={x(values.length - 1)} cy={y(last)} r={1.4} fill={colour} />
    </svg>
  );
}
