"use client";

/**
 * Which forecast to draw against which realised volatility.
 *
 * A forecast has to be drawn against a measure of what it forecasts. The h-day
 * forecast is an average over the next h days, so its pair is the realised
 * volatility of those h days, both on the forecast date. The daily forecast is
 * for one day, whose only unbiased measure - the squared return - is too noisy to
 * read as a line, so its pair is the five days centred on that day. Drawing the
 * daily forecast against an h-day window mixes the two, and the window's
 * placement then decides how far one line seems to lead the other.
 */
export type VolPair = "horizon" | "daily";

export function VolPairToggle({ value, onChange, horizon }: {
  value: VolPair;
  onChange: (v: VolPair) => void;
  horizon: number;
}) {
  const options: [VolPair, string][] = [["horizon", `${horizon}-day`], ["daily", "Daily"]];
  return (
    <div role="tablist" aria-label="Forecast and realised pair"
         className="inline-flex rounded border border-line bg-white p-0.5">
      {options.map(([v, label]) => (
        <button key={v} role="tab" aria-selected={value === v} onClick={() => onChange(v)}
                className={"rounded px-2.5 py-0.5 text-[11px] transition " + (value === v
                  ? "bg-navy font-medium text-white"
                  : "text-navy hover:bg-lineSoft/60")}>
          {label}
        </button>
      ))}
    </div>
  );
}

/** The caption for each pair, in the chart's own terms. */
export function volPairCaption(pair: VolPair, horizon: number, what = "volatility"): string {
  return pair === "horizon"
    ? `The ${horizon}-day forecast made on each date against the realised ${what} of the ${horizon} trading days after it, both drawn on the forecast date: the pairing the bias statistic scores. The forecast cannot see a shock coming, so realised rises first.`
    : `The next-day forecast against the realised ${what} of the five days centred on the day it is for, both drawn on that day: close to a point estimate of the day's ${what}, which a single squared return is too noisy to show.`;
}
