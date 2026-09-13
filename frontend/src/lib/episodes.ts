/**
 * Market episodes, drawn as faint bands behind the time-series charts.
 *
 * This is editorial context, not model output. Nothing here comes from the
 * pipeline and nothing here feeds it — the dates are the conventional ones and the
 * bands exist so that a spike in a volatility chart reads as "the 2020 crash"
 * rather than as an unexplained excursion. Every panel that shows them says so in
 * its caption, because a shaded region on a risk chart could otherwise be mistaken
 * for something the model detected.
 */

export interface Episode {
  label: string;
  /** Short form, used when the band is too narrow for the full label. */
  short: string;
  start: string;
  end: string;
}

export const EPISODES: Episode[] = [
  { label: "Global financial crisis", short: "GFC",
    start: "2008-09-15", end: "2009-03-09" },
  { label: "Euro sovereign crisis", short: "EUR",
    start: "2011-07-01", end: "2012-07-26" },
  { label: "Oil collapse", short: "Oil",
    start: "2014-11-27", end: "2016-02-11" },
  { label: "COVID crash", short: "COVID",
    start: "2020-02-19", end: "2020-04-07" },
  { label: "Inflation shock", short: "Inflation",
    start: "2022-01-03", end: "2022-10-14" },
  { label: "Regional bank stress", short: "SVB",
    start: "2023-03-08", end: "2023-05-01" },
];

/**
 * Plotly `shapes` and `annotations` for a date axis.
 *
 * Bands are clipped to the data range so a chart starting in 2012 does not draw a
 * GFC label off-canvas, and dropped entirely when they would cover less than a few
 * percent of the width — a two-pixel sliver with a label attached is noise.
 */
export function episodeLayout(
  firstDate: string | undefined,
  lastDate: string | undefined,
  opts: { labels?: boolean } = {},
): { shapes: any[]; annotations: any[] } {
  const shapes: any[] = [];
  const annotations: any[] = [];
  if (!firstDate || !lastDate) return { shapes, annotations };

  const lo = new Date(firstDate).getTime();
  const hi = new Date(lastDate).getTime();
  const span = hi - lo;
  if (!(span > 0)) return { shapes, annotations };

  for (const ep of EPISODES) {
    const s = Math.max(new Date(ep.start).getTime(), lo);
    const e = Math.min(new Date(ep.end).getTime(), hi);
    if (e <= s) continue;

    const width = (e - s) / span;
    if (width < 0.012) continue;

    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0: new Date(s).toISOString().slice(0, 10),
      x1: new Date(e).toISOString().slice(0, 10),
      y0: 0, y1: 1,
      fillcolor: "#8C3A2E", opacity: 0.055,
      line: { width: 0 }, layer: "below",
    });

    if (opts.labels !== false && width > 0.035) {
      annotations.push({
        x: new Date(s + (e - s) / 2).toISOString().slice(0, 10),
        y: 1, xref: "x", yref: "paper",
        text: ep.short, showarrow: false,
        yanchor: "bottom", yshift: 1,
        font: { size: 9, color: "#A98077" },
      });
    }
  }
  return { shapes, annotations };
}
