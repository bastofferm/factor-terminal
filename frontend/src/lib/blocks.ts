/**
 * One accent colour per factor block, used everywhere a factor appears.
 *
 * The point is navigational rather than decorative. With forty factors on one
 * screen, a consistent colour per block turns the sidebar into a map and lets
 * "this security's risk is mostly rates" be read off a chart instead of inferred
 * from a table.
 *
 * The nine hues are spaced around the wheel but held at a muted saturation so they
 * sit inside the paper palette rather than fighting it, and their lightness values
 * are spread far enough apart to stay distinguishable in greyscale — these pages
 * get printed.
 */

export interface BlockStyle {
  id: string;
  label: string;
  colour: string;
  /** Same hue at low opacity, for row tints and heatmap rails. */
  tint: string;
}

export const BLOCKS: Record<string, BlockStyle> = {
  equity:     { id: "equity",     label: "Equity",      colour: "#2F4D73", tint: "#2F4D7314" },
  style:      { id: "style",      label: "Style",       colour: "#6B5B95", tint: "#6B5B9514" },
  rates:      { id: "rates",      label: "Rates",       colour: "#1F6F5C", tint: "#1F6F5C14" },
  credit:     { id: "credit",     label: "Credit",      colour: "#A14E3A", tint: "#A14E3A14" },
  fx:         { id: "fx",         label: "FX",          colour: "#B5892B", tint: "#B5892B14" },
  commodity:  { id: "commodity",  label: "Commodity",   colour: "#7D6134", tint: "#7D613414" },
  volatility: { id: "volatility", label: "Volatility",  colour: "#8C3A5E", tint: "#8C3A5E14" },
  liquidity:  { id: "liquidity",  label: "Liquidity",   colour: "#3E7A8C", tint: "#3E7A8C14" },
  arp:        { id: "arp",        label: "Alt. Premia", colour: "#5B7A3A", tint: "#5B7A3A14" },
};

const FALLBACK: BlockStyle = {
  id: "other", label: "Other", colour: "#6F7890", tint: "#6F789014",
};

export function blockStyle(blockId: string | null | undefined): BlockStyle {
  return (blockId && BLOCKS[blockId]) || FALLBACK;
}

export function blockColour(blockId: string | null | undefined): string {
  return blockStyle(blockId).colour;
}

/** Block order for anything that groups by block, matching ref_factor_block.sort_order. */
export const BLOCK_ORDER = [
  "equity", "style", "rates", "credit", "fx",
  "commodity", "volatility", "liquidity", "arp",
];

/**
 * Colour a list of factors by their block, but keep repeated blocks legible by
 * darkening successive members. Without this, six rates factors on one chart are
 * six identical green lines.
 */
export function seriesColours(
  factorIds: string[],
  blockOf: (id: string) => string | null | undefined,
): Record<string, string> {
  const seen: Record<string, number> = {};
  const out: Record<string, string> = {};
  for (const id of factorIds) {
    const block = blockOf(id) ?? "other";
    const n = (seen[block] = (seen[block] ?? 0) + 1);
    out[id] = shade(blockColour(block), (n - 1) * 0.16);
  }
  return out;
}

/** Lighten a hex colour by `amount` (0–1). */
function shade(hex: string, amount: number): string {
  if (amount <= 0) return hex;
  const m = hex.replace("#", "");
  const num = parseInt(m, 16);
  const mix = (c: number) => Math.round(c + (235 - c) * Math.min(amount, 0.6));
  const r = mix((num >> 16) & 255);
  const g = mix((num >> 8) & 255);
  const b = mix(num & 255);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}
