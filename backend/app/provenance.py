"""Where a series came from, and what it is called at the far end.

`IVV` says nothing about whether the number was fetched from Yahoo this morning or
mirrored out of a warehouse table that was last refreshed in June. `ECB:BUND_10Y`
says nothing about the ECB. An analyst checking a suspicious figure needs the
provider, the identifier that provider knows it by, and the span of history
actually held — and none of the three is recoverable from an instrument id.

Shared because two pages answer the same question: the factor profile resolves a
construction into its inputs, and the Raw Explorer describes a single series. Two
tables of provider labels would drift, and the one that drifted would be the one
nobody was looking at.
"""

from __future__ import annotations

from typing import Any

# What `ref_instrument.source_table` means in an analyst's terms.
#
# The distinction that matters is fetched-here versus mirrored: the nightly ingest
# covers the factor universe only, so anything arriving over the warehouse view is
# as current as the last warehouse refresh and no more. That is the difference
# between a `live` instrument and a `stale` one on the Raw Explorer, and this is
# where a reader finds out which they are looking at.
SOURCE_TABLE_LABEL: dict[str, str] = {
    "yahoo": "Yahoo Finance",
    "fact_cross_asset": "Yahoo Finance, mirrored via the xbrl_sec warehouse",
    "fact_prices_us": "US equity prices, mirrored via the xbrl_sec warehouse",
}

# Level-series ids are namespaced by provider: FRED:DGS10, ECB:BUND_10Y.
SOURCE_PREFIX_LABEL: dict[str, str] = {
    "FRED": "Federal Reserve Economic Data (FRED)",
    "ECB": "European Central Bank",
    "MOF_JP": "Japanese Ministry of Finance",
    "BOJ": "Bank of Japan",
    "SNB": "Swiss National Bank",
}


def instrument_source(source_table: str | None) -> str:
    if not source_table:
        return "unknown"
    return SOURCE_TABLE_LABEL.get(source_table, source_table)


def level_series_source(series_id: str | None) -> str:
    if not series_id:
        return "unknown"
    prefix = str(series_id).split(":", 1)[0]
    return SOURCE_PREFIX_LABEL.get(prefix, prefix)


def walk_inputs(node: Any, out: list[str] | None = None) -> list[str]:
    """Every identifier named anywhere in a construction's inputs, in order.

    Keys that describe the recipe rather than its ingredients are skipped: a curve
    factor's `shape` is "slope", not a series, and resolving it against the
    instrument table would come back empty and look like a missing input.
    """
    out = [] if out is None else out
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for v in node:
            walk_inputs(v, out)
    elif isinstance(node, dict):
        for key, v in node.items():
            if key in _RECIPE_KEYS:
                continue
            walk_inputs(v, out)
    return list(dict.fromkeys(out))


_RECIPE_KEYS = {
    "shape", "transform", "window", "lookback", "skip", "vol_window", "sparse",
    "scale_to_vol", "tenor", "tenors", "carry", "excess", "sign", "inverted",
}
