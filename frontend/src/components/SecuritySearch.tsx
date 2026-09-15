"use client";

/**
 * Search the whole catalogue, not the handful already mirrored.
 *
 * The picker used to be a datalist over `ref_instrument` — 198 rows, almost all of
 * them factor inputs, and for a while not even containing the box's own default.
 * The warehouse can price 9,400 securities, and `sync_security` pulls any of them
 * on demand, so the list to search is the catalogue.
 *
 * A datalist is the wrong shape at that size: it cannot rank, cannot show why one
 * match beats another, and ships every option to the browser. This queries as you
 * type and shows what the API ranked — exact ticker, then ticker prefix, then a
 * name match, longest history first.
 */

import { useEffect, useRef, useState } from "react";
import { api, Security } from "@/lib/api";

const TYPE_LABEL: Record<string, string> = {
  equity: "equity", etf: "fund", index: "index",
  futures: "futures", fx: "fx", crypto: "crypto",
};

export function SecuritySearch({
  value, onChange, types = "equity,etf",
}: {
  value: string;
  onChange: (id: string) => void;
  types?: string;
}) {
  const [query, setQuery] = useState(value);
  const [results, setResults] = useState<Security[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  // Follow the value when it is set from outside, so the box does not keep showing
  // a stale query after the page picks a security for some other reason.
  useEffect(() => setQuery(value), [value]);

  /**
   * Debounced, and the response is dropped if a newer keystroke has already gone
   * out. Without the second guard a slow request for "a" can land after a fast one
   * for "aapl" and overwrite the better list.
   */
  useEffect(() => {
    if (!open) return;
    let current = true;
    const id = window.setTimeout(() => {
      api.securities(query, types)
        .then((r) => {
          if (!current) return;
          setResults(r.results);
          setTotal(r.total);
          setActive(0);
        })
        .catch(() => current && setResults([]));
    }, 160);
    return () => { current = false; window.clearTimeout(id); };
  }, [query, types, open]);

  // Click outside closes it. Mousedown rather than click, so selecting a result
  // is not cancelled by the close that a click would fire first.
  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, []);

  const choose = (s: Security) => {
    onChange(s.instrument_id);
    setQuery(s.instrument_id);
    setOpen(false);
  };

  return (
    <div className="relative" ref={box}>
      <input
        className="field mt-1 w-full font-mono"
        value={query}
        placeholder="AAPL, Toyota, gold…"
        onChange={(e) => {
          setQuery(e.target.value);
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (!open || !results.length) return;
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((i) => Math.min(i + 1, results.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((i) => Math.max(i - 1, 0));
          } else if (e.key === "Enter") {
            e.preventDefault();
            choose(results[active]);
          } else if (e.key === "Escape") {
            setOpen(false);
          }
        }}
        role="combobox"
        aria-expanded={open}
        aria-autocomplete="list"
      />

      {open && results.length > 0 && (
        <div className="absolute left-0 right-0 top-[calc(100%+2px)] z-40 max-h-[320px]
                        overflow-auto rounded border border-line bg-panel shadow-lg">
          {results.map((s, i) => (
            <button
              key={s.instrument_id}
              onMouseDown={(e) => { e.preventDefault(); choose(s); }}
              onMouseEnter={() => setActive(i)}
              className={`block w-full px-2 py-1.5 text-left transition ${
                i === active ? "bg-lineSoft" : ""}`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[12px] text-navy">
                  {s.instrument_id}
                </span>
                <span className="shrink-0 text-2xs uppercase tracking-label text-muted">
                  {TYPE_LABEL[s.security_type] ?? s.security_type}
                  {/*
                    A security with no local history is not unavailable — it just
                    costs one warehouse read before the first estimate, which the
                    endpoint does on its own. Saying so beats letting the first
                    click feel inexplicably slower than the rest.
                  */}
                  {!s.mirrored && (
                    <span className="ml-1 text-warn" title="Not stored yet; the first
                      estimate pulls its history from the warehouse">
                      · sync
                    </span>
                  )}
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-[11px] text-muted">
                  {s.name ?? s.ticker}
                </span>
                <span className="shrink-0 font-mono text-[10px] text-muted">
                  {s.n_obs ? `${s.n_obs.toLocaleString()}d` : ""}
                </span>
              </div>
              {s.sector && (
                <div className="truncate text-[10px] text-navy3">{s.sector}</div>
              )}
            </button>
          ))}
        </div>
      )}

      <p className="mt-1 text-[10px] leading-tight text-muted">
        {total !== null
          ? `${total.toLocaleString()} searchable. `
          : ""}
        Anything the warehouse prices; its history is pulled on first use.
      </p>
    </div>
  );
}
