"use client";

/**
 * What a factor actually is, and where its data comes from.
 *
 * The sidebar shows forty identifiers. `liq_risk_off` is not self-explanatory, and
 * neither is the fact that it nets three haven assets against two risk assets and
 * then removes equity, rates and credit from what is left. Opening that up matters
 * more than any chart on the page: a loading is uninterpretable until you know what
 * it is a loading on.
 */

import { useEffect, useState } from "react";
import { blockStyle } from "@/lib/blocks";
import { num, pct } from "@/lib/api";

interface Profile {
  factor_id: string;
  name: string;
  block_id: string;
  block_name: string;
  block_description: string;
  hierarchy_level: number;
  method: string;
  method_prose: string;
  inputs: Record<string, unknown>;
  note: string | null;
  instruments: Array<{
    instrument_id: string; source_ticker: string; source: string;
    asset_class: string | null; currency: string; is_total_return: boolean;
    is_live: boolean; first_obs: string | null; last_obs: string | null;
    n_obs: number | null; notes: string | null;
  }>;
  level_series: Array<{
    series_id: string; name: string | null; source: string; category: string;
    unit: string; transform: string; tenor_years: number | null;
    first_obs: string | null; last_obs: string | null; n_obs: number | null;
  }>;
  orthogonalised_against: Array<{ factor_id: string; name: string; block_id: string }>;
  coverage: {
    first_date: string | null; last_date: string | null;
    n_obs: number | null; vol_ann: number | null;
  } | null;
}

export function FactorProfile({
  factorId,
  onClose,
}: {
  factorId: string;
  onClose: () => void;
}) {
  const [p, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setProfile(null);
    setError(null);
    fetch(`/api/factors/${factorId}/profile`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then(setProfile)
      .catch((e) => setError(String(e.message ?? e)));
  }, [factorId]);

  // Escape closes, which is what anyone tries first on a panel like this.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const bs = blockStyle(p?.block_id);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/20 p-6
                 backdrop-blur-[1px]"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`${factorId} details`}
    >
      <div
        className="enter mt-10 max-h-[82vh] w-full max-w-[620px] overflow-y-auto
                   rounded border border-line bg-panel shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header
          className="sticky top-0 flex items-start justify-between gap-4 border-b
                     border-line px-5 py-3"
          style={{ background: bs.tint, backdropFilter: "blur(6px)" }}
        >
          <div>
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5 rounded-[2px]"
                style={{ background: bs.colour }}
              />
              <span className="font-mono text-[13px] font-semibold text-navy">
                {factorId}
              </span>
            </div>
            {p && (
              <div className="mt-0.5 text-[12px] text-muted">
                {p.name} · {p.block_name}
              </div>
            )}
          </div>
          <button className="btn-ghost shrink-0" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="space-y-4 px-5 py-4 text-[12px] leading-snug">
          {error && <p className="text-fail">Could not load the profile: {error}</p>}
          {!p && !error && <p className="text-muted">Loading…</p>}

          {p && (
            <>
              <Section title="What it is">
                <p>{p.method_prose}</p>
                {p.note && <p className="mt-1.5 italic text-muted">{p.note}</p>}
              </Section>

              {p.instruments.length > 0 && (
                <Section title="Built from">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Ticker</th>
                        <th className="th">Class</th>
                        <th className="th">Source</th>
                        <th className="th">Coverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {p.instruments.map((i) => (
                        <tr key={i.instrument_id} className="border-t border-lineSoft">
                          <td className="cell">
                            {i.instrument_id}
                            {!i.is_live && (
                              <span className="ml-1 text-2xs text-fail">dead</span>
                            )}
                            {!i.is_total_return && (
                              <span className="ml-1 text-2xs text-warn">price-return</span>
                            )}
                          </td>
                          <td className="cell font-sans">{i.asset_class ?? "—"}</td>
                          <td className="cell font-sans text-muted">{i.source}</td>
                          <td className="cell text-muted">
                            {i.first_obs?.slice(0, 7)} → {i.last_obs?.slice(0, 7)}
                            {i.n_obs ? ` · ${i.n_obs.toLocaleString()}` : ""}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Section>
              )}

              {p.level_series.length > 0 && (
                <Section
                  title="Level series"
                  hint="Quoted levels, not returns. Each is made stationary by the
                        transform shown before it can enter the model."
                >
                  <table className="w-full border-collapse">
                    <thead>
                      <tr>
                        <th className="th">Series</th>
                        <th className="th">Provider</th>
                        <th className="th">Transform</th>
                        <th className="th">Tenor</th>
                      </tr>
                    </thead>
                    <tbody>
                      {p.level_series.map((l) => (
                        <tr key={l.series_id} className="border-t border-lineSoft">
                          <td className="cell" title={l.name ?? ""}>{l.series_id}</td>
                          <td className="cell font-sans text-muted">{l.source}</td>
                          <td className="cell">{l.transform}</td>
                          <td className="cell text-muted">
                            {l.tenor_years ? `${num(l.tenor_years, 2)}y` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Section>
              )}

              {p.orthogonalised_against.length > 0 && (
                <Section title="Overlap removed">
                  <p>
                    This factor overlaps with{" "}
                    {p.orthogonalised_against.map((o, i) => (
                      <span key={o.factor_id}>
                        {i > 0 && (i === p.orthogonalised_against.length - 1 ? " and " : ", ")}
                        <span className="font-mono text-navy">{o.factor_id}</span>
                      </span>
                    ))}
                    , which sit above it in the hierarchy. The model regresses it on
                    them and keeps what is left.
                  </p>
                  <p className="mt-1.5 text-muted">
                    Without that step the two would be strongly correlated, the
                    regression design would be ill-conditioned, and their loadings
                    would come out as large offsetting numbers that mean nothing
                    individually. The <b>Incremental</b> view on the page shows this
                    leftover series — what this factor adds beyond the ones above it.
                    The <b>Factor</b> view shows the series itself, which is what it
                    earned.
                  </p>
                </Section>
              )}

              {p.coverage && (
                <Section title="The factor series">
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <Fact label="From" value={p.coverage.first_date ?? "—"} />
                    <Fact label="To" value={p.coverage.last_date ?? "—"} />
                    <Fact
                      label="Observations"
                      value={p.coverage.n_obs?.toLocaleString() ?? "—"}
                    />
                    <Fact label="Ann. volatility" value={pct(p.coverage.vol_ann)} />
                  </div>
                  <p className="mt-2 text-muted">
                    Hierarchy level {p.hierarchy_level} · method{" "}
                    <span className="font-mono">{p.method}</span> · all figures on the
                    raw excess-return series.
                  </p>
                </Section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({
  title, hint, children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h3 className="label mb-1">{title}</h3>
      {hint && <p className="mb-1.5 text-[11px] text-muted">{hint}</p>}
      {children}
    </section>
  );
}

function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <div className="text-2xs uppercase tracking-label text-muted">{label}</div>
      <div className="font-mono text-[12px] text-navy">{value}</div>
    </div>
  );
}
