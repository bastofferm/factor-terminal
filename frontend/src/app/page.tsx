"use client";

/**
 * Overview: the front door.
 *
 * Every other page answers a question you already have. This one is for the
 * reader who does not have one yet — a colleague sent them a link, or they are
 * deciding whether the model is worth their afternoon. So it states what the
 * model is, shows the state it is actually in today, and says which tab answers
 * which question.
 *
 * Nothing here is computed that is not on another page. The figures are counts,
 * spans and ranges over what `/api/meta/*` returns, and each says which. A
 * landing page that quietly derived its own numbers would be the easiest place
 * in the app for a figure to disagree with the model behind it.
 */

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Panel, Stat } from "@/components/Chart";
import { Sparkline } from "@/components/Sparkline";
import { Block, FactorMeta, Instrument, api, num } from "@/lib/api";
import { BLOCK_ORDER, blockStyle } from "@/lib/blocks";
import { EPISODES } from "@/lib/episodes";
import { usePublishSnapshot } from "@/lib/chat-context";

interface Health {
  status: string;
  as_of: string;
  active_factors: number;
}

export default function OverviewPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [factors, setFactors] = useState<FactorMeta[]>([]);
  const [inputs, setInputs] = useState<Instrument[]>([]);
  const [sparks, setSparks] = useState<Record<string, number[]>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // One round of requests, all of them cheap reference reads. Failures are
    // tolerated individually: a missing sparkline set should leave the page
    // standing, because the prose on it is the point and is not data-dependent.
    Promise.all([
      api.health().catch(() => null),
      api.blocks().catch(() => []),
      api.factors().catch(() => []),
      api.instruments("factor_input").catch(() => []),
      api.factorSparklines("orth").catch(() => ({ points: 0, series: {} })),
    ])
      .then(([h, b, f, i, s]) => {
        setHealth(h as Health | null);
        setBlocks(b as Block[]);
        setFactors(f as FactorMeta[]);
        setInputs(i as Instrument[]);
        setSparks((s as { series: Record<string, number[]> }).series ?? {});
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const panel = useMemo(() => summarise(factors), [factors]);

  usePublishSnapshot(
    factors.length
      ? {
          page_is: "the overview / landing page",
          as_of: health?.as_of ?? null,
          active_factors: health?.active_factors ?? factors.length,
          blocks: blocks.map((b) => ({ id: b.block_id, name: b.name,
                                       n_factors: b.n_factors })),
          panel_first_date: panel.first,
          panel_last_date: panel.last,
          longest_factor_observations: panel.maxObs,
          factor_inputs_tracked: inputs.length,
          verdicts: panel.verdicts,
        }
      : null
  );

  return (
    <div className="space-y-5">
      <Masthead health={health} factors={factors} />

      {error && (
        <p className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
          Could not reach the API: {error}. The description below still applies;
          the figures do not.
        </p>
      )}

      <Figures panel={panel} health={health} inputs={inputs.length} />
      <Coverage blocks={blocks} factors={factors} panel={panel} />
      <Blocks blocks={blocks} factors={factors} sparks={sparks} />
      <Pipeline />
      <Directory />
      <Limits />
    </div>
  );
}

// --- the model, in one line ----------------------------------------------

/**
 * One expression, typeset once KaTeX has arrived.
 *
 * The front door should not wait on a few hundred kilobytes of layout engine and
 * fonts to say what the model is, and this page uses three short expressions
 * where the factor profile uses hundreds. So the ASCII renders immediately — it
 * is the same string the CLI prints — and is replaced in place when the chunk
 * lands. Both come from `formula.py`'s vocabulary, so the swap changes the
 * typography and nothing else.
 */
function TeX({ latex, plain }: { latex: string; plain: string }) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    import("@/components/Math")
      .then((m) => {
        if (live) setHtml(m.render(latex, false));
      })
      .catch(() => {
        /* Leave the ASCII standing. It is not a degraded rendering, just a
           plainer one, and it says exactly the same thing. */
      });
    return () => {
      live = false;
    };
  }, [latex]);

  if (!html) return <span className="font-mono">{plain}</span>;
  return <span className="tex" dangerouslySetInnerHTML={{ __html: html }} />;
}

function Masthead({ health, factors }: { health: Health | null; factors: FactorMeta[] }) {
  const k = health?.active_factors ?? factors.length;
  const live = health?.status === "ok";

  return (
    <section className="enter rounded border border-line bg-panel px-6 py-6
                        shadow-[0_1px_2px_rgba(42,47,58,0.04)]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-[760px]">
          <div className="label">Multi-asset factor model</div>
          <h1 className="mt-1.5 text-[26px] font-semibold leading-tight tracking-tight
                         text-navy">
            Return-based risk, built from series you can open and check
          </h1>
          <p className="mt-2.5 text-[13px] leading-relaxed text-muted">
            {k ? <b className="text-navy">{k} daily factors</b> : "Daily factors"} across
            nine blocks, each one an excess return rather than a score;
            residualised down a hierarchy so their loadings mean something
            individually; shrunk into a single covariance matrix; and measured
            against what actually happened.
          </p>
        </div>

        <div className="shrink-0 text-right">
          <div className="flex items-center justify-end gap-1.5">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full
                          ${live ? "bg-pass" : "bg-muted"}`}
            />
            <span className="text-2xs uppercase tracking-label text-muted">
              {live ? "API live" : "API unreachable"}
            </span>
          </div>
          <div className="mt-1 font-mono text-[12px] text-navy">
            {health?.as_of ? `as of ${health.as_of}` : "—"}
          </div>
        </div>
      </div>

      {/* The model stated, rather than described. An institutional reader gets
          more from one line of notation than from a paragraph about it. */}
      <div className="mt-5 overflow-x-auto rounded border border-lineSoft bg-canvas
                      px-4 py-3">
        <div className="eqn text-navy">
          <TeX
            latex={`r_{i,t} = \\alpha_i + \\sum_{k=1}^{${k || "K"}} \\beta_{i,k}\\,` +
                   `\\tilde{f}_{k,t} + \\varepsilon_{i,t}`}
            plain={`r_it = a_i + sum_k b_ik * f~_kt + e_it`}
          />
        </div>
      </div>
      <p className="mt-1.5 text-[11px] leading-snug text-muted">
        <TeX latex={String.raw`\tilde{f}_{k,t}`} plain="f~_kt" /> is the
        orthogonalised factor, not the raw one — which is why{" "}
        <TeX latex={String.raw`\beta_{i,k}`} plain="b_ik" /> reads as
        &ldquo;over and above everything above it in the hierarchy&rdquo;. Both panels
        are available throughout; the Factor Explorer shows a factor against its
        own residual.
      </p>
    </section>
  );
}

// --- four figures ---------------------------------------------------------

function Figures({
  panel, health, inputs,
}: {
  panel: PanelSummary;
  health: Health | null;
  inputs: number;
}) {
  const years =
    panel.first && panel.last
      ? (new Date(panel.last).getTime() - new Date(panel.first).getTime()) /
        (365.25 * 24 * 3600 * 1000)
      : null;

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Card index={0}>
        <Stat
          label="Active factors"
          size="hero"
          animate={health?.active_factors ?? null}
          format={(v) => String(Math.round(v))}
          hint="Across nine blocks, every one an excess return"
        />
      </Card>
      <Card index={1}>
        <Stat
          label="Years of history"
          size="hero"
          animate={years}
          format={(v) => num(v, 1)}
          hint={panel.first && panel.last ? `${panel.first} → ${panel.last}` : undefined}
        />
      </Card>
      <Card index={2}>
        <Stat
          label="Days, longest factor"
          size="hero"
          animate={panel.maxObs}
          format={(v) => Math.round(v).toLocaleString("en-US")}
          hint="Trading days on the orthogonalised panel"
        />
      </Card>
      <Card index={3}>
        <Stat
          label="Inputs tracked"
          size="hero"
          animate={inputs || null}
          format={(v) => String(Math.round(v))}
          hint="Priced instruments feeding factor construction"
        />
      </Card>
    </div>
  );
}

function Card({ children, index }: { children: React.ReactNode; index: number }) {
  return (
    <div
      className="enter rounded border border-line bg-panel px-4 py-3
                 shadow-[0_1px_2px_rgba(42,47,58,0.04)]"
      style={{ animationDelay: `${index * 45}ms` }}
    >
      {children}
    </div>
  );
}

// --- coverage -------------------------------------------------------------

/**
 * When each block's history starts, against the episodes it has lived through.
 *
 * This answers a question that comes up before any other: how far back can I
 * estimate? The blocks do not all start together — the ETFs behind the style
 * block were not listed in 2003 — and a window that looks generous on one block
 * silently drops another.
 */
function Coverage({
  blocks, factors, panel,
}: {
  blocks: Block[];
  factors: FactorMeta[];
  panel: PanelSummary;
}) {
  const rows = useMemo(() => {
    if (!panel.first || !panel.last) return [];
    return ordered(blocks).map((b) => {
      const mine = factors.filter((f) => f.block_id === b.block_id && f.first_date);
      const first = mine.length
        ? mine.reduce((a, f) => (f.first_date! < a ? f.first_date! : a), mine[0].first_date!)
        : null;
      return { block: b, first, n: mine.length };
    }).filter((r) => r.first);
  }, [blocks, factors, panel]);

  if (!rows.length || !panel.first || !panel.last) return null;

  const lo = new Date(panel.first).getTime();
  const hi = new Date(panel.last).getTime();
  const span = hi - lo || 1;
  const at = (iso: string) => ((new Date(iso).getTime() - lo) / span) * 100;

  const years: number[] = [];
  for (let y = new Date(panel.first).getFullYear() + 1;
       y <= new Date(panel.last).getFullYear(); y += 4) {
    years.push(y);
  }

  return (
    <Panel
      title="Coverage, and the regimes it spans"
      caption={
        <>
          Each bar starts at the first observation of the earliest factor in that
          block. The shaded columns are well-known market episodes, marked as
          editorial context — they are not something the model detected, and
          nothing in the pipeline reads them.
        </>
      }
      index={4}
    >
      <div className="relative mt-1">
        {/* Episode bands run the full height, behind everything. */}
        <div className="pointer-events-none absolute inset-0">
          {EPISODES.map((ep) => {
            const x0 = Math.max(at(ep.start), 0);
            const x1 = Math.min(at(ep.end), 100);
            if (x1 <= x0) return null;
            return (
              <div
                key={ep.label}
                className="absolute top-0 bottom-[18px]"
                style={{ left: `${x0}%`, width: `${Math.max(x1 - x0, 0.35)}%`,
                         background: "#8C3A2E", opacity: 0.1 }}
                title={`${ep.label} · ${ep.start} → ${ep.end}`}
              />
            );
          })}
        </div>

        <div className="relative space-y-[3px]">
          {rows.map((r) => {
            const bs = blockStyle(r.block.block_id);
            const x = at(r.first!);
            return (
              <div key={r.block.block_id} className="flex items-center gap-2">
                <div className="w-[104px] shrink-0 text-right text-[10px] text-muted">
                  {bs.label}
                </div>
                <div className="relative h-[14px] flex-1">
                  {/* The start year rides the left edge of the bar rather than
                      sitting in a column of its own: parked at a fixed x it
                      annotates nothing, and on a bar that starts in 2012 it
                      reads as the year the bar ends. */}
                  <div
                    className="absolute inset-y-0 flex items-center rounded-[2px] px-1"
                    style={{ left: `${x}%`, right: 0, background: bs.colour,
                             opacity: 0.88 }}
                    title={`${r.block.name}: ${r.n} factor${r.n === 1 ? "" : "s"}, from ${r.first}`}
                  >
                    <span className="font-mono text-[9px] leading-none text-white/85">
                      {r.first!.slice(0, 4)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Year ticks, on the same scale as the bars above them. */}
        <div className="relative ml-[112px] mt-1.5 h-[14px] border-t
                        border-lineSoft">
          {years.map((y) => (
            <span
              key={y}
              className="absolute top-0.5 -translate-x-1/2 font-mono text-[9px] text-muted"
              style={{ left: `${at(`${y}-01-01`)}%` }}
            >
              {y}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted">
        {EPISODES.map((ep) => (
          <span key={ep.label}>
            <span className="mr-1 inline-block h-2 w-2 -mb-px rounded-[1px]"
                  style={{ background: "#8C3A2E", opacity: 0.25 }} />
            {ep.label}
          </span>
        ))}
      </div>
    </Panel>
  );
}

// --- the nine blocks ------------------------------------------------------

function Blocks({
  blocks, factors, sparks,
}: {
  blocks: Block[];
  factors: FactorMeta[];
  sparks: Record<string, number[]>;
}) {
  const grouped = useMemo(
    () =>
      ordered(blocks).map((b) => ({
        block: b,
        members: factors
          .filter((f) => f.block_id === b.block_id)
          .sort((a, c) => a.hierarchy_level - c.hierarchy_level),
      })),
    [blocks, factors]
  );

  if (!grouped.length) return null;

  return (
    <Panel
      title="Nine blocks, forty factors"
      caption="Each block groups factors that price the same kind of risk. The
               sparklines are the cumulative orthogonalised return over each
               factor's own history, so they show shape rather than level."
      index={5}
    >
      <div className="mt-1 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {grouped.map(({ block, members }) => {
          const bs = blockStyle(block.block_id);
          const vols = members.map((m) => m.vol_pct).filter((v): v is number => v != null);
          const shown = members.slice(0, 4);
          return (
            <Link
              key={block.block_id}
              href="/factors"
              className="group rounded border border-line bg-canvas px-3 py-2.5
                         transition hover:border-navy2"
            >
              <div className="flex items-baseline justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-2.5 rounded-[2px]"
                        style={{ background: bs.colour }} />
                  <span className="text-[12px] font-semibold text-navy">
                    {block.name}
                  </span>
                </div>
                <span className="font-mono text-[10px] text-muted">
                  {block.n_factors}f
                  {vols.length > 1 &&
                    ` · ${num(Math.min(...vols), 0)}–${num(Math.max(...vols), 0)}% vol`}
                </span>
              </div>

              <p className="mt-1 text-[11px] leading-snug text-muted">
                {block.description}
              </p>

              <div className="mt-2 space-y-[3px] border-t border-lineSoft pt-1.5">
                {shown.map((m) => (
                  <div key={m.factor_id} className="flex items-center gap-2">
                    <span className="flex-1 truncate font-mono text-[10px] text-muted">
                      {m.factor_id}
                    </span>
                    <Sparkline values={sparks[m.factor_id]} colour={bs.colour}
                               width={46} height={12} />
                  </div>
                ))}
                {members.length > shown.length && (
                  <div className="text-[10px] text-muted">
                    + {members.length - shown.length} more
                  </div>
                )}
              </div>
            </Link>
          );
        })}
      </div>
    </Panel>
  );
}

// --- how a number gets here ----------------------------------------------

const PIPELINE: Array<{ step: string; body: React.ReactNode }> = [
  {
    step: "Ingest",
    body: <>Prices and quoted levels land in the warehouse from Yahoo, FRED and
           the central-bank SDMX feeds. Each series resumes from its own
           watermark, so a lagging one can catch up.</>,
  },
  {
    step: "Make stationary",
    body: <>Prices become log total returns; yields become duration-scaled bond
           returns; levels are differenced or standardised. Every input is put on
           a return footing before the model sees it.</>,
  },
  {
    step: "Build factors",
    body: <>Forty construction rules — single, spread, basket, curve shape, carry,
           trend — evaluated over named instruments. The rule is stored, so the
           formula on screen cannot drift from the series beside it.</>,
  },
  {
    step: "Orthogonalise",
    body: <>Each factor is regressed on the ones above it in the hierarchy over a
           trailing window and refitted periodically. The window ends the day
           before, so no factor value contains its own future.</>,
  },
  {
    step: "Covariance",
    body: <>The panel is shrunk toward a structured target and reported with its
           condition number and shrinkage intensity, because a matrix you cannot
           invert is a risk number you cannot trust.</>,
  },
  {
    step: "Loadings and risk",
    body: <>A security is regressed on the factors to get its betas; the betas and
           the matrix give a predicted volatility, which is then checked against
           the realised one.</>,
  },
];

function Pipeline() {
  return (
    <Panel
      title="How a number gets here"
      caption="Six stages, each one runnable and inspectable on its own. The
               Operations tab shows what each reads and writes."
      index={6}
    >
      <ol className="mt-1 grid gap-2.5 md:grid-cols-2 xl:grid-cols-3">
        {PIPELINE.map((s, i) => (
          <li key={s.step} className="rounded border border-lineSoft bg-canvas
                                      px-3 py-2">
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-2xs text-navy3">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="text-[12px] font-semibold text-navy">{s.step}</span>
            </div>
            <p className="mt-1 text-[11px] leading-snug text-muted">{s.body}</p>
          </li>
        ))}
      </ol>
    </Panel>
  );
}

// --- where to look --------------------------------------------------------

const PAGES = [
  { href: "/factors", label: "Factor Explorer",
    when: "You want to know what a factor is, how it is built, and whether its series behaves." },
  { href: "/raw", label: "Raw Explorer",
    when: "You suspect the input rather than the model, and want the series before it was touched." },
  { href: "/matrix", label: "Covariance & PCA",
    when: "You need the correlation structure, the block risk budget, or the conditioning of the matrix." },
  { href: "/loadings", label: "Loadings Lab",
    when: "You want one security's betas, and whether they hold still under a different window." },
  { href: "/risk", label: "Risk Lens",
    when: "You want the forecast tested: bias, Mincer-Zarnowitz, VaR coverage." },
  { href: "/ops", label: "Operations",
    when: "You want to refresh the data, or see what the last run actually did." },
  { href: "/health", label: "Data Health",
    when: "Something looks wrong and you want to know whether an input went stale or died." },
];

function Directory() {
  return (
    <Panel
      title="Where to look"
      caption="Seven screens, each answering one kind of question."
      index={7}
    >
      <div className="mt-1 grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {PAGES.map((p) => (
          <Link
            key={p.href}
            href={p.href}
            className="group rounded border border-lineSoft bg-canvas px-3 py-2
                       transition hover:border-navy2"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-[12px] font-semibold text-navy">{p.label}</span>
              <span className="text-[12px] text-navy3 transition
                               group-hover:translate-x-0.5">
                →
              </span>
            </div>
            <p className="mt-0.5 text-[11px] leading-snug text-muted">{p.when}</p>
          </Link>
        ))}
      </div>
    </Panel>
  );
}

// --- what it does not do --------------------------------------------------

function Limits() {
  return (
    <Panel
      title="What this model does not do"
      caption="Stated here rather than discovered later. Each is recorded in the
               methodology the assistant reads, and it will raise them unprompted
               when they bear on a question."
      index={8}
    >
      <ul className="mt-1 space-y-1.5 text-[11px] leading-snug text-muted">
        <li>
          <b className="text-navy">It forecasts risk, not return.</b> There is no
          alpha model here, and no view on direction. A loading is a description
          of exposure, not a recommendation.
        </li>
        <li>
          <b className="text-navy">FX carry is approximated</b> from policy-rate
          differentials, because the warehouse holds no forward points. Covered
          interest parity makes that a stand-in, not the thing itself.
        </li>
        <li>
          <b className="text-navy">The style block is long-only proxies.</b> ETF
          returns are not the academic long-short factors, and a loading on them
          should not be read as one.
        </li>
        <li>
          <b className="text-navy">Daily data, linear betas.</b> A convex payoff —
          the variance premium is one — is only first-order captured, and nothing
          here models a term structure of risk.
        </li>
      </ul>
      <p className="mt-2.5 border-t border-lineSoft pt-2 text-[11px] text-muted">
        The full write-up, including every construction formula, is in{" "}
        <span className="font-mono text-navy">Documentation/factor-model-paper.pdf</span>;
        the formulas alone are in{" "}
        <span className="font-mono text-navy">factor-formulas.md</span>. Each
        factor&rsquo;s own formula is on its profile, behind the{" "}
        <span className="font-mono text-navy">i</span> beside it in the Factor
        Explorer.
      </p>
    </Panel>
  );
}

// --- shared -------------------------------------------------------------

interface PanelSummary {
  first: string | null;
  last: string | null;
  maxObs: number | null;
  verdicts: Record<string, number>;
}

function summarise(factors: FactorMeta[]): PanelSummary {
  const dated = factors.filter((f) => f.first_date && f.last_date);
  const verdicts: Record<string, number> = {};
  for (const f of factors) {
    const v = f.verdict ?? "none";
    verdicts[v] = (verdicts[v] ?? 0) + 1;
  }
  return {
    first: dated.length
      ? dated.reduce((a, f) => (f.first_date! < a ? f.first_date! : a), dated[0].first_date!)
      : null,
    last: dated.length
      ? dated.reduce((a, f) => (f.last_date! > a ? f.last_date! : a), dated[0].last_date!)
      : null,
    maxObs: factors.length
      ? Math.max(...factors.map((f) => f.n_obs ?? 0)) || null
      : null,
    verdicts,
  };
}

/** Blocks in the model's own order, with anything unrecognised after them. */
function ordered(blocks: Block[]): Block[] {
  return [...blocks].sort((a, b) => {
    const ia = BLOCK_ORDER.indexOf(a.block_id);
    const ib = BLOCK_ORDER.indexOf(b.block_id);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}
