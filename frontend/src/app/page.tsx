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
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // One round of requests, all of them cheap reference reads. Failures are
    // tolerated individually: a missing block list should leave the page
    // standing, because the prose on it is the point and is not data-dependent.
    Promise.all([
      api.health().catch(() => null),
      api.blocks().catch(() => []),
      api.factors().catch(() => []),
      api.instruments("factor_input").catch(() => []),
    ])
      .then(([h, b, f, i]) => {
        setHealth(h as Health | null);
        setBlocks(b as Block[]);
        setFactors(f as FactorMeta[]);
        setInputs(i as Instrument[]);
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
      <Hero health={health} factors={factors} panel={panel} />

      {error && (
        <p className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
          Could not reach the API: {error}. The description above still applies;
          the figures do not.
        </p>
      )}

      <Figures panel={panel} health={health} inputs={inputs.length} />
      <Coverage blocks={blocks} factors={factors} panel={panel} />
      <Pipeline />
      <Directory />
    </div>
  );
}

// --- the hero -------------------------------------------------------------

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

/**
 * The masthead.
 *
 * Grey rather than paper so the front door reads as a cover and the working
 * screens behind it read as working screens. The blue is the same navy the rest
 * of the app uses for anything that matters — the card changes the ground, not
 * the voice.
 *
 * It carries the model as an equation, the state it is in today, and just
 * enough of the coverage panel below to make the span concrete. Everything it
 * shows is repeated somewhere with more room, on purpose: a masthead that was
 * the only place a figure appeared would be a masthead nobody could check.
 */
function Hero({
  health, factors, panel,
}: {
  health: Health | null;
  factors: FactorMeta[];
  panel: PanelSummary;
}) {
  const k = health?.active_factors ?? factors.length;
  const live = health?.status === "ok";
  const years = spanYears(panel);

  return (
    <section
      className="enter relative overflow-hidden rounded border px-6 py-6
                 shadow-[0_1px_3px_rgba(42,47,58,0.07)] sm:px-8 sm:py-7"
      style={{
        // A warm grey, so it sits with the paper palette rather than cutting a
        // cold rectangle out of it.
        background: "linear-gradient(152deg, #EDECE9 0%, #E4E2DE 58%, #DBD9D4 100%)",
        borderColor: "#D2CFC8",
      }}
    >
      {/* A hairline rule in the block colours: the nine hues that key every
          chart in the app, introduced here before they mean anything. */}
      <div className="absolute inset-x-0 top-0 flex h-[2px]">
        {BLOCK_ORDER.map((id, i) => (
          <div
            key={id}
            className="sweep flex-1"
            style={{ background: blockStyle(id).colour, opacity: 0.55,
                     animationDelay: `${120 + i * 55}ms` }}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-start justify-between gap-x-8 gap-y-3">
        <div className="max-w-[720px]">
          <div className="rise label" style={{ animationDelay: "40ms" }}>
            Multi-asset factor model
          </div>
          <h1
            className="rise mt-2 text-[28px] font-semibold leading-[1.15]
                       tracking-tight text-navy sm:text-[32px]"
            style={{ animationDelay: "90ms" }}
          >
            Return-based risk, built from series
            <br className="hidden sm:inline" /> you can open and check
          </h1>
          <p
            className="rise mt-3 max-w-[640px] text-[13px] leading-relaxed"
            style={{ animationDelay: "150ms", color: "#5A6275" }}
          >
            {k ? <b className="text-navy">{k} daily factors</b> : "Daily factors"} across
            nine blocks, each one an excess return rather than a score;
            residualised down a hierarchy so their loadings mean something
            individually; shrunk into a single covariance matrix; and measured
            against what actually happened.
          </p>
        </div>

        <div className="rise shrink-0 text-right" style={{ animationDelay: "200ms" }}>
          <div className="flex items-center justify-end gap-1.5">
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full
                          ${live ? "pulse bg-pass" : "bg-muted"}`}
            />
            <span className="text-2xs uppercase tracking-label"
                  style={{ color: "#5A6275" }}>
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
      <div
        className="rise mt-5 overflow-x-auto rounded border px-4 py-3"
        style={{ animationDelay: "260ms", background: "#F6F5F3CC",
                 borderColor: "#D2CFC8" }}
      >
        <div className="eqn text-navy">
          <TeX
            latex={`r_{i,t} = \\alpha_i + \\sum_{k=1}^{${k || "K"}} \\beta_{i,k}\\,` +
                   `\\tilde{f}_{k,t} + \\varepsilon_{i,t}`}
            plain="r_it = a_i + sum_k b_ik * f~_kt + e_it"
          />
        </div>
      </div>
      <p className="rise mt-1.5 text-[11px] leading-snug"
         style={{ animationDelay: "300ms", color: "#5A6275" }}>
        <TeX latex={String.raw`\tilde{f}_{k,t}`} plain="f~_kt" /> is the
        orthogonalised factor, not the raw one — which is why{" "}
        <TeX latex={String.raw`\beta_{i,k}`} plain="b_ik" /> reads as
        &ldquo;over and above everything above it in the hierarchy&rdquo;.
      </p>

      <HeroSpan panel={panel} years={years} />
    </section>
  );
}

/**
 * The span, and the regimes inside it.
 *
 * The smallest useful piece of the coverage panel further down: how long the
 * history is, and what it has been through. The bands are the same editorial
 * episodes marked there, and the names cycle so the reader notices they are
 * dates on a line rather than decoration.
 */
function HeroSpan({ panel, years }: { panel: PanelSummary; years: number | null }) {
  const [lit, setLit] = useState(0);
  const calm = useReducedMotion();

  useEffect(() => {
    if (calm || !panel.first) return;
    const t = setInterval(() => setLit((i) => (i + 1) % EPISODES.length), 2600);
    return () => clearInterval(t);
  }, [calm, panel.first]);

  if (!panel.first || !panel.last || years === null) return null;

  const lo = new Date(panel.first).getTime();
  const hi = new Date(panel.last).getTime();
  const span = hi - lo || 1;
  const at = (iso: string) => ((new Date(iso).getTime() - lo) / span) * 100;

  return (
    <div className="rise mt-5 border-t pt-4" style={{ animationDelay: "350ms",
                                                      borderColor: "#D2CFC8" }}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-[12px] text-navy">
          <b className="font-semibold">{num(years, 1)} years</b> of daily history
        </div>
        <div className="font-mono text-[10px]" style={{ color: "#5A6275" }}>
          {panel.first} → {panel.last}
        </div>
      </div>

      <div className="relative mt-2 h-[8px] overflow-hidden rounded-[2px]"
           style={{ background: "#CFCCC5" }}>
        <div className="sweep h-full w-full" style={{ animationDelay: "420ms" }}>
          {EPISODES.map((ep, i) => {
            const x0 = Math.max(at(ep.start), 0);
            const x1 = Math.min(at(ep.end), 100);
            if (x1 <= x0) return null;
            return (
              <div
                key={ep.label}
                className="absolute inset-y-0 transition-opacity duration-500"
                style={{ left: `${x0}%`, width: `${Math.max(x1 - x0, 0.5)}%`,
                         background: "#8C3A2E",
                         opacity: !calm && i === lit ? 0.85 : 0.4 }}
                title={`${ep.label} · ${ep.start} → ${ep.end}`}
              />
            );
          })}
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px]">
        <span style={{ color: "#5A6275" }}>Through</span>
        {EPISODES.map((ep, i) => (
          <span
            key={ep.label}
            className="transition-colors duration-500"
            style={{ color: !calm && i === lit ? "#2F4D73" : "#7B8194",
                     fontWeight: !calm && i === lit ? 600 : 400 }}
          >
            {ep.label}
          </span>
        ))}
      </div>
    </div>
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
          animate={spanYears(panel)}
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

// --- how a number gets here ----------------------------------------------

const PIPELINE: Array<{ step: string; body: string }> = [
  { step: "Ingest",
    body: "Prices and quoted levels from Yahoo, FRED and the central-bank feeds, each series resuming from its own watermark." },
  { step: "Make stationary",
    body: "Prices to log total returns, yields to duration-scaled bond returns, levels differenced or standardised." },
  { step: "Build factors",
    body: "Forty stored construction rules evaluated over named instruments, so the formula cannot drift from the series." },
  { step: "Orthogonalise",
    body: "Each factor residualised on the ones above it, over a trailing window that ends the day before." },
  { step: "Covariance",
    body: "Shrunk toward a structured target, reported with its condition number and shrinkage intensity." },
  { step: "Loadings and risk",
    body: "Betas for one security, a predicted volatility from them, then a check against the realised one." },
];

/**
 * The pipeline as one card rather than six.
 *
 * A rail with six stops reads as a sequence at a glance, which six separate
 * boxes did not — and the sequence is the content here. The Operations tab has
 * the same six stages with what each reads and writes; this is the summary that
 * sends you there.
 */
function Pipeline() {
  return (
    <section
      className="enter relative overflow-hidden rounded border px-5 py-5
                 shadow-[0_1px_2px_rgba(42,47,58,0.05)] sm:px-6"
      style={{
        background: "linear-gradient(160deg, #F2F1EE 0%, #EAE8E4 100%)",
        borderColor: "#D8D5CE",
        animationDelay: "180ms",
      }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="label">How a number gets here</h2>
        <Link href="/ops" className="text-[11px] text-navy2 hover:text-navy">
          Run it →
        </Link>
      </div>
      <p className="mt-1 text-[11px]" style={{ color: "#5A6275" }}>
        Six stages, each runnable and inspectable on its own.
      </p>

      <ol className="relative mt-5 grid gap-y-6 sm:grid-cols-3 xl:grid-cols-6">
        {/* The rail the stops sit on. Only drawn at six columns, where the
            stages really are one row: at three they wrap onto two, and a single
            horizontal line would run through the first row and abandon the
            second. It stops at the sixth circle rather than at the edge of the
            grid, because a rail that continues past the last stop suggests a
            seventh stage that does not exist. */}
        <div
          className="sweep pointer-events-none absolute left-[11px] top-[11px]
                     hidden h-px xl:block"
          style={{ background: "#C9C5BC", right: "calc(100% / 6 - 11px)",
                   animationDelay: "260ms" }}
        />

        {PIPELINE.map((s, i) => (
          <li key={s.step} className="rise relative pr-4"
              style={{ animationDelay: `${300 + i * 70}ms` }}>
            <div
              className="relative z-10 flex h-[22px] w-[22px] items-center
                         justify-center rounded-full border font-mono text-[10px]
                         font-semibold text-navy"
              style={{ background: "#FBFAF7", borderColor: "#C9C5BC" }}
            >
              {i + 1}
            </div>
            <div className="mt-2 text-[12px] font-semibold leading-tight text-navy">
              {s.step}
            </div>
            <p className="mt-1 text-[11px] leading-snug" style={{ color: "#5A6275" }}>
              {s.body}
            </p>
          </li>
        ))}
      </ol>
    </section>
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
      index={6}
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

function spanYears(panel: PanelSummary): number | null {
  if (!panel.first || !panel.last) return null;
  return (
    (new Date(panel.last).getTime() - new Date(panel.first).getTime()) /
    (365.25 * 24 * 3600 * 1000)
  );
}

/** Blocks in the model's own order, with anything unrecognised after them. */
function ordered(blocks: Block[]): Block[] {
  return [...blocks].sort((a, b) => {
    const ia = BLOCK_ORDER.indexOf(a.block_id);
    const ib = BLOCK_ORDER.indexOf(b.block_id);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}

/**
 * Whether the reader has asked for less motion.
 *
 * The CSS animations are handled by a media query in globals.css, but the
 * episode cycle is a timer in JavaScript and no stylesheet can switch that off.
 */
function useReducedMotion(): boolean {
  const [calm, setCalm] = useState(false);
  useEffect(() => {
    const q = window.matchMedia("(prefers-reduced-motion: reduce)");
    setCalm(q.matches);
    const on = () => setCalm(q.matches);
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return calm;
}
