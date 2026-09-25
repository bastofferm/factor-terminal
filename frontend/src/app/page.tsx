"use client";

/**
 * Overview: the front door.
 *
 * Every other page answers a question you already have. This one is for the
 * reader who does not have one yet — a colleague sent them a link, or they are
 * deciding whether the model is worth their afternoon. So it says what the model
 * is, shows the state it is in today, and points at the screen that answers each
 * kind of question.
 *
 * It is the one screen in the app that moves. Everywhere else a chart draws and
 * stops, because an analyst reading a volatility series does not want it
 * animating under them. Here the motion is the argument: a pipeline that runs
 * through its stages explains itself faster than six paragraphs about it.
 *
 * Nothing shown is computed that is not on another page. The figures are counts
 * and spans over what `/api/meta/*` returns, and each says which.
 */

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Stat } from "@/components/Chart";
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
    // One round of cheap reference reads. Failures are tolerated individually:
    // the prose is the point of this page and does not depend on any of them.
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
    <div className="space-y-4">
      <Hero health={health} factors={factors} panel={panel} />

      {error && (
        <p className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
          Could not reach the API: {error}. The description above still applies;
          the figures do not.
        </p>
      )}

      <Figures panel={panel} health={health} inputs={inputs.length} />
      <Pipeline />
      <Tiles />
    </div>
  );
}

// --- the hero -------------------------------------------------------------

/**
 * The masthead.
 *
 * Grey rather than paper so the front door reads as a cover and the working
 * screens behind it read as working screens. The blue is the same navy the rest
 * of the app uses for anything that matters — the card changes the ground, not
 * the voice.
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

  return (
    <section
      className="enter relative overflow-hidden rounded border px-6 py-7
                 shadow-[0_1px_3px_rgba(42,47,58,0.07)] sm:px-9 sm:py-9"
      style={{
        // A warm grey, so it sits with the paper palette rather than cutting a
        // cold rectangle out of it.
        background: "linear-gradient(152deg, #EDECE9 0%, #E4E2DE 58%, #DBD9D4 100%)",
        borderColor: "#D2CFC8",
      }}
    >
      {/* A hairline in the nine block colours: the hues that key every chart in
          the app, introduced here before they mean anything. */}
      <div className="absolute inset-x-0 top-0 flex h-[3px]">
        {BLOCK_ORDER.map((id, i) => (
          <div
            key={id}
            className="sweep flex-1"
            style={{ background: blockStyle(id).colour, opacity: 0.6,
                     animationDelay: `${120 + i * 55}ms` }}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-start justify-between gap-x-8 gap-y-3">
        <div className="max-w-[760px]">
          <div className="rise label" style={{ animationDelay: "40ms" }}>
            Multi-asset factor model
          </div>
          <h1
            className="rise mt-2 text-[30px] font-semibold leading-[1.12]
                       tracking-tight text-navy sm:text-[38px]"
            style={{ animationDelay: "90ms" }}
          >
            Return-based risk, built from series
            <br className="hidden sm:inline" /> you can open and check
          </h1>
          <p
            className="rise mt-3.5 max-w-[660px] text-[13.5px] leading-relaxed"
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

      <HeroSpan panel={panel} />
    </section>
  );
}

/**
 * The span, and the regimes inside it.
 *
 * How long the history is and what it has been through, as one bar. The bands
 * are the conventional episode dates — editorial context, not something the
 * model detected, and nothing in the pipeline reads them. The names cycle so a
 * reader notices they are dates on a line rather than decoration.
 */
function HeroSpan({ panel }: { panel: PanelSummary }) {
  const [lit, setLit] = useState(0);
  const calm = useReducedMotion();
  const years = spanYears(panel);

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
    <div className="rise mt-6 border-t pt-4" style={{ animationDelay: "280ms",
                                                      borderColor: "#D2CFC8" }}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="text-[12.5px] text-navy">
          <b className="font-semibold">{num(years, 1)} years</b> of daily history
        </div>
        <div className="font-mono text-[10px]" style={{ color: "#5A6275" }}>
          {panel.first} → {panel.last}
        </div>
      </div>

      <div className="relative mt-2 h-[9px] overflow-hidden rounded-[2px]"
           style={{ background: "#CFCCC5" }}>
        <div className="sweep h-full w-full" style={{ animationDelay: "360ms" }}>
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
                         opacity: !calm && i === lit ? 0.9 : 0.38 }}
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

const STEP_MS = 1900;

/**
 * The pipeline, running.
 *
 * A rail with six stops, walked one at a time on a loop. The animation is the
 * explanation: data moving through stages in order is the whole shape of the
 * thing, and a reader gets it from watching once without reading any of the six
 * captions.
 *
 * Hovering takes the wheel — pointing at a stage holds it, because someone who
 * has reached for a stage wants to read it, not to watch it hand over. The loop
 * resumes on leaving. Under prefers-reduced-motion nothing cycles and every
 * stage renders at full strength, which is also how it prints.
 */
function Pipeline() {
  const calm = useReducedMotion();
  const [active, setActive] = useState(0);
  const [held, setHeld] = useState<number | null>(null);

  useEffect(() => {
    if (calm || held !== null) return;
    const t = setInterval(() => setActive((i) => (i + 1) % PIPELINE.length), STEP_MS);
    return () => clearInterval(t);
  }, [calm, held]);

  const current = held ?? active;
  // With motion off there is no "current" stage, so none is dimmed.
  const isOn = (i: number) => calm || i <= current;
  const filled = calm ? 100 : (current / (PIPELINE.length - 1)) * 100;

  return (
    <section
      className="enter relative overflow-hidden rounded border px-5 py-5
                 shadow-[0_1px_2px_rgba(42,47,58,0.05)] sm:px-7 sm:py-6"
      style={{
        background: "linear-gradient(160deg, #F2F1EE 0%, #EAE8E4 100%)",
        borderColor: "#D8D5CE",
        animationDelay: "180ms",
      }}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="label">How a number gets here</h2>
        <Link href="/ops" className="text-[11px] text-navy2 transition
                                     hover:text-navy">
          Run it →
        </Link>
      </div>
      <p className="mt-1 text-[11px]" style={{ color: "#5A6275" }}>
        Six stages, each runnable and inspectable on its own.
        {!calm && " Hover a stage to hold it."}
      </p>

      <ol
        className="relative mt-6 grid gap-y-7 sm:grid-cols-3 xl:grid-cols-6"
        onMouseLeave={() => setHeld(null)}
      >
        {/* The rail, and the progress along it. Drawn only at six columns,
            where the stages really are one row: at three they wrap onto two and
            a single horizontal line would run through the first and abandon the
            second. It stops at the sixth circle rather than at the edge of the
            grid, because a rail continuing past the last stop suggests a
            seventh stage that does not exist. */}
        <div
          className="pointer-events-none absolute left-[13px] top-[13px] hidden
                     h-px xl:block"
          style={{ background: "#CFCBC2", right: "calc(100% / 6 - 13px)" }}
        >
          <div
            className="h-full transition-[width] ease-out"
            style={{ width: `${filled}%`, background: "#2F4D73",
                     transitionDuration: `${STEP_MS * 0.55}ms` }}
          />
        </div>

        {PIPELINE.map((s, i) => (
          <li
            key={s.step}
            className="rise relative pr-4"
            style={{ animationDelay: `${300 + i * 70}ms` }}
            onMouseEnter={() => setHeld(i)}
          >
            <div
              className="relative z-10 flex h-[26px] w-[26px] items-center
                         justify-center rounded-full border font-mono text-[11px]
                         font-semibold transition-all duration-300"
              style={{
                background: isOn(i) ? "#2F4D73" : "#FBFAF7",
                borderColor: isOn(i) ? "#2F4D73" : "#CFCBC2",
                color: isOn(i) ? "#FFFFFF" : "#8A8FA0",
                transform: !calm && i === current ? "scale(1.14)" : "scale(1)",
                boxShadow: !calm && i === current
                  ? "0 0 0 4px rgba(47,77,115,0.13)" : "none",
              }}
            >
              {i + 1}
            </div>
            <div
              className="mt-2.5 text-[12.5px] font-semibold leading-tight
                         transition-colors duration-300"
              style={{ color: isOn(i) ? "#2F4D73" : "#8A8FA0" }}
            >
              {s.step}
            </div>
            {/* The body does not dim with the walk. The circle, the title and
                the rail already carry where it has got to, and a reader who
                arrives mid-cycle should not find five of the six stages greyed
                out — the captions are the content, not the chrome. */}
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
  { href: "/factors", label: "Factor Explorer", glyph: "line",
    when: "What a factor is, how it is built, and whether its series behaves." },
  { href: "/raw", label: "Raw Explorer", glyph: "bars",
    when: "The input series before the model touched it." },
  { href: "/matrix", label: "Covariance", glyph: "grid",
    when: "Correlation structure, block risk budget, conditioning." },
  { href: "/loadings", label: "Loadings Lab", glyph: "beta",
    when: "One security's betas, and whether they hold still." },
  { href: "/risk", label: "Risk Lens", glyph: "bell",
    when: "The forecast tested: bias, Mincer-Zarnowitz, VaR coverage." },
  { href: "/ops", label: "Operations", glyph: "refresh",
    when: "Refresh the data, or see what the last run did." },
  { href: "/health", label: "Data Health", glyph: "pulse",
    when: "Whether an input went stale or died." },
];

/** Seven tiles on one line, each a way in. */
function Tiles() {
  return (
    <div>
      <h2 className="label mb-2 px-0.5">Where to look</h2>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7">
        {PAGES.map((p, i) => (
          <Link
            key={p.href}
            href={p.href}
            title={p.when}
            className="enter group relative overflow-hidden rounded border
                       border-line bg-panel px-3 pb-3 pt-3.5 transition-all
                       duration-200 hover:-translate-y-0.5 hover:border-navy2
                       hover:shadow-[0_3px_10px_rgba(42,47,58,0.09)]"
            style={{ animationDelay: `${240 + i * 45}ms` }}
          >
            {/* An accent that draws itself across the top on hover. */}
            <span
              className="absolute inset-x-0 top-0 h-[2px] origin-left scale-x-0
                         bg-navy transition-transform duration-300
                         group-hover:scale-x-100"
            />
            <div className="flex items-start justify-between">
              <Glyph name={p.glyph} />
              <span
                className="text-[12px] leading-none text-navy3 transition-transform
                           duration-200 group-hover:translate-x-1 group-hover:text-navy2"
              >
                →
              </span>
            </div>
            <div className="mt-2 text-[12px] font-semibold leading-tight text-navy">
              {p.label}
            </div>
            <p className="mt-1 text-[10.5px] leading-snug text-muted">{p.when}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

/**
 * A mark per tile.
 *
 * Abstract on purpose. Each one echoes the shape of what the page draws — a
 * series, a matrix, a distribution — without standing for any particular
 * figure, because a glyph that looked like data would be data nobody computed.
 */
function Glyph({ name }: { name: string }) {
  const common = {
    width: 20, height: 20, viewBox: "0 0 20 20", fill: "none",
    stroke: "currentColor", strokeWidth: 1.4,
    strokeLinecap: "round" as const, strokeLinejoin: "round" as const,
  };
  return (
    <span className="block text-navy3 transition-colors duration-200
                     group-hover:text-navy">
      <svg {...common} aria-hidden="true">
        {name === "line" && <path d="M2 14 L6 8 L9 11 L13 4 L18 9" />}
        {name === "bars" && (
          <>
            <path d="M3 17 V11" /><path d="M7.5 17 V6" />
            <path d="M12 17 V13" /><path d="M16.5 17 V8" />
          </>
        )}
        {name === "grid" && (
          <>
            <rect x="2.5" y="2.5" width="15" height="15" rx="1.5" />
            <path d="M7.5 2.5 V17.5" /><path d="M12.5 2.5 V17.5" />
            <path d="M2.5 7.5 H17.5" /><path d="M2.5 12.5 H17.5" />
          </>
        )}
        {name === "beta" && (
          <>
            <path d="M3 5 H15" /><path d="M3 10 H10" /><path d="M3 15 H17" />
          </>
        )}
        {name === "bell" && (
          <>
            <path d="M2 16 C6 16 6 5 10 5 C14 5 14 16 18 16" />
            <path d="M2 16 H18" />
          </>
        )}
        {name === "refresh" && (
          <>
            <path d="M16.5 10 A6.5 6.5 0 1 1 14 5" />
            <path d="M14 1.5 V5.5 H10" />
          </>
        )}
        {name === "pulse" && (
          <path d="M2 10 H6 L8 5 L11 15 L13 10 H18" />
        )}
      </svg>
    </span>
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

/**
 * Whether the reader has asked for less motion.
 *
 * The CSS animations are handled by a media query in globals.css, but the
 * pipeline walk and the episode cycle are timers in JavaScript, and no
 * stylesheet can switch those off.
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
