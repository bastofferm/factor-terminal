"use client";

/**
 * Typesetting, for the pages that show the model's own mathematics.
 *
 * `formula.py` renders every expression twice from one description: LaTeX and
 * ASCII. The ASCII is what the CLI prints and what the assistant quotes, because
 * it has to survive a cp1252 console; a browser has no such excuse and gets the
 * LaTeX. Both come from the same call, so the two cannot disagree — and the
 * ASCII stays on the element title, one hover away, for anyone who wants the
 * string the rest of the system passes around.
 *
 * KaTeX and its fonts are a few hundred kilobytes. Whatever imports this pays
 * for them, so the factor profile is loaded on the click that opens it rather
 * than with its page.
 */

import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/** KaTeX, with the options every call on this panel shares. */
export function render(latex: string, display: boolean): string {
  return katex.renderToString(latex, {
    displayMode: display,
    throwOnError: true,
    strict: "error",
    // The input is our own registry rather than anything a reader supplies, but
    // a link command has no business in a factor definition either.
    trust: false,
  });
}

/**
 * A display equation, in its own boxed line or lines.
 *
 * `formula.py` renders every expression twice from one description: LaTeX and
 * ASCII. The ASCII is what the CLI prints and what the assistant quotes, because
 * it has to survive a cp1252 console; this panel has a browser and shows the
 * LaTeX. Both come from the same call, so the two cannot disagree — and the ASCII
 * is still on the element title, one hover away, for anyone who wants the string
 * the rest of the system passes around.
 *
 * The parts are split on a blank line, which is how `formula.py` separates the
 * lines of a multi-line expression; 44 of the registry's 80 expressions have at
 * least one. LaTeX treats a blank line inside mathematics as ordinary space, so
 * rendering the string whole runs the definition of a factor straight into the
 * definition of the estimator below it.
 *
 * Every expression in the registry was checked against KaTeX in strict mode
 * before this was wired up. The fallback is for what that check cannot cover: a
 * formula added later that KaTeX refuses. Dropping back to monospace is a much
 * smaller failure than rendering a red error where the mathematics should be.
 */
export function Equation({ latex, plain }: { latex?: string; plain: string }) {
  const parts = useMemo(() => {
    if (!latex) return null;
    try {
      return latex
        .split(/\n\s*\n/)
        .map((s) => s.trim())
        .filter(Boolean)
        .map((s) => render(s, true));
    } catch {
      return null;
    }
  }, [latex]);

  const box = "overflow-x-auto rounded border border-lineSoft bg-canvas px-3 py-2";

  if (!parts?.length) {
    return (
      <pre className={`${box} font-mono text-[11.5px] leading-relaxed text-navy`}>
        {plain}
      </pre>
    );
  }
  return (
    <div className={`eqn space-y-1.5 ${box} text-navy`} title={plain}>
      {parts.map((html, k) => (
        <div key={k} dangerouslySetInnerHTML={{ __html: html }} />
      ))}
    </div>
  );
}

/** One symbol, set inline in running text or as a term in the symbol table. */
export function TeX({ latex, plain }: { latex?: string; plain: string }) {
  const html = useMemo(() => {
    if (!latex) return null;
    try {
      return render(latex, false);
    } catch {
      return null;
    }
  }, [latex]);

  if (!html) return <span className="font-mono">{plain}</span>;
  return (
    <span className="tex" title={plain} dangerouslySetInnerHTML={{ __html: html }} />
  );
}

type Token =
  | { html: string }   // typeset mathematics
  | { em: string }
  | { text: string };

const MATH_SPAN = /\$([^$]*)\$/g;
const EMPHASIS = /\*([^*]+)\*/g;

function tokenise(source: string, hasMath: boolean): Token[] {
  const out: Token[] = [];

  // `*emphasis*` is the only markup the registry's prose uses, and it is there
  // for the Markdown appendix — printed raw, it reads as stray asterisks.
  const pushText = (s: string) => {
    let last = 0;
    for (const m of s.matchAll(EMPHASIS)) {
      if (m.index > last) out.push({ text: s.slice(last, m.index) });
      out.push({ em: m[1] });
      last = m.index + m[0].length;
    }
    if (last < s.length) out.push({ text: s.slice(last) });
  };

  if (!hasMath) {
    pushText(source);
    return out;
  }

  let last = 0;
  for (const m of source.matchAll(MATH_SPAN)) {
    if (m.index > last) pushText(source.slice(last, m.index));
    out.push({ html: render(m[1], false) });
    last = m.index + m[0].length;
  }
  if (last < source.length) pushText(source.slice(last));
  return out;
}

/**
 * A sentence that may carry mathematics.
 *
 * Half the notation in this panel lives in prose rather than in the equations —
 * a step that says "subtract the overnight rate" and then writes it out, a
 * symbol whose meaning is a convexity formula. `formula.py` gives those a second
 * rendering with the mathematics in `$...$` spans; this splits on them and
 * typesets what is inside.
 *
 * A span that KaTeX refuses drops the whole sentence back to the ASCII, rather
 * than leaving one run of raw LaTeX in the middle of an otherwise typeset line.
 */
export function Prose({ text, tex }: { text: string; tex?: string | null }) {
  const tokens = useMemo(() => {
    if (!tex) return tokenise(text, false);
    try {
      return tokenise(tex, true);
    } catch {
      return tokenise(text, false);
    }
  }, [text, tex]);

  return (
    <>
      {tokens.map((t, i) =>
        "html" in t ? (
          <span key={i} className="tex" dangerouslySetInnerHTML={{ __html: t.html }} />
        ) : "em" in t ? (
          <em key={i}>{t.em}</em>
        ) : (
          <span key={i}>{t.text}</span>
        )
      )}
    </>
  );
}
