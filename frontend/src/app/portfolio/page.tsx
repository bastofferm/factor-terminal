"use client";

/**
 * Portfolio Optimization: not built yet.
 *
 * The section exists in the navigation before it exists as a tool so that its
 * place in the order is settled — after the risk model and the return forecasts,
 * because it consumes both. This page says what it will do and what it is
 * waiting on, rather than leaving a dead tab.
 */

import Link from "next/link";
import { Panel } from "@/components/Chart";

const INPUTS = [
  { href: "/matrix", label: "Covariance & PCA",
    what: "Factor covariance and specific risk — the risk side of every objective." },
  { href: "/loadings", label: "Loadings Lab",
    what: "Security betas — what factor-exposure constraints would bound." },
  { href: "/returns", label: "Ten-Year Forecasts",
    what: "Expected returns and their bands — the return side, or the views in Black–Litterman." },
  { href: "/risk#portfolio", label: "Risk Lens · Portfolio",
    what: "The existing portfolio risk run — how a constructed portfolio would be scored." },
];

export default function PortfolioPage() {
  return (
    <div className="max-w-[880px] space-y-4">
      <Panel
        title="Portfolio Optimization"
        caption="In development. Nothing on this page computes anything yet."
      >
        <p className="text-[12.5px] leading-relaxed text-navy">
          This section will construct portfolios from the two models the terminal
          already has: the factor risk model and the ten-year return forecasts. Which
          techniques are worth building — minimum variance, risk parity,
          constrained mean-variance, Black–Litterman — depends on what the data can
          support, and in particular on whether the security attributes are granular
          enough to bound weights by asset class, country, industry, factor exposure
          and ESG criteria.
        </p>
        <p className="mt-2 text-[12.5px] leading-relaxed text-muted">
          That assessment is being written up in{" "}
          <span className="font-mono text-[11.5px]">
            Documentation/portfolio-optimization-plan.md
          </span>
          . As with the rest of the terminal, any optimizer will be judged out of
          sample on the portfolio it produces, not on its in-sample fit.
        </p>
      </Panel>

      <Panel title="What it will build on" index={1}>
        <ul className="divide-y divide-lineSoft">
          {INPUTS.map((i) => (
            <li key={i.href} className="flex items-baseline gap-4 py-2">
              <Link href={i.href}
                    className="w-[170px] shrink-0 text-[12px] font-medium text-navy2
                               hover:text-navy hover:underline">
                {i.label}
              </Link>
              <span className="text-[12px] text-muted">{i.what}</span>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  );
}
