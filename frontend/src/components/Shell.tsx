"use client";

/**
 * The application chrome: navigation, the assistant toggle, and the docked panel.
 *
 * Client-side because it needs the current path — both to mark the active tab and
 * to give the assistant its page context — and because the panel holds state that
 * must survive navigation between pages.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { ChatPanel } from "@/components/ChatPanel";
import { ChatProvider, useChat } from "@/lib/chat-context";

type Page = { href: string; label: string };
type Group = { label: string; pages: Page[]; planned?: boolean };

// The landing page stands on its own, ahead of the sections: it introduces all
// four, so it belongs to none of them, and it has no second row.
const HOME: Page = { href: "/", label: "Overview" };

// Four sections in the order the work happens in: the inputs, the risk model
// built on them, the ten-year return forecasts that sit beside it, and the
// portfolios constructed from both. Data is the state of the inputs - whether
// they are current, and the run that makes them so. Looking at the series
// themselves is Raw Explorer, which sits beside Factor Explorer because the two
// answer the same question before and after the model touches the data.
const GROUPS: Group[] = [
  {
    label: "Data",
    pages: [
      { href: "/health", label: "Data Health" },
      { href: "/ops", label: "Operations" },
    ],
  },
  {
    label: "Factor Risk",
    pages: [
      { href: "/factors", label: "Factor Explorer" },
      { href: "/raw", label: "Raw Explorer" },
      { href: "/matrix", label: "Covariance & PCA" },
      { href: "/loadings", label: "Loadings Lab" },
      { href: "/risk", label: "Risk Lens" },
    ],
  },
  {
    label: "Return Forecasts",
    pages: [{ href: "/returns", label: "Ten-Year Forecasts" }],
  },
  {
    label: "Portfolio Optimization",
    planned: true,
    pages: [{ href: "/portfolio", label: "Roadmap" }],
  },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname() || "/";
  return (
    <ChatProvider page={pathname}>
      <Chrome pathname={pathname}>{children}</Chrome>
    </ChatProvider>
  );
}

function Chrome({ pathname, children }: { pathname: string; children: React.ReactNode }) {
  const { open, setOpen, snapshot, snapshots } = useChat();
  const current = GROUPS.find((g) => g.pages.some((p) => p.href === pathname));

  // The page last open in each section, so going back to a section returns to
  // where you were in it rather than to its first page. Held here because the
  // chrome survives navigation; a reload starts over, which is fine.
  const [last, setLast] = useState<Record<string, string>>({});
  useEffect(() => {
    if (current) setLast((l) => ({ ...l, [current.label]: pathname }));
  }, [current, pathname]);

  return (
    <>
      <header className="sticky top-0 z-30 border-b border-line bg-panel/95 backdrop-blur">
        <div className="mx-auto grid max-w-[1600px] grid-cols-[auto_1fr_auto] items-center
                        gap-x-6 px-5">
          <Link href="/" className="shrink-0 py-2.5">
            <span className="font-mono text-[13px] font-bold tracking-tight text-navy">
              FACTOR&nbsp;TERMINAL
            </span>
          </Link>

          <nav aria-label="Sections" className="flex gap-1 overflow-x-auto no-scrollbar">
            <Link
              href={HOME.href}
              aria-current={pathname === HOME.href ? "page" : undefined}
              className={`whitespace-nowrap rounded px-2.5 py-1 text-[12px] font-medium transition
                ${pathname === HOME.href
                  ? "bg-navy text-white"
                  : "text-muted hover:bg-lineSoft hover:text-navy"}`}
            >
              {HOME.label}
            </Link>
            {GROUPS.map((g) => {
              const active = g === current;
              return (
                <Link
                  key={g.label}
                  href={last[g.label] ?? g.pages[0].href}
                  aria-current={active ? "true" : undefined}
                  className={`flex items-center gap-1.5 whitespace-nowrap rounded px-2.5 py-1
                    text-[12px] font-medium transition
                    ${active
                      ? "bg-navy text-white"
                      : "text-muted hover:bg-lineSoft hover:text-navy"}`}
                >
                  {g.label}
                  {g.planned && (
                    <span className={`rounded-sm px-1 text-[9px] font-semibold uppercase
                      tracking-wide ${active ? "bg-white/20 text-white" : "bg-lineSoft text-navy3"}`}>
                      soon
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          <button
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            className={`shrink-0 rounded border px-2.5 py-1 text-[12px] transition
              ${open
                ? "border-navy bg-navy text-white"
                : "border-line bg-white text-navy hover:border-navy2"}`}
          >
            Ask
          </button>

          {/* The pages of the open section, under the section tabs rather than under
              the logo, so the eye reads them as belonging to the row above. */}
          {current && (
            <nav aria-label={current.label}
                 className="col-start-2 col-end-4 -mt-0.5 flex gap-4 overflow-x-auto
                            no-scrollbar">
              {current.pages.map((p) => {
                const active = p.href === pathname;
                return (
                  <Link
                    key={p.href}
                    href={p.href}
                    aria-current={active ? "page" : undefined}
                    className={`whitespace-nowrap border-b-2 px-0.5 pb-1.5 pt-1 text-[12px]
                      transition
                      ${active
                        ? "border-navy text-navy"
                        : "border-transparent text-muted hover:text-navy"}`}
                  >
                    {p.label}
                  </Link>
                );
              })}
            </nav>
          )}
        </div>
      </header>

      <main
        className="mx-auto max-w-[1600px] px-5 py-5 transition-[padding] duration-200"
        style={{ paddingRight: open ? 436 : undefined }}
      >
        {children}
      </main>

      <ChatPanel open={open} onClose={() => setOpen(false)} snapshot={snapshot}
                 snapshots={snapshots} />
    </>
  );
}
