"use client";

/**
 * The application chrome: navigation, the assistant toggle, and the docked panel.
 *
 * Client-side because it needs the current path — both to mark the active tab and
 * to give the assistant its page context — and because the panel holds state that
 * must survive navigation between the five pages.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChatPanel } from "@/components/ChatPanel";
import { ChatProvider, useChat } from "@/lib/chat-context";

const NAV = [
  { href: "/", label: "Data Health" },
  { href: "/factors", label: "Factor Explorer" },
  { href: "/raw", label: "Raw Explorer" },
  { href: "/matrix", label: "Covariance & PCA" },
  { href: "/loadings", label: "Loadings Lab" },
  { href: "/risk", label: "Risk Lens" },
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
  const { open, setOpen, snapshot } = useChat();

  return (
    <>
      <header className="sticky top-0 z-30 border-b border-line bg-panel/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center gap-6 px-5 py-2.5">
          <Link href="/" className="shrink-0">
            <span className="font-mono text-[13px] font-bold tracking-tight text-navy">
              FACTOR&nbsp;TERMINAL
            </span>
          </Link>

          <nav className="flex flex-1 gap-1 overflow-x-auto no-scrollbar">
            {NAV.map((n) => {
              const active = n.href === pathname;
              return (
                <Link
                  key={n.href}
                  href={n.href}
                  aria-current={active ? "page" : undefined}
                  className={`whitespace-nowrap rounded px-2.5 py-1 text-[12px] transition
                    ${active
                      ? "bg-navy text-white"
                      : "text-muted hover:bg-lineSoft hover:text-navy"}`}
                >
                  {n.label}
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
        </div>
      </header>

      <main
        className="mx-auto max-w-[1600px] px-5 py-5 transition-[padding] duration-200"
        style={{ paddingRight: open ? 436 : undefined }}
      >
        {children}
      </main>

      <ChatPanel open={open} onClose={() => setOpen(false)} snapshot={snapshot} />
    </>
  );
}
