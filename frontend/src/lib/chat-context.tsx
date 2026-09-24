"use client";

/**
 * Carries what every page has rendered to the assistant panel.
 *
 * The panel lives in the layout so it survives navigation, but only a page knows
 * what it has drawn. Rather than have the panel guess from the URL and re-fetch
 * — which would risk answering about numbers different from the ones on screen —
 * each page publishes exactly what it rendered.
 *
 * Snapshots accumulate per page instead of being replaced, so a question asked
 * on the covariance screen can still be about a loading seen two tabs ago. The
 * isolation that used to come from wiping on navigation now comes from labelling
 * instead: every snapshot carries its page and the moment it was captured, the
 * active page is marked as such, and the prompt draws the distinction. Dropping
 * the labels rather than the data would be the mistake — a figure from another
 * tab quoted as if it were on screen is worse than not having it.
 */

import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import type { ChatSnapshot } from "@/components/ChatPanel";

/** One page's rendered figures, with when they were rendered. */
export interface PageSnapshot {
  page: string;
  captured_at: string;
  data: Record<string, unknown>;
}

interface ChatContextValue {
  /** The active page's figures, for the panel's own chip and starters. */
  snapshot: ChatSnapshot;
  /** Every page that has published, the active one included. */
  snapshots: PageSnapshot[];
  publish: (s: Omit<ChatSnapshot, "page">) => void;
  open: boolean;
  setOpen: (v: boolean) => void;
}

const Ctx = createContext<ChatContextValue>({
  snapshot: { page: "/" },
  snapshots: [],
  publish: () => {},
  open: false,
  setOpen: () => {},
});

export function ChatProvider({
  page,
  children,
}: {
  page: string;
  children: React.ReactNode;
}) {
  const [byPage, setByPage] = useState<Record<string, PageSnapshot>>({});
  const [open, setOpen] = useState(false);

  const publish = useCallback(
    (s: Omit<ChatSnapshot, "page">) =>
      setByPage((prev) => ({
        ...prev,
        [page]: { page, captured_at: new Date().toISOString(), data: s },
      })),
    [page]
  );

  // The active page's own figures, or nothing if it has not drawn yet. This is
  // deliberately not a fallback to some other page: the chip above the input
  // says what the question will be answered from, and it has to be honest when
  // the answer is "this screen, nothing yet".
  const snapshot: ChatSnapshot = useMemo(
    () => ({ page, ...(byPage[page]?.data ?? {}) }),
    [byPage, page]
  );

  const snapshots = useMemo(
    () =>
      Object.values(byPage).sort((a, b) =>
        a.page === page ? -1 : b.page === page ? 1 : a.page.localeCompare(b.page)
      ),
    [byPage, page]
  );

  const value = useMemo(
    () => ({ snapshot, snapshots, publish, open, setOpen }),
    [snapshot, snapshots, publish, open]
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useChat() {
  return useContext(Ctx);
}

/**
 * Publish a snapshot from a page.
 *
 * The payload is compared by serialised value rather than by reference, because
 * pages rebuild these objects on every render and a reference check would republish
 * — and so re-render the panel — continuously.
 */
export function usePublishSnapshot(payload: Omit<ChatSnapshot, "page"> | null) {
  const { publish } = useChat();
  const serialised = payload ? JSON.stringify(payload) : "";
  const last = useRef("");

  useEffect(() => {
    if (!serialised || serialised === last.current) return;
    last.current = serialised;
    publish(JSON.parse(serialised));
  }, [serialised, publish]);
}
