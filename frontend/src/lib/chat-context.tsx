"use client";

/**
 * Carries the current page's rendered figures to the assistant panel.
 *
 * The panel lives in the layout so it survives navigation, but only a page knows
 * what it has drawn. Rather than have the panel guess from the URL and re-fetch
 * — which would risk answering about numbers different from the ones on screen —
 * each page publishes exactly what it rendered.
 */

import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState,
} from "react";
import type { ChatSnapshot } from "@/components/ChatPanel";

interface ChatContextValue {
  snapshot: ChatSnapshot;
  publish: (s: Omit<ChatSnapshot, "page">) => void;
  open: boolean;
  setOpen: (v: boolean) => void;
}

const Ctx = createContext<ChatContextValue>({
  snapshot: { page: "/" },
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
  const [snapshot, setSnapshot] = useState<ChatSnapshot>({ page });
  const [open, setOpen] = useState(false);

  const publish = useCallback(
    (s: Omit<ChatSnapshot, "page">) => setSnapshot({ page, ...s }),
    [page]
  );

  // A new page starts with nothing published, so a stale snapshot from the
  // previous page can never be attached to a question about this one.
  useEffect(() => setSnapshot({ page }), [page]);

  const value = useMemo(
    () => ({ snapshot, publish, open, setOpen }),
    [snapshot, publish, open]
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
