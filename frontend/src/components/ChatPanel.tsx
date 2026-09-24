"use client";

/**
 * The docked assistant.
 *
 * Two things distinguish it from a generic chat box. It posts a snapshot of the
 * numbers the current page has rendered, so answers are about what the analyst is
 * actually looking at; and it shows what that snapshot contained, in the context
 * chip above the composer. An assistant that quotes figures should be legible about
 * where they came from — otherwise a wrong answer and a right one look identical.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { PageSnapshot } from "@/lib/chat-context";

export interface ChatSnapshot {
  page: string;
  [key: string]: unknown;
}

interface Msg {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  error?: boolean;
}

export function ChatPanel({
  open,
  onClose,
  snapshot,
  snapshots = [],
}: {
  open: boolean;
  onClose: () => void;
  snapshot: ChatSnapshot;
  /** Every page that has drawn, so a question can be about any of them. */
  snapshots?: PageSnapshot[];
}) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<any>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/chat/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus({ configured: false }));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || busy) return;

      const history = messages
        .filter((m) => !m.error)
        .map(({ role, content }) => ({ role, content }));

      setMessages((m) => [...m, { role: "user", content: question },
                              { role: "assistant", content: "" }]);
      setInput("");
      setBusy(true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: question, history, snapshot,
                                 snapshots, thread_id: threadId }),
        });

        if (!res.ok || !res.body) {
          const detail = await res.json().catch(() => null);
          throw new Error(detail?.detail ?? `${res.status} ${res.statusText}`);
        }

        // Server-sent events, parsed by hand: EventSource cannot POST, and the
        // payload here is one request rather than a long-lived subscription.
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let event = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const chunks = buffer.split("\n\n");
          buffer = chunks.pop() ?? "";

          for (const chunk of chunks) {
            for (const line of chunk.split("\n")) {
              if (line.startsWith("event: ")) event = line.slice(7).trim();
              else if (line.startsWith("data: ")) {
                const data = JSON.parse(line.slice(6));
                if (event === "grounding") {
                  setThreadId(data.thread_id);
                  setMessages((m) => patchLast(m, (last) => ({ ...last, sources: data.sources })));
                } else if (event === "delta") {
                  setMessages((m) => patchLast(m, (last) => ({
                    ...last, content: last.content + data.text,
                  })));
                } else if (event === "error") {
                  setMessages((m) => patchLast(m, () => ({
                    role: "assistant", content: data.message, error: true,
                  })));
                }
              }
            }
          }
        }
      } catch (e: any) {
        setMessages((m) => patchLast(m, () => ({
          role: "assistant",
          content: String(e?.message ?? e),
          error: true,
        })));
      } finally {
        setBusy(false);
      }
    },
    [busy, messages, snapshot, snapshots, threadId]
  );

  if (!open) return null;

  const starters: string[] = status?.starters?.[snapshot.page] ?? [];
  const chip = describeSnapshot(snapshot);

  return (
    <aside
      className="fixed right-0 top-0 z-40 flex h-screen w-full max-w-[420px] flex-col
                 border-l border-line bg-panel shadow-[-8px_0_24px_-12px_rgba(0,0,0,0.18)]"
      aria-label="Model assistant"
    >
      <header className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <div>
          <div className="text-2xs font-semibold uppercase tracking-label text-muted">
            Assistant
          </div>
          <div className="text-[11px] text-muted">
            {status?.configured
              ? `${status.model} · methodology ${Math.round((status.corpus_tokens_approx ?? 0) / 1000)}k tokens`
              : "no API key configured"}
          </div>
        </div>
        <button className="btn-ghost" onClick={onClose} aria-label="Close assistant">
          Close
        </button>
      </header>

      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-3">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-[12px] leading-snug text-muted">
              Ask about the methodology or about the numbers on this page. The
              assistant can only see what is rendered here — it cannot query other
              factors or run an estimation.
            </p>
            {starters.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                className="block w-full rounded border border-line bg-white px-3 py-2
                           text-left text-[12px] leading-snug text-navy transition
                           hover:border-navy2 hover:bg-lineSoft"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <Bubble key={i} msg={m} streaming={busy && i === messages.length - 1} />
        ))}
      </div>

      <footer className="border-t border-line px-4 py-2.5">
        <div className="mb-1.5 flex items-center gap-1.5 text-[10px] text-muted">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-pass" />
          <span className="truncate" title={chip}>sees: {chip}</span>
        </div>
        <div className="flex gap-2">
          <textarea
            rows={2}
            className="field resize-none font-sans"
            placeholder={status?.configured ? "Ask about this page…" : "No API key configured"}
            value={input}
            disabled={!status?.configured}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
          />
          <button
            className="btn self-end"
            onClick={() => send(input)}
            disabled={busy || !input.trim() || !status?.configured}
          >
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </footer>
    </aside>
  );
}

function Bubble({ msg, streaming }: { msg: Msg; streaming: boolean }) {
  if (msg.role === "user") {
    return (
      <div className="ml-6 rounded border border-navy/20 bg-navy/5 px-3 py-2
                      text-[12px] leading-snug text-navy whitespace-pre-wrap">
        {msg.content}
      </div>
    );
  }
  return (
    <div className={`rounded border px-3 py-2 text-[12px] leading-relaxed
                    ${msg.error ? "border-fail/30 bg-fail/5 text-fail"
                                : "border-line bg-white"}`}>
      {msg.content ? (
        <div className="chat-md">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>
      ) : streaming ? (
        <span className="inline-flex gap-1 py-1">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-navy3"
              style={{ animationDelay: `${i * 160}ms` }}
            />
          ))}
        </span>
      ) : null}

      {msg.sources && msg.sources.length > 0 && !msg.error && (
        <div className="mt-2 border-t border-lineSoft pt-1.5 text-[10px] text-muted">
          grounded on: {msg.sources.join(" · ")}
        </div>
      )}
    </div>
  );
}

function patchLast(messages: Msg[], fn: (last: Msg) => Msg): Msg[] {
  if (messages.length === 0) return messages;
  const out = messages.slice();
  out[out.length - 1] = fn(out[out.length - 1]);
  return out;
}

/** A one-line summary of what was sent, for the context chip. */
function describeSnapshot(s: ChatSnapshot): string {
  const bits: string[] = [s.page];
  for (const key of ["factor_id", "instrument_id", "spec_id"]) {
    const v = s[key];
    if (typeof v === "string" && v) bits.push(key === "spec_id" ? v.slice(0, 8) : v);
  }
  const stats = s.stats as Record<string, unknown> | undefined;
  if (stats && typeof stats.n_obs === "number") {
    bits.push(`${stats.n_obs.toLocaleString()} obs`);
  }
  return bits.join(" · ");
}
