"""The chat endpoint.

Streams over server-sent events. Neither source repository streams — both run the
whole call and then hand back a finished string — but there is no tool loop here, so
streaming is only a matter of forwarding deltas, and a five-second wait staring at
"Thinking…" is a poor experience for a question that is often one sentence long.

The message order is deliberate and load-bearing: system prompt, then the stable
methodology corpus, then the volatile canonical block, then the conversation.
DeepSeek prices a context-cache hit at roughly a fiftieth of a miss, and measured on
this corpus 95% of the prompt comes back cached, so anything that changes between
turns has to sit behind anything that does not.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app import db
from backend.app.ai import context, corpus, prompts, providers

router = APIRouter()

# The same cap the MZQA panel uses. Long enough to hold a line of questioning,
# short enough that the volatile part of the prompt stays small.
HISTORY_LIMIT = 20
MAX_QUESTION_CHARS = 4000


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)
    # Whatever the page has rendered. Shape is deliberately loose — each page knows
    # what it is showing, and pinning a schema here would mean editing this file
    # every time a panel gains a number.
    snapshot: dict | None = None
    # Every page that has rendered this session, the active one included, each
    # carrying its own page and capture time. Sent so a question asked on one
    # screen can be about a figure seen on another; the labels are what keep
    # those apart.
    snapshots: list[dict] | None = None
    thread_id: str | None = None


@router.get("/status")
async def status() -> dict:
    info = corpus.info()
    return {
        "configured": providers.has_api_key(),
        "model": providers.DEFAULT_MODEL,
        "corpus_chars": info["chars"],
        "corpus_tokens_approx": info["approx_tokens"],
        "starters": prompts.STARTERS,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _persist_thread(thread_id: uuid.UUID, page: str | None) -> None:
    await db.execute(
        """
        INSERT INTO chat_thread (thread_id, page) VALUES ($1, $2)
        ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()
        """,
        thread_id, page,
    )


async def _persist_message(thread_id: uuid.UUID, role: str, content: str,
                           grounding: dict | None = None,
                           model: str | None = None,
                           usage: providers.Usage | None = None) -> None:
    await db.execute(
        """
        INSERT INTO chat_message
            (thread_id, seq, role, content, grounding, model,
             prompt_tokens, cached_tokens, completion_tokens)
        SELECT $1, COALESCE(max(seq), 0) + 1, $2, $3, $4::jsonb, $5, $6, $7, $8
        FROM chat_message WHERE thread_id = $1
        """,
        thread_id, role, content,
        json.dumps(grounding) if grounding is not None else None,
        model,
        usage.prompt_tokens if usage else None,
        usage.cached_tokens if usage else None,
        usage.completion_tokens if usage else None,
    )


@router.post("")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Answer one turn, streaming the text as it arrives."""
    question = (req.message or "").strip()
    if not question:
        raise HTTPException(400, "message is empty")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"message exceeds {MAX_QUESTION_CHARS} characters")
    if not providers.has_api_key():
        raise HTTPException(
            503, "No DeepSeek API key is configured on the server "
                 "(set DEEPSEEK_API_KEY).")

    canonical, sources = await context.build(question, req.snapshot, req.snapshots)

    messages: list[dict] = [
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "system", "content": corpus.build()},
    ]
    if canonical:
        messages.append({
            "role": "system",
            "content": prompts.canonical_header() + "\n" + canonical,
        })
    for m in req.history[-HISTORY_LIMIT:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": question})

    thread_id = uuid.UUID(req.thread_id) if req.thread_id else uuid.uuid4()
    page = (req.snapshot or {}).get("page")

    async def event_stream() -> AsyncIterator[str]:
        # Tell the client what the answer is grounded on before any text arrives,
        # so the context chip is populated while the model is still thinking.
        yield _sse("grounding", {"sources": sources, "thread_id": str(thread_id)})

        collected: list[str] = []
        finish: str | None = None
        try:
            async for delta, reason in providers.stream(messages):
                if delta:
                    collected.append(delta)
                    yield _sse("delta", {"text": delta})
                if reason:
                    finish = reason
        except providers.LLMError as exc:
            yield _sse("error", {"message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - the stream must always close cleanly
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
            return

        text = providers.sanitize("".join(collected)).strip()

        if not text:
            # The documented reasoning-tier failure: the whole budget went on hidden
            # reasoning. Say so rather than showing an empty bubble.
            detail = (
                "The model returned no text and stopped because it ran out of "
                "output budget — the reasoning tiers can spend the entire budget "
                "before writing anything. Try a shorter question, or set "
                "DEEPSEEK_MODEL to a non-reasoning model."
                if finish == "length" else
                "The model returned no text."
            )
            yield _sse("error", {"message": detail})
            return

        try:
            await _persist_thread(thread_id, page)
            await _persist_message(thread_id, "user", question)
            await _persist_message(
                thread_id, "assistant", text,
                grounding={"sources": sources, "snapshot": req.snapshot},
                model=providers.DEFAULT_MODEL)
        except Exception:
            # Losing the transcript must not lose the answer the analyst is reading.
            pass

        yield _sse("done", {"finish_reason": finish, "thread_id": str(thread_id)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # stops a reverse proxy buffering the stream
            "Connection": "keep-alive",
        },
    )


@router.get("/threads")
async def threads(limit: int = 20) -> list[dict]:
    return await db.fetch(
        """
        SELECT t.thread_id, t.page, t.created_at, t.updated_at,
               count(m.seq) AS n_messages,
               min(m.content) FILTER (WHERE m.role = 'user') AS first_question
        FROM chat_thread t
        LEFT JOIN chat_message m USING (thread_id)
        GROUP BY t.thread_id
        ORDER BY t.updated_at DESC LIMIT $1
        """,
        limit,
    )


@router.get("/threads/{thread_id}")
async def thread(thread_id: str) -> dict:
    rows = await db.fetch(
        "SELECT seq, role, content, grounding, model, created_at "
        "FROM chat_message WHERE thread_id = $1 ORDER BY seq",
        uuid.UUID(thread_id),
    )
    if not rows:
        raise HTTPException(404, "no such thread")
    return {"thread_id": thread_id, "messages": rows}
