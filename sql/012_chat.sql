-- 012_chat.sql — chat threads and messages.
--
-- Every assistant answer is stored with the canonical block it was grounded on.
-- That is the point of persisting at all: a claim the model made last month can be
-- checked against the numbers it actually saw, rather than against today's. It
-- matches how the rest of this project treats provenance — fact_factor_build keeps
-- the construction that produced a factor value, etl_item_state keeps what each
-- ingest actually fetched.

CREATE TABLE IF NOT EXISTS chat_thread (
    thread_id   UUID PRIMARY KEY,
    title       TEXT,
    page        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_message (
    thread_id   UUID NOT NULL REFERENCES chat_thread(thread_id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    -- Assistant turns only: what the answer was allowed to see, and where it came
    -- from. Null on user turns.
    grounding   JSONB,
    model       TEXT,
    prompt_tokens     INTEGER,
    cached_tokens     INTEGER,
    completion_tokens INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, seq),
    CONSTRAINT chat_message_role_chk CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX IF NOT EXISTS idx_chat_message_thread
    ON chat_message (thread_id, seq);
CREATE INDEX IF NOT EXISTS idx_chat_thread_recent
    ON chat_thread (updated_at DESC);
