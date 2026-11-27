-- 020_relative_value.sql — runs and summaries of the Relative Value & Timing page.
--
-- Additive only: the main checkout shares this database.
--
-- A run scores every relative-value view (value, carry, trend, macro) against the
-- latest ten-year forecast and is stored with the assumptions it used, like a
-- return_forecast_run. A summary is DeepSeek's reading of one run. It is stored
-- only after every number in it was found in the data it was given, so a stored
-- summary never quotes a figure the pipeline did not compute.

CREATE TABLE IF NOT EXISTS relative_value_run (
    run_id           TEXT PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    as_of            DATE,
    data_as_of       DATE,                      -- last price date the signals saw
    forecast_run_id  TEXT,                      -- the ten-year anchor (return_forecast_run)
    assumptions      JSONB NOT NULL,
    result           JSONB
);

CREATE INDEX IF NOT EXISTS idx_relative_value_run_created ON relative_value_run (created_at DESC);

CREATE TABLE IF NOT EXISTS relative_value_summary (
    summary_id         BIGSERIAL PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES relative_value_run (run_id) ON DELETE CASCADE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    model              TEXT NOT NULL,
    payload_sha        TEXT NOT NULL,           -- the data the model was shown, hashed
    content            JSONB NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 1,
    prompt_tokens      INTEGER,
    cached_tokens      INTEGER,
    completion_tokens  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_relative_value_summary_run
    ON relative_value_summary (run_id, created_at DESC);
