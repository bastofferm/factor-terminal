-- 008_ops.sql — run state and staging (simplified from sec.pipeline_stage_run +
-- sec.market_source_item_state, the strongest pattern in the source warehouse)

CREATE TABLE IF NOT EXISTS etl_run (
    run_id      UUID PRIMARY KEY,
    job         TEXT NOT NULL,
    mode        TEXT NOT NULL DEFAULT 'incremental',   -- incremental | full
    status      TEXT NOT NULL DEFAULT 'running',       -- running | succeeded | failed | partial
    scope       JSONB,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    rows_in     BIGINT NOT NULL DEFAULT 0,
    rows_out    BIGINT NOT NULL DEFAULT 0,
    n_failed    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    CONSTRAINT etl_run_status_chk CHECK (status IN ('running','succeeded','failed','partial'))
);

CREATE INDEX IF NOT EXISTS idx_etl_run_job ON etl_run (job, started_at DESC);

CREATE TABLE IF NOT EXISTS etl_item_state (
    job         TEXT NOT NULL,
    item_key    TEXT NOT NULL,
    run_id      UUID REFERENCES etl_run(run_id) ON DELETE SET NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    min_date    DATE,
    max_date    DATE,
    rows_in     INTEGER,
    rows_out    INTEGER,
    error       TEXT,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job, item_key),
    CONSTRAINT etl_item_state_status_chk
        CHECK (status IN ('pending','running','succeeded','failed','skipped'))
);

CREATE INDEX IF NOT EXISTS idx_etl_item_state_run ON etl_item_state (run_id, status);
CREATE INDEX IF NOT EXISTS idx_etl_item_finished ON etl_item_state (job, finished_at DESC);

-- Run-scoped staging. Rows are promoted to fact_* in one server-side statement,
-- then deleted with the run.
CREATE TABLE IF NOT EXISTS stage_return (
    run_id        UUID NOT NULL REFERENCES etl_run(run_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    date          DATE NOT NULL,
    close         DOUBLE PRECISION,
    adj_close     DOUBLE PRECISION,
    ret_simple    DOUBLE PRECISION,
    ret_log       DOUBLE PRECISION,
    volume        BIGINT,
    currency      TEXT,
    source        TEXT,
    PRIMARY KEY (run_id, instrument_id, date)
);

CREATE TABLE IF NOT EXISTS stage_level (
    run_id    UUID NOT NULL REFERENCES etl_run(run_id) ON DELETE CASCADE,
    series_id TEXT NOT NULL,
    date      DATE NOT NULL,
    value     DOUBLE PRECISION,
    source    TEXT,
    PRIMARY KEY (run_id, series_id, date)
);

-- Data-health view backing the UI banner.
CREATE OR REPLACE VIEW v_data_health AS
SELECT
    i.instrument_id,
    i.source_ticker,
    i.asset_class,
    i.role,
    i.is_live,
    i.is_total_return,
    i.last_obs,
    (CURRENT_DATE - i.last_obs) AS days_stale,
    i.n_obs
FROM ref_instrument i;
