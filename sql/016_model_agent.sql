-- 016_model_agent.sql — the model agent, and what the risk backtest now records.
--
-- Additive only. The main checkout shares this database and its writers name
-- their columns, so new nullable columns and new tables leave them working.

-- Dimson lags on a named subset of factors. NULL means every factor, which is
-- what every spec stored before this column existed did.
ALTER TABLE dim_model_spec
    ADD COLUMN IF NOT EXISTS dimson_factors JSONB;

-- The covariance of each window's betas, after any Dimson collapse. A forecast
-- built on noisy betas overstates systematic risk by tr(Sigma V); removing that
-- needs V itself, not only the standard errors on its diagonal.
CREATE TABLE IF NOT EXISTS fact_loading_cov (
    spec_id       TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    window_end    DATE NOT NULL,
    factor_ids    TEXT[] NOT NULL,
    cov           DOUBLE PRECISION[] NOT NULL,   -- row-major, len(factor_ids)^2
    PRIMARY KEY (spec_id, instrument_id, window_end)
);

-- What each forecast was compared against, and how wide its tails were taken.
ALTER TABLE fact_risk_forecast
    ADD COLUMN IF NOT EXISTS n_factors           INTEGER,
    ADD COLUMN IF NOT EXISTS sigma_own_ewma_ann  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sigma_capm_ann      DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS t_df                DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS in_selection_sample BOOLEAN;

-- The backtest gains a loss, two naive benchmarks and the test of whether the
-- model beats them. `scope` separates a full-history backtest from the agent's
-- held-out test span, which are different claims about the same forecasts.
ALTER TABLE fact_risk_backtest
    ADD COLUMN IF NOT EXISTS scope          TEXT,
    ADD COLUMN IF NOT EXISTS bias_stat      DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS qlike          DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS qlike_own_ewma DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS qlike_capm     DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_t_own_ewma  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_p_own_ewma  DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_t_capm      DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_p_capm      DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mz_hac_lags    INTEGER,
    ADD COLUMN IF NOT EXISTS var_dist       TEXT;

-- One agent run: one security, one search, one verdict.
CREATE TABLE IF NOT EXISTS agent_run (
    run_id          UUID PRIMARY KEY,
    instrument_id   TEXT NOT NULL,
    horizon_days    INTEGER NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    budget          JSONB,
    profile         JSONB,
    validation_start DATE,
    validation_end   DATE,
    test_start      DATE,
    test_end        DATE,
    chosen_candidate TEXT,
    spec_id         TEXT REFERENCES dim_model_spec(spec_id) ON DELETE SET NULL,
    cov_method      TEXT,
    risk_spec       JSONB,
    verdict         TEXT,
    gates           JSONB,
    test_metrics    JSONB,
    llm_model       TEXT,
    error           TEXT,
    CONSTRAINT agent_run_status_chk
        CHECK (status IN ('running','succeeded','failed'))
);

CREATE INDEX IF NOT EXISTS idx_agent_run_instrument
    ON agent_run (instrument_id, started_at DESC);

-- Every specification the agent tried, where it came from, and how it scored on
-- the validation span. Test-span numbers never appear here: the search never saw
-- them.
CREATE TABLE IF NOT EXISTS agent_candidate (
    run_id        UUID NOT NULL REFERENCES agent_run(run_id) ON DELETE CASCADE,
    candidate_id  TEXT NOT NULL,
    round         INTEGER NOT NULL,
    origin        TEXT NOT NULL,          -- seed | llm | critic
    estimation    JSONB NOT NULL,
    risk          JSONB NOT NULL,
    spec_hash     TEXT NOT NULL,
    metrics       JSONB,
    complexity    DOUBLE PRECISION,
    in_mcs        BOOLEAN,
    mcs_p         DOUBLE PRECISION,
    status        TEXT NOT NULL DEFAULT 'scored',
    note          TEXT,
    PRIMARY KEY (run_id, candidate_id)
);

-- The trace: what each node did, in order. Raw LLM replies are kept here, so a
-- search decision can always be traced to the text that produced it.
CREATE TABLE IF NOT EXISTS agent_step (
    run_id   UUID NOT NULL REFERENCES agent_run(run_id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    node     TEXT NOT NULL,
    message  TEXT NOT NULL,
    payload  JSONB,
    at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);
