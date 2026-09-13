-- 003_factors.sql — constructed daily factor returns

CREATE TABLE IF NOT EXISTS fact_factor_return (
    factor_id   TEXT NOT NULL REFERENCES ref_factor(factor_id) ON DELETE CASCADE,
    date        DATE NOT NULL,
    ret_log     DOUBLE PRECISION,   -- raw constructed return
    ret_excess  DOUBLE PRECISION,   -- excess over cash (DFF), the modelling series
    ret_orth    DOUBLE PRECISION,   -- after block-hierarchy orthogonalisation
    n_inputs    SMALLINT,
    is_complete BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (factor_id, date)
);

CREATE INDEX IF NOT EXISTS idx_factor_return_date ON fact_factor_return (date DESC);

-- Per-build provenance so a factor value can be traced to the code that made it.
CREATE TABLE IF NOT EXISTS fact_factor_build (
    factor_id     TEXT NOT NULL REFERENCES ref_factor(factor_id) ON DELETE CASCADE,
    built_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id        UUID,
    construction  JSONB NOT NULL,
    version       INTEGER NOT NULL,
    first_date    DATE,
    last_date     DATE,
    n_obs         INTEGER,
    PRIMARY KEY (factor_id, built_at)
);
