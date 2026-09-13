-- 005_estimation.sql — model specs, rolling factor loadings, regression diagnostics
--
-- Every distinct parameter combination hashes to a spec_id, so a repeated UI request
-- is a table read rather than a re-fit.

CREATE TABLE IF NOT EXISTS dim_model_spec (
    spec_id          TEXT PRIMARY KEY,      -- sha256 of the canonicalised spec
    name             TEXT,
    factor_set       JSONB NOT NULL,        -- ordered list of factor_id
    estimator        TEXT NOT NULL DEFAULT 'ols',   -- ols | ridge | huber | elasticnet
    window_days      INTEGER NOT NULL,
    step_days        INTEGER NOT NULL,      -- 1 | 5 (1w) | 21 (1m) | 63 (3m)
    weighting        TEXT NOT NULL DEFAULT 'equal', -- equal | ewma
    ewma_halflife    INTEGER,
    hac_lags         INTEGER,               -- NULL => Newey-West automatic rule
    ridge_lambda     DOUBLE PRECISION,
    l1_ratio         DOUBLE PRECISION,
    dimson_lags      SMALLINT NOT NULL DEFAULT 0,
    orthogonalized   BOOLEAN NOT NULL DEFAULT TRUE,
    winsor_lo        DOUBLE PRECISION,
    winsor_hi        DOUBLE PRECISION,
    min_obs          INTEGER NOT NULL DEFAULT 126,
    base_ccy         TEXT NOT NULL DEFAULT 'USD',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT dim_model_spec_estimator_chk
        CHECK (estimator IN ('ols','ridge','huber','elasticnet')),
    CONSTRAINT dim_model_spec_weighting_chk
        CHECK (weighting IN ('equal','ewma')),
    CONSTRAINT dim_model_spec_step_chk CHECK (step_days > 0),
    CONSTRAINT dim_model_spec_window_chk CHECK (window_days >= 21)
);

-- One row per (spec, instrument, window_end, factor). Long format so the factor
-- set can change between specs without a schema change.
CREATE TABLE IF NOT EXISTS fact_loading (
    spec_id       TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    window_end    DATE NOT NULL,
    factor_id     TEXT NOT NULL,
    beta          DOUBLE PRECISION,
    se            DOUBLE PRECISION,   -- Newey-West HAC
    t_stat        DOUBLE PRECISION,
    p_value       DOUBLE PRECISION,
    vif           DOUBLE PRECISION,
    PRIMARY KEY (spec_id, instrument_id, window_end, factor_id)
);

CREATE INDEX IF NOT EXISTS idx_loading_lookup
    ON fact_loading (spec_id, instrument_id, window_end DESC);
CREATE INDEX IF NOT EXISTS idx_loading_factor
    ON fact_loading (spec_id, factor_id, window_end DESC);

CREATE TABLE IF NOT EXISTS fact_regression_meta (
    spec_id           TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id     TEXT NOT NULL,
    window_end        DATE NOT NULL,
    window_start      DATE NOT NULL,
    n_obs             INTEGER NOT NULL,
    alpha             DOUBLE PRECISION,   -- annualised
    se_alpha          DOUBLE PRECISION,
    t_alpha           DOUBLE PRECISION,
    r2                DOUBLE PRECISION,
    adj_r2            DOUBLE PRECISION,
    f_stat            DOUBLE PRECISION,
    f_p               DOUBLE PRECISION,
    rmse              DOUBLE PRECISION,
    resid_vol_ann     DOUBLE PRECISION,
    durbin_watson     DOUBLE PRECISION,
    condition_number  DOUBLE PRECISION,
    max_vif           DOUBLE PRECISION,
    lb_resid_p        DOUBLE PRECISION,   -- residual autocorrelation
    arch_lm_resid_p   DOUBLE PRECISION,   -- residual conditional heteroskedasticity
    beta_shift_l1     DOUBLE PRECISION,   -- L1 change vs previous window: drift alert
    beta_corr_prev    DOUBLE PRECISION,   -- correlation of loading vector vs previous
    quality_score     SMALLINT,           -- 0-5 flag count, see v_regression_quality
    PRIMARY KEY (spec_id, instrument_id, window_end)
);

CREATE INDEX IF NOT EXISTS idx_reg_meta_lookup
    ON fact_regression_meta (spec_id, instrument_id, window_end DESC);

-- Quality scorecard, generalising sec.v_factor_quality from the source warehouse.
CREATE OR REPLACE VIEW v_regression_quality AS
SELECT
    spec_id, instrument_id, window_end, n_obs, adj_r2, f_p,
    condition_number, durbin_watson, max_vif,
    (adj_r2 > 0.20)::int
  + (f_p < 0.05)::int
  + (n_obs >= 126)::int
  + (condition_number < 30)::int
  + (durbin_watson BETWEEN 1.5 AND 2.5)::int AS score
FROM fact_regression_meta;
