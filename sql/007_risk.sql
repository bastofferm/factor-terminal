-- 007_risk.sql — ex-ante risk forecasts and out-of-sample validation (plan section 8)

-- Predicted risk uses betas and covariance known at as_of_date only; realised risk
-- is measured over the forward horizon. Keeping both on one row makes the lag
-- discipline auditable rather than implicit.
CREATE TABLE IF NOT EXISTS fact_risk_forecast (
    spec_id            TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id      TEXT NOT NULL,
    as_of_date         DATE NOT NULL,
    horizon_days       INTEGER NOT NULL,
    cov_method         TEXT NOT NULL,
    sigma_pred_ann     DOUBLE PRECISION NOT NULL,
    sigma_factor_ann   DOUBLE PRECISION,
    sigma_specific_ann DOUBLE PRECISION,
    factor_risk_share  DOUBLE PRECISION,
    sigma_realized_ann DOUBLE PRECISION,   -- NULL until the horizon has elapsed
    bias_ratio         DOUBLE PRECISION,   -- realised / predicted; E[.] = 1 if correct
    var95_pred         DOUBLE PRECISION,
    var99_pred         DOUBLE PRECISION,
    es97_5_pred        DOUBLE PRECISION,
    n_exceptions_95    INTEGER,
    n_exceptions_99    INTEGER,
    betas_window_end   DATE NOT NULL,      -- provenance: which loading window was used
    PRIMARY KEY (spec_id, instrument_id, as_of_date, horizon_days, cov_method)
);

CREATE INDEX IF NOT EXISTS idx_risk_forecast_lookup
    ON fact_risk_forecast (spec_id, instrument_id, as_of_date DESC);

-- Risk decomposition by factor block over time.
CREATE TABLE IF NOT EXISTS fact_risk_contribution (
    spec_id       TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id TEXT NOT NULL,
    as_of_date    DATE NOT NULL,
    block_id      TEXT NOT NULL,
    risk_contrib  DOUBLE PRECISION NOT NULL,   -- annualised vol units, sums to sigma_pred
    share         DOUBLE PRECISION,
    PRIMARY KEY (spec_id, instrument_id, as_of_date, block_id)
);

-- Backtest summary over a sample. This is the answer to "is the predicted risk right".
CREATE TABLE IF NOT EXISTS fact_risk_backtest (
    spec_id               TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id         TEXT NOT NULL,
    sample_start          DATE NOT NULL,
    sample_end            DATE NOT NULL,
    horizon_days          INTEGER NOT NULL,
    cov_method            TEXT NOT NULL,
    n_forecasts           INTEGER NOT NULL,

    mean_bias             DOUBLE PRECISION,   -- mean(realised / predicted)
    median_bias           DOUBLE PRECISION,
    z_std                 DOUBLE PRECISION,   -- sd of r_t / sigma_hat_{t-1}; target 1
    z_kurtosis            DOUBLE PRECISION,

    -- Mincer-Zarnowitz: sigma_real^2 = a + b * sigma_pred^2, testing a=0, b=1.
    mz_alpha              DOUBLE PRECISION,
    mz_beta               DOUBLE PRECISION,
    mz_alpha_p            DOUBLE PRECISION,
    mz_beta_p             DOUBLE PRECISION,   -- H0: b = 1
    mz_joint_p            DOUBLE PRECISION,   -- H0: a = 0 and b = 1
    mz_r2                 DOUBLE PRECISION,

    -- VaR coverage.
    exceptions_95         INTEGER,
    exceptions_99         INTEGER,
    expected_95           DOUBLE PRECISION,
    expected_99           DOUBLE PRECISION,
    kupiec_stat_95        DOUBLE PRECISION,
    kupiec_p_95           DOUBLE PRECISION,
    kupiec_stat_99        DOUBLE PRECISION,
    kupiec_p_99           DOUBLE PRECISION,
    christoffersen_stat_95 DOUBLE PRECISION,
    christoffersen_p_95   DOUBLE PRECISION,
    christoffersen_stat_99 DOUBLE PRECISION,
    christoffersen_p_99   DOUBLE PRECISION,

    computed_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (spec_id, instrument_id, sample_start, sample_end, horizon_days, cov_method)
);
