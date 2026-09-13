-- 006_covariance.sql — factor covariance, specific risk (plan section 7 / PDF 7.1)

-- Long format, upper triangle stored twice for query simplicity (factor counts are
-- 25-45, so the space cost is negligible and the join logic stays trivial).
CREATE TABLE IF NOT EXISTS fact_factor_cov (
    spec_id    TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,
    method     TEXT NOT NULL,   -- sample | ewma | ledoit_wolf | blend
    factor_i   TEXT NOT NULL,
    factor_j   TEXT NOT NULL,
    cov_value  DOUBLE PRECISION NOT NULL,   -- annualised
    corr_value DOUBLE PRECISION,
    PRIMARY KEY (spec_id, as_of_date, method, factor_i, factor_j)
);

CREATE INDEX IF NOT EXISTS idx_factor_cov_lookup
    ON fact_factor_cov (spec_id, method, as_of_date DESC);

CREATE TABLE IF NOT EXISTS fact_factor_cov_meta (
    spec_id           TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    as_of_date        DATE NOT NULL,
    method            TEXT NOT NULL,
    n_factors         INTEGER NOT NULL,
    n_obs             INTEGER NOT NULL,
    halflife          INTEGER,
    shrink_intensity  DOUBLE PRECISION,   -- Ledoit-Wolf delta, must lie in [0,1]
    blend_weight      DOUBLE PRECISION,   -- weight on EWMA in the blend
    min_eigenvalue    DOUBLE PRECISION,
    max_eigenvalue    DOUBLE PRECISION,
    condition_number  DOUBLE PRECISION,
    is_psd            BOOLEAN NOT NULL,
    psd_repaired      BOOLEAN NOT NULL DEFAULT FALSE,  -- Higham projection bound
    pc1_share         DOUBLE PRECISION,   -- variance share of the first eigenvector
    pc3_share         DOUBLE PRECISION,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (spec_id, as_of_date, method)
);

-- Specific (idiosyncratic) risk per instrument.
-- sigma_eps^2 = max(floor, Shrink[EWMA(eps^2), peer_median])  -- PDF 6.4
CREATE TABLE IF NOT EXISTS fact_specific_risk (
    spec_id        TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    instrument_id  TEXT NOT NULL,
    as_of_date     DATE NOT NULL,
    sigma_eps_ann  DOUBLE PRECISION NOT NULL,
    sigma_raw_ann  DOUBLE PRECISION,   -- EWMA before floor and shrinkage
    sigma_peer_ann DOUBLE PRECISION,
    shrink_w       DOUBLE PRECISION,
    floor_applied  BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (spec_id, instrument_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_specific_risk_lookup
    ON fact_specific_risk (spec_id, instrument_id, as_of_date DESC);

-- Residual PCA (PDF 5.2): common structure the economic factors missed.
-- A high pc1_share is a model-completeness warning, not a result.
CREATE TABLE IF NOT EXISTS fact_residual_pca (
    spec_id      TEXT NOT NULL REFERENCES dim_model_spec(spec_id) ON DELETE CASCADE,
    as_of_date   DATE NOT NULL,
    component    SMALLINT NOT NULL,
    eigenvalue   DOUBLE PRECISION,
    var_share    DOUBLE PRECISION,
    cum_share    DOUBLE PRECISION,
    top_loadings JSONB,
    PRIMARY KEY (spec_id, as_of_date, component)
);
