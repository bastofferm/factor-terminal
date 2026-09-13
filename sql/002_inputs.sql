-- 002_inputs.sql — mirrored raw inputs, synced from the xbrl_sec warehouse over postgres_fdw
-- plus rows fetched directly by this project's own ingestion jobs.

-- Return series for every instrument. ret_log is the modelling series;
-- ret_simple is kept for compounding and reporting.
CREATE TABLE IF NOT EXISTS fact_input_return (
    instrument_id TEXT NOT NULL REFERENCES ref_instrument(instrument_id) ON DELETE CASCADE,
    date          DATE NOT NULL,
    close         DOUBLE PRECISION,
    adj_close     DOUBLE PRECISION,
    ret_simple    DOUBLE PRECISION,
    ret_log       DOUBLE PRECISION,
    volume        BIGINT,
    currency      TEXT NOT NULL DEFAULT 'USD',
    source        TEXT NOT NULL DEFAULT 'warehouse',
    PRIMARY KEY (instrument_id, date)
);

CREATE INDEX IF NOT EXISTS idx_input_return_date ON fact_input_return (date DESC);

-- Level series: yields, spreads, index levels. NOT directly usable as factors —
-- transforms.py differences them first (see plan section 3).
CREATE TABLE IF NOT EXISTS fact_input_level (
    series_id  TEXT NOT NULL,
    date       DATE NOT NULL,
    value      DOUBLE PRECISION,
    source     TEXT NOT NULL DEFAULT 'warehouse',
    PRIMARY KEY (series_id, date)
);

CREATE INDEX IF NOT EXISTS idx_input_level_date ON fact_input_level (date DESC);

-- Metadata for level series so the transform rule is explicit and auditable.
CREATE TABLE IF NOT EXISTS ref_level_series (
    series_id       TEXT PRIMARY KEY,
    name            TEXT,
    category        TEXT,           -- rates | credit | fx | volatility | liquidity | inflation | stress
    unit            TEXT,           -- percent | index | bp
    transform       TEXT NOT NULL DEFAULT 'diff',  -- diff | log_diff | diff_std | level
    tenor_years     DOUBLE PRECISION,              -- for yields: used for duration scaling
    curve_id        TEXT,                          -- US_TSY | EA_AAA | JP_JGB
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    first_obs       DATE,
    last_obs        DATE,
    CONSTRAINT ref_level_series_transform_chk
        CHECK (transform IN ('diff','log_diff','diff_std','level'))
);

CREATE INDEX IF NOT EXISTS idx_ref_level_series_curve ON ref_level_series (curve_id, tenor_years);

-- FX, for base-currency conversion via the log-additivity identity r_usd = r_local + r_fx.
CREATE TABLE IF NOT EXISTS fact_input_fx (
    ccy          TEXT NOT NULL,
    date         DATE NOT NULL,
    usd_per_unit DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (ccy, date)
);

CREATE INDEX IF NOT EXISTS idx_input_fx_date ON fact_input_fx (date DESC);

-- Published reference factors (Fama-French, AQR). Validation only, never a live
-- model input: Ken French publishes with a one-to-two month lag.
CREATE TABLE IF NOT EXISTS fact_reference_factor (
    dataset    TEXT NOT NULL,
    factor     TEXT NOT NULL,
    date       DATE NOT NULL,
    ret_pct    DOUBLE PRECISION,
    ret_log    DOUBLE PRECISION,
    PRIMARY KEY (dataset, factor, date)
);

CREATE INDEX IF NOT EXISTS idx_reference_factor_date ON fact_reference_factor (date DESC);
