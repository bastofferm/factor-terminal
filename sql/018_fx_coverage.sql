-- 018_fx_coverage.sql — the liveness gate for the warehouse FX table.
--
-- Additive only: the main checkout shares this database.
--
-- ref_instrument.is_live gates every return series, but fact_input_fx is keyed by
-- currency, not instrument, and nothing watched it. It stopped on 2026-07-09
-- (JPY on 06-25) while returns ran on to October, and the factor builder filled
-- the last rate forward without limit, so rt_uk carried no GBP move for three
-- months. sync.refresh_coverage now records each currency's coverage here against
-- the same as-of date and staleness threshold as the instruments.
CREATE TABLE IF NOT EXISTS ref_fx_coverage (
    ccy        TEXT PRIMARY KEY,
    first_obs  DATE,
    last_obs   DATE,
    n_obs      INTEGER,
    is_live    BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
