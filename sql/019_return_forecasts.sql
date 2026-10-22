-- 019_return_forecasts.sql — inputs and audit trail for the Return Forecasts page.
--
-- Additive only: the main checkout shares this database.
--
-- The risk model needs no fundamentals; a ten-year building-block return forecast
-- needs valuation history long enough to say what "normal" is. None of it is in
-- the warehouse (its company-level market caps are wrong historically and start
-- in FY2010), so three external files are loaded by hand into the tables below:
--
--   * Shiller's monthly US data from 1871 (ingest_shiller)          -> fact_shiller_us
--   * Barclays' monthly CAPE by country from 1981 (ingest_barclays_cape) -> fact_cape_country
--   * the Jorda-Schularick-Taylor Macrohistory panel, annual from 1870
--     (ingest_macrohistory)                                          -> fact_macrohistory
--
-- Two of the three carry licence conditions, so the licence of every external
-- source is a row here and the page prints it next to the numbers.

CREATE TABLE IF NOT EXISTS fact_shiller_us (
    month          DATE PRIMARY KEY,            -- month-end
    price          DOUBLE PRECISION,            -- S&P composite, monthly average of closes
    dividend       DOUBLE PRECISION,            -- trailing 12-month dividends per share
    earnings       DOUBLE PRECISION,            -- trailing 12-month earnings per share
    cpi            DOUBLE PRECISION,
    gs10           DOUBLE PRECISION,            -- long rate, percent
    real_price     DOUBLE PRECISION,
    real_dividend  DOUBLE PRECISION,
    real_tr_price  DOUBLE PRECISION,            -- real total-return index
    real_earnings  DOUBLE PRECISION,
    cape           DOUBLE PRECISION,
    tr_cape        DOUBLE PRECISION,
    bond_tr_gross  DOUBLE PRECISION,            -- one month's nominal gross return of the 10-year bond
    real_bond_tr   DOUBLE PRECISION,            -- real total-return index of the 10-year bond
    source_sha     TEXT,
    loaded_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Long format so a second source for the same country (Shiller's US CAPE beside
-- Barclays') sits next to the first rather than overwriting it.
CREATE TABLE IF NOT EXISTS fact_cape_country (
    country     TEXT NOT NULL,                  -- the source's own label: Europe, Japan, USA ...
    month       DATE NOT NULL,                  -- month-end
    cape        DOUBLE PRECISION NOT NULL,
    source      TEXT NOT NULL,                  -- barclays | shiller
    source_sha  TEXT,
    loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (country, month, source)
);

CREATE TABLE IF NOT EXISTS fact_macrohistory (
    iso       TEXT NOT NULL,                    -- JST's ISO-3 code
    country   TEXT,
    year      INTEGER NOT NULL,
    variable  TEXT NOT NULL,                    -- JST's own variable name: eq_tr, eq_dp, cpi ...
    value     DOUBLE PRECISION NOT NULL,
    release   TEXT NOT NULL,                    -- R6
    PRIMARY KEY (iso, year, variable, release)
);

-- Published capital market assumptions typed in by hand (JPM LTCMA, Invesco CMA,
-- an in-house view). The "conviction" group's second member next to the building
-- blocks; never fetched, so never redistributed.
CREATE TABLE IF NOT EXISTS ref_cma_external (
    source           TEXT NOT NULL,
    as_of            DATE NOT NULL,
    asset_class      TEXT NOT NULL,             -- return_forecast.CLASSES key
    expected_return  DOUBLE PRECISION NOT NULL, -- compound annual, decimal
    note             TEXT,
    entered_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, as_of, asset_class)
);

-- Every forecast run with the assumptions it used: a building-block forecast is
-- only as good as the record of what was assumed.
CREATE TABLE IF NOT EXISTS return_forecast_run (
    run_id       TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    as_of        DATE,
    assumptions  JSONB NOT NULL,
    result       JSONB
);

CREATE INDEX IF NOT EXISTS idx_return_forecast_run_created ON return_forecast_run (created_at DESC);

CREATE TABLE IF NOT EXISTS ref_data_licence (
    source       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    licence      TEXT NOT NULL,
    attribution  TEXT NOT NULL,
    url          TEXT,
    accessed_on  DATE,
    restriction  TEXT
);

INSERT INTO ref_data_licence (source, name, licence, attribution, url, accessed_on, restriction)
VALUES
    ('shiller', 'Robert J. Shiller, U.S. stock market data (ie_data.xls)',
     'Published without a licence; the disclaimer puts the risk of any use on the user.',
     'Robert J. Shiller, Irrational Exuberance data, shillerdata.com',
     'https://shillerdata.com/', DATE '2026-10-22', NULL),
    ('barclays', 'Barclays Indices, Historic CAPE Ratio by country (cape.csv)',
     'Barclays intellectual property. No distribution and no use in any product without Barclays'' written consent.',
     'Source: Barclays Indices, Historic CAPE Ratio by country',
     'https://indices.cib.barclays/file.app?action=shared&path=shiller/cape.csv', DATE '2026-10-22',
     'Internal, private use only; not passed to the assistant or redistributed.'),
    ('jst', 'Jorda-Schularick-Taylor Macrohistory Database, release 6',
     'CC BY-NC-SA 4.0: non-commercial use, attribution, share alike.',
     'Jorda, Knoll, Kuvshinov, Schularick and Taylor (2019), The Rate of Return on Everything, 1870-2015, Quarterly Journal of Economics 134(3)',
     'https://www.macrohistory.net/database/', DATE '2026-10-22',
     'Non-commercial use only; not passed to the assistant or redistributed.')
ON CONFLICT (source) DO UPDATE SET
    name = EXCLUDED.name, licence = EXCLUDED.licence, attribution = EXCLUDED.attribution,
    url = EXCLUDED.url, accessed_on = EXCLUDED.accessed_on, restriction = EXCLUDED.restriction;
