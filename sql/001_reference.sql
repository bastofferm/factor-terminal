-- 001_reference.sql — factor registry, instrument master, trading calendar
-- Idempotent. Applied by backend/pipeline/apply_schema.py.

CREATE TABLE IF NOT EXISTS ref_factor_block (
    block_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    description TEXT
);

COMMENT ON TABLE ref_factor_block IS
    'The nine factor blocks of the model, plus any derived blocks.';

-- Instrument master. One row per tradable/observable series that can feed a factor
-- or be analysed as a security. `is_total_return` is load-bearing: price-return
-- index levels (Yahoo ^-tickers) must never be used as equity market factors.
CREATE TABLE IF NOT EXISTS ref_instrument (
    instrument_id   TEXT PRIMARY KEY,
    source_ticker   TEXT NOT NULL,
    source_table    TEXT NOT NULL,          -- fact_cross_asset | fact_prices_us | fact_prices_jp | yahoo
    asset_class     TEXT,
    currency        TEXT NOT NULL DEFAULT 'USD',
    is_total_return BOOLEAN NOT NULL DEFAULT TRUE,
    role            TEXT NOT NULL DEFAULT 'factor_input',  -- factor_input | analysis | both
    is_live         BOOLEAN NOT NULL DEFAULT TRUE,
    first_obs       DATE,
    last_obs        DATE,
    n_obs           INTEGER,
    notes           TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ref_instrument_role_chk CHECK (role IN ('factor_input','analysis','both'))
);

CREATE INDEX IF NOT EXISTS idx_ref_instrument_role  ON ref_instrument (role, is_live);
CREATE INDEX IF NOT EXISTS idx_ref_instrument_class ON ref_instrument (asset_class);

-- Factor registry. `construction` is the versioned JSONB spec that build_factors.py
-- executes, so every factor definition change is auditable.
CREATE TABLE IF NOT EXISTS ref_factor (
    factor_id             TEXT PRIMARY KEY,
    block_id              TEXT NOT NULL REFERENCES ref_factor_block(block_id),
    name                  TEXT NOT NULL,
    construction          JSONB NOT NULL,
    base_ccy              TEXT NOT NULL DEFAULT 'USD',
    orthogonalize_against TEXT[] NOT NULL DEFAULT '{}',
    hierarchy_level       INTEGER NOT NULL DEFAULT 0,  -- lower levels orthogonalised first
    is_active             BOOLEAN NOT NULL DEFAULT TRUE,
    inception_date        DATE,
    version               INTEGER NOT NULL DEFAULT 1,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ref_factor_block ON ref_factor (block_id, hierarchy_level);

-- Trading calendar, derived from observed data rather than a holiday library.
CREATE TABLE IF NOT EXISTS ref_calendar (
    date              DATE PRIMARY KEY,
    is_trading_day    BOOLEAN NOT NULL DEFAULT TRUE,
    n_live_instruments INTEGER
);

INSERT INTO ref_factor_block (block_id, name, sort_order, description) VALUES
    ('equity',      'Equity Market',              1, 'Global, regional and sector equity; total return minus cash'),
    ('style',       'Equity Style',             2, 'Value, Momentum, Quality, Low Vol, Size, Profitability'),
    ('rates',       'Rates',                    3, 'Level, slope, curvature per curve; duration-normalised returns'),
    ('credit',      'Credit',                   4, 'IG, HY, Loans, EM hard/local; excess over duration-matched govt'),
    ('fx',          'FX',                       5, 'Spot vs base currency plus carry basket'),
    ('commodity',   'Commodities',              6, 'Futures total return including roll yield'),
    ('volatility',  'Volatility',               7, 'VIX futures strategy return, variance premium, rates vol'),
    ('liquidity',   'Liquidity & Stress',   8, 'Funding spreads, stress indices, cross-asset risk-off basket'),
    ('arp',         'Alternative Risk Premia',  9, 'Rule-based trend, carry, value, momentum across assets')
ON CONFLICT (block_id) DO UPDATE
    SET name = EXCLUDED.name, sort_order = EXCLUDED.sort_order, description = EXCLUDED.description;
