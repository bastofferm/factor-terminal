-- 015_security_catalogue.sql — every security the warehouse can price, searchable.
--
-- The Loadings Lab could only offer what had already been mirrored into
-- ref_instrument: 198 rows, almost all of them factor inputs. The warehouse holds
-- 5,376 US securities, 3,878 Japanese ones and 182 cross-asset funds, and any of
-- them can be estimated — sync_security pulls one on demand.
--
-- Searching the warehouse directly on every keystroke is not an option. Deriving
-- coverage means aggregating fact_prices_us, which is 15.8M rows and takes eight
-- seconds; fact_prices_jp is another 15.4M. So the catalogue is materialised here
-- and refreshed by the nightly sync, and a search is then an index lookup against
-- ten thousand local rows.
--
-- This is a catalogue of what *could* be estimated. ref_instrument stays the record
-- of what actually has returns stored, and the two are deliberately separate: a
-- security appearing here does not mean its history has been pulled yet.

CREATE TABLE IF NOT EXISTS ref_security (
    instrument_id  TEXT PRIMARY KEY,   -- US:AAPL, JP:7203, or a bare ETF ticker
    ticker         TEXT NOT NULL,      -- as the warehouse knows it
    name           TEXT,
    security_type  TEXT NOT NULL,      -- equity | etf | index | futures | fx | crypto
    jurisdiction   TEXT,               -- US | JP | global
    exchange       TEXT,
    sector         TEXT,
    currency       TEXT,
    source_table   TEXT NOT NULL,      -- which warehouse table prices it
    first_date     DATE,
    last_date      DATE,
    n_obs          INTEGER,
    refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ref_security_type_chk CHECK (
        security_type IN ('equity', 'etf', 'index', 'futures', 'fx', 'crypto'))
);

-- Search is "ticker starts with" and "name contains", the two things anyone types.
-- text_pattern_ops so a LIKE 'AAP%' can use the index under any collation.
CREATE INDEX IF NOT EXISTS idx_ref_security_ticker
    ON ref_security (ticker text_pattern_ops);
CREATE INDEX IF NOT EXISTS idx_ref_security_type
    ON ref_security (security_type, n_obs DESC);

-- Case-insensitive name search. A trigram index would be better but pg_trgm is not
-- guaranteed present, and lower(name) with a prefix match covers what is typed.
CREATE INDEX IF NOT EXISTS idx_ref_security_name
    ON ref_security (lower(name) text_pattern_ops);
