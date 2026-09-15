-- 010_fdw_import.sql — (re)import the foreign schema.
--
-- Split from 009 because IMPORT FOREIGN SCHEMA is not idempotent: dropping and
-- re-importing is how a warehouse schema change gets picked up. Runs only after
-- apply_schema.py has created the user mapping.

DROP SCHEMA IF EXISTS warehouse_sec CASCADE;
CREATE SCHEMA warehouse_sec;

-- dim_company_us and dim_company_jp are here for the security catalogue: they
-- carry the name, exchange and GICS sector for the 5,376 US and 6,025 Japanese
-- securities the price tables cover, which is what makes those searchable by
-- something other than a bare ticker.
IMPORT FOREIGN SCHEMA sec
    LIMIT TO (fact_cross_asset, dim_cross_asset, fact_macro, ref_macro_series,
              fact_fama_french, dim_ff_dataset, fact_prices_us, fact_prices_jp,
              fact_fx, dim_company_us, dim_company_jp)
    FROM SERVER warehouse INTO warehouse_sec;
