-- 010_fdw_import.sql — (re)import the foreign schema.
--
-- Split from 009 because IMPORT FOREIGN SCHEMA is not idempotent: dropping and
-- re-importing is how a warehouse schema change gets picked up. Runs only after
-- apply_schema.py has created the user mapping.

DROP SCHEMA IF EXISTS warehouse_sec CASCADE;
CREATE SCHEMA warehouse_sec;

IMPORT FOREIGN SCHEMA sec
    LIMIT TO (fact_cross_asset, dim_cross_asset, fact_macro, ref_macro_series,
              fact_fama_french, dim_ff_dataset, fact_prices_us, fact_prices_jp, fact_fx)
    FROM SERVER warehouse INTO warehouse_sec;
