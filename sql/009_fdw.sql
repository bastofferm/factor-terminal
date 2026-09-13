-- 009_fdw.sql — read-only foreign tables over the xbrl_sec warehouse.
--
-- Keeps the factors DB standalone while avoiding a second copy of credentials or a
-- Python row-marshalling step: sync becomes a single server-side INSERT ... SELECT.
--
-- The USER MAPPING is created by apply_schema.py, which has the password from the
-- environment (PGPASSWORD); it cannot be expressed portably here.

CREATE EXTENSION IF NOT EXISTS postgres_fdw;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_foreign_server WHERE srvname = 'warehouse') THEN
        CREATE SERVER warehouse
            FOREIGN DATA WRAPPER postgres_fdw
            OPTIONS (host '127.0.0.1', port '5432', dbname 'xbrl_sec',
                     fetch_size '50000', use_remote_estimate 'true');
    END IF;
END
$$;
