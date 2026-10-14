-- 017_sectors_horizon_portfolios.sql — sector factors, horizon-consistent risk,
-- and the portfolio-level backtest.
--
-- Additive only: the main checkout shares this database.

-- A factor's role. 'core' is the forty-factor model every regression may use;
-- 'sector' is the GICS sector factor a single stock can add for its own sector
-- only. Default factor sets, the overview's block counts and the covariance page
-- read 'core'; a sector factor enters a regression only when asked for.
ALTER TABLE ref_factor
    ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'core';

INSERT INTO ref_factor_block (block_id, name, sort_order, description) VALUES
    ('sector', 'Equity Sector', 10,
     'GICS sector ETFs residualised against the market and the cyclical tilt; '
     'a single stock carries only its own')
ON CONFLICT (block_id) DO UPDATE
    SET name = EXCLUDED.name, sort_order = EXCLUDED.sort_order,
        description = EXCLUDED.description;

-- The h-day forecast. sigma_pred_ann is the daily-consistent volatility a VaR is
-- read from; with a lead-lag adjustment the h-day figure adds the cross-day
-- covariance of factors that close at different times, and is scored against
-- h-day returns.
ALTER TABLE fact_risk_forecast
    ADD COLUMN IF NOT EXISTS sigma_horizon_ann DOUBLE PRECISION;

ALTER TABLE fact_risk_backtest
    ADD COLUMN IF NOT EXISTS daily_loss              DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS daily_loss_own_ewma     DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_t_daily_own_ewma     DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_p_daily_own_ewma     DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS horizon_loss            DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS horizon_loss_own_ewma   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_t_horizon_own_ewma   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS dm_p_horizon_own_ewma   DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS nw_lags                 INTEGER;

-- Portfolio-level validation: does the model's risk aggregate? One row per
-- portfolio and configuration of a validation run.
CREATE TABLE IF NOT EXISTS fact_portfolio_backtest (
    run_id        UUID NOT NULL,
    computed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    config        TEXT NOT NULL,          -- e.g. 'no sector', 'own sector'
    portfolio     TEXT NOT NULL,
    family        TEXT NOT NULL,          -- market | sector | pair | random | cross
    n_names       INTEGER NOT NULL,
    horizon_days  INTEGER NOT NULL,
    metrics       JSONB NOT NULL,
    PRIMARY KEY (run_id, config, portfolio)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_backtest_run
    ON fact_portfolio_backtest (computed_at DESC);
