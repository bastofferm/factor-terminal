-- 011_risk_reliability.sql — mark forecasts built on ill-conditioned windows.
--
-- With 40 factors on a 252-day window, a period in which only a handful of factors
-- yet exist (2002-03 here) leaves the design matrix near-singular: betas explode in
-- offsetting pairs, and beta' Sigma beta with them. AAPL produced a 360% predicted
-- volatility that way, which then dominated the Mincer-Zarnowitz regression as a
-- single leverage point.
--
-- The forecast is still stored — suppressing the model's own output would hide the
-- problem — but it is flagged, and the backtest ignores unreliable rows.
-- Model uncertainty is penalised rather than ignored: a forecast built on a
-- near-singular design is recorded with the reason it cannot be trusted.

ALTER TABLE fact_risk_forecast
    ADD COLUMN IF NOT EXISTS condition_number DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS max_vif          DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS is_reliable      BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS unreliable_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_risk_forecast_reliable
    ON fact_risk_forecast (spec_id, instrument_id, is_reliable, as_of_date DESC);

ALTER TABLE fact_risk_backtest
    ADD COLUMN IF NOT EXISTS n_excluded INTEGER NOT NULL DEFAULT 0;
