-- 004_diagnostics.sql — stationarity and data-quality battery (plan section 3)
--
-- One row per (series, as-of date, window). Diagnostics are computed both on the
-- full history and on the trailing estimation window: a series can be stationary
-- over 20 years and broken over the last six months.

CREATE TABLE IF NOT EXISTS fact_series_diagnostics (
    series_key        TEXT NOT NULL,
    series_type       TEXT NOT NULL,   -- factor | instrument | level
    as_of_date        DATE NOT NULL,
    window_days       INTEGER NOT NULL,   -- 0 = full history

    -- Unit root: ADF (H0 = unit root) vs KPSS (H0 = stationary). The joint reading
    -- is the verdict; neither test alone is sufficient.
    adf_stat          DOUBLE PRECISION,
    adf_p             DOUBLE PRECISION,
    adf_lags          INTEGER,
    kpss_stat         DOUBLE PRECISION,
    kpss_p            DOUBLE PRECISION,
    kpss_lags         INTEGER,
    pp_stat           DOUBLE PRECISION,
    pp_p              DOUBLE PRECISION,

    -- Structural break: Zivot-Andrews allows one endogenous break under the null.
    za_stat           DOUBLE PRECISION,
    za_p              DOUBLE PRECISION,
    za_break_date     DATE,

    -- Stale pricing: Lo-MacKinlay variance ratios, heteroskedasticity-robust.
    -- VR < 1 => mean reversion / bid-ask bounce; VR > 1 => trending or smoothed NAV.
    vr2               DOUBLE PRECISION,
    vr5               DOUBLE PRECISION,
    vr10              DOUBLE PRECISION,
    vr2_p             DOUBLE PRECISION,
    vr5_p             DOUBLE PRECISION,
    vr10_p            DOUBLE PRECISION,

    -- Autocorrelation and conditional heteroskedasticity.
    lb10_stat         DOUBLE PRECISION,
    lb10_p            DOUBLE PRECISION,
    lb_sq10_stat      DOUBLE PRECISION,
    lb_sq10_p         DOUBLE PRECISION,
    arch_lm_stat      DOUBLE PRECISION,
    arch_lm_p         DOUBLE PRECISION,
    ac1               DOUBLE PRECISION,   -- first-order autocorrelation

    -- Distribution.
    mean_ann          DOUBLE PRECISION,
    sd_ann            DOUBLE PRECISION,
    skew              DOUBLE PRECISION,
    excess_kurtosis   DOUBLE PRECISION,
    jb_stat           DOUBLE PRECISION,
    jb_p              DOUBLE PRECISION,

    -- Coverage and liquidity.
    n_obs             INTEGER,
    n_gaps            INTEGER,
    zero_return_share DOUBLE PRECISION,
    max_abs_return    DOUBLE PRECISION,

    -- Verdict: fail blocks the series from estimation; warn surfaces amber in the UI
    -- and feeds the model confidence score (PDF section 7.3).
    verdict           TEXT NOT NULL,
    verdict_reason    TEXT,
    flags             TEXT[] NOT NULL DEFAULT '{}',
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (series_key, series_type, as_of_date, window_days),
    CONSTRAINT fact_series_diagnostics_verdict_chk CHECK (verdict IN ('pass','warn','fail'))
);

CREATE INDEX IF NOT EXISTS idx_diagnostics_verdict
    ON fact_series_diagnostics (verdict, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_diagnostics_type
    ON fact_series_diagnostics (series_type, as_of_date DESC);

-- Latest diagnostic per series, which is what the UI reads.
CREATE OR REPLACE VIEW v_series_diagnostics_latest AS
SELECT DISTINCT ON (series_key, series_type, window_days) *
FROM fact_series_diagnostics
ORDER BY series_key, series_type, window_days, as_of_date DESC;
