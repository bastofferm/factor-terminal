-- 013_beta_overlap.sql — repair NaN-as-NULL, and record the comparison basis.
--
-- Two problems, both visible as gaps in the Beta Stability chart.
--
-- First: the stability columns were holding IEEE NaN rather than NULL. The writer
-- correctly produced None for "not computed", but a pandas DataFrame turns None in
-- a float column into NaN, and double precision accepts NaN happily. The result is
-- worse than a cosmetic gap — NaN is not NULL, so `IS NULL` never matched, count()
-- counted them, and every aggregate came back NaN. avg(beta_shift_l1) for AAPL read
-- NaN where the true mean was 6.41. bulk_insert now maps non-finite to NULL for
-- every caller, so this is a one-off repair of what is already stored.
--
-- Second: the comparison was skipped whenever the usable factor set changed between
-- windows, which is 169 of the 178 gaps. Style ETFs start in 2012 and coverage
-- filtering moves factors in and out, so the basis changes often — and a gap lands
-- exactly where an analyst most wants to know whether the shared exposures moved.
-- The shift is now measured on the factors common to both windows, and this column
-- records how many those were, so a narrowed basis is visible rather than implied.

ALTER TABLE fact_regression_meta
    ADD COLUMN IF NOT EXISTS beta_overlap INTEGER;

COMMENT ON COLUMN fact_regression_meta.beta_overlap IS
    'Factors common to this window and the previous one; the basis beta_shift_l1 '
    'and beta_corr_prev were measured on. NULL for the first window of a series.';

UPDATE fact_regression_meta SET
    beta_shift_l1  = CASE WHEN beta_shift_l1  = 'NaN'::float8 THEN NULL ELSE beta_shift_l1  END,
    beta_corr_prev = CASE WHEN beta_corr_prev = 'NaN'::float8 THEN NULL ELSE beta_corr_prev END
WHERE beta_shift_l1 = 'NaN'::float8 OR beta_corr_prev = 'NaN'::float8;

-- The same hazard applies anywhere else a DataFrame reached a float column.
UPDATE fact_risk_forecast SET
    sigma_realized_ann = CASE WHEN sigma_realized_ann = 'NaN'::float8 THEN NULL ELSE sigma_realized_ann END,
    bias_ratio         = CASE WHEN bias_ratio         = 'NaN'::float8 THEN NULL ELSE bias_ratio END
WHERE sigma_realized_ann = 'NaN'::float8 OR bias_ratio = 'NaN'::float8;
