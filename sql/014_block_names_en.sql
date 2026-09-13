-- 014_block_names_en.sql — the last two German block names.
--
-- The block registry was seeded straight from the concept note's section headings,
-- so seven of the nine arrived in English and two did not. The interface is
-- English throughout, and "Liquiditaet und Stress" appearing in a factor's profile
-- next to "Credit" and "Volatility" reads as a bug rather than as provenance.
--
-- 001_reference.sql is corrected too, so a fresh database never has the problem;
-- this repairs the ones already seeded.

UPDATE ref_factor_block SET name = 'Equity Market'    WHERE block_id = 'equity';
UPDATE ref_factor_block SET name = 'Liquidity & Stress' WHERE block_id = 'liquidity';
