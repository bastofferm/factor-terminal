"""Specs keep their identity, and the agent's plumbing leaves old callers alone.

A spec_id is a cache key and a foreign key. Adding a field to Spec must not
change the id of any spec stored before the field existed, and excluding a
diagnostics-blocked factor must not change the id of the spec being estimated.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.ai import providers
from backend.pipeline import run_estimation as est

FACTORS = ["eq_global", "eq_us", "rt_us_level", "cr_hy", "fx_usd"]


def test_new_fields_at_their_defaults_leave_the_hash_alone():
    spec = est.Spec(factor_set=FACTORS)
    assert "dimson_factors" not in spec.canonical()
    legacy = {k: v for k, v in spec.canonical().items()}
    blob = json.dumps(legacy, sort_keys=True, separators=(",", ":"))
    import hashlib
    assert spec.spec_id == hashlib.sha256(blob.encode()).hexdigest()[:32]


def test_a_dimson_subset_is_a_different_spec():
    a = est.Spec(factor_set=FACTORS, dimson_lags=1)
    b = est.Spec(factor_set=FACTORS, dimson_lags=1, dimson_factors=["eq_global", "fx_usd"])
    c = est.Spec(factor_set=FACTORS, dimson_lags=1, dimson_factors=["fx_usd", "eq_global"])
    assert a.spec_id != b.spec_id
    assert b.spec_id == c.spec_id, "the subset is a set; its order is not part of the spec"


def test_from_row_round_trips_and_refuses_a_mismatch():
    spec = est.Spec(factor_set=FACTORS, window_days=504, estimator="ridge",
                    ridge_lambda=0.1, weighting="ewma", ewma_halflife=126,
                    dimson_lags=1, dimson_factors=["eq_global"], orthogonalized=False)
    row = {**{k: v for k, v in spec.canonical().items()},
           "factor_set": json.dumps(spec.factor_set),
           "dimson_factors": json.dumps(spec.dimson_factors), "spec_id": spec.spec_id}
    assert est.Spec.from_row(row).spec_id == spec.spec_id
    row["spec_id"] = "0" * 32
    try:
        est.Spec.from_row(row)
    except ValueError:
        pass
    else:
        raise AssertionError("a row whose parameters do not hash to its id must be refused")


def test_excluding_a_factor_keeps_the_spec_and_drops_the_factor():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    F = pd.DataFrame(rng.standard_normal((400, len(FACTORS))) * 0.01, index=idx, columns=FACTORS)
    y = pd.Series(F["eq_global"] * 1.1 + rng.standard_normal(400) * 0.005, index=idx)
    spec = est.Spec(factor_set=FACTORS, window_days=252, step_days=63)
    before = spec.spec_id
    loadings, meta, covs = est.estimate_with_cov(y, F, spec, exclude={"cr_hy"})
    assert spec.spec_id == before and spec.factor_set == FACTORS
    assert "cr_hy" not in set(loadings["factor_id"])
    names, V = next(iter(covs.values()))
    assert "cr_hy" not in names and V.shape == (len(names), len(names))


def test_json_mode_is_sent_only_when_asked_for():
    plain = providers._payload([{"role": "user", "content": "x"}], "m", 10, 0.2, False)
    assert "response_format" not in plain
    js = providers._payload([{"role": "user", "content": "x"}], "m", 10, 0.2, False,
                            json_mode=True)
    assert js["response_format"] == {"type": "json_object"}
