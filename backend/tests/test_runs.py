"""The run inspector: does it describe the job that actually ran?

Two failure modes this guards against, both silent. A job renamed in the pipeline
leaves the description table pointing at a name nothing emits any more, and the
panel goes blank with no error. And a spec key that stops being unpacked drops the
one parameter distinguishing two otherwise identical runs — which is exactly the
`orthogonalized` flag, the choice that changes what a beta means.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from backend.app import runs

# Every spec_id-bearing job resolves a dim_model_spec row; these are the columns
# the router selects for it.
SPEC = {
    "spec_id": "abc123", "name": None, "estimator": "ols", "window_days": 252,
    "step_days": 21, "weighting": "equal", "ewma_halflife": None, "hac_lags": None,
    "ridge_lambda": 0.0, "dimson_lags": 0, "orthogonalized": True,
    "winsor_lo": None, "winsor_hi": None, "min_obs": 126, "base_ccy": "USD",
    "n_factors": 40,
}


def labels(chips: list[dict]) -> list[str]:
    return [c["label"] for c in chips]


def value_of(chips: list[dict], label: str) -> str | None:
    return next((c["value"] for c in chips if c["label"] == label), None)


# ---------------------------------------------------------------------------
# descriptions
# ---------------------------------------------------------------------------

def pipeline_job_names() -> set[str]:
    """The JOB constant of every pipeline module that has one."""
    import backend.pipeline as pipeline

    names = set()
    for mod in pkgutil.iter_modules(pipeline.__path__):
        m = importlib.import_module(f"backend.pipeline.{mod.name}")
        job = getattr(m, "JOB", None)
        if isinstance(job, str):
            names.add(job)
    return names


def test_every_job_that_can_run_has_a_description():
    """Rename a job and this fails, instead of the panel quietly going blank."""
    missing = [j for j in pipeline_job_names() if not runs.describe(j)]
    assert not missing, f"jobs with no description: {sorted(missing)}"


def test_descriptions_say_what_the_job_does_not_what_it_is_called():
    for job in pipeline_job_names():
        text = runs.describe(job)
        assert len(text) > 80, f"{job}: description is too thin to be useful"
        assert not text.lower().startswith(job.replace("_", " ")), \
            f"{job}: description just restates the name"


def test_an_unknown_job_yields_an_empty_description_rather_than_raising():
    assert runs.describe("something_new") == ""


# ---------------------------------------------------------------------------
# chips
# ---------------------------------------------------------------------------

def test_the_factor_panel_is_the_first_chip_and_is_emphasised():
    """Two risk runs differing only in the panel are identical in `scope`. If this
    chip is missing there is nothing on screen that tells them apart."""
    chips = runs.chips("run_risk", "incremental",
                       {"spec_id": "abc123", "horizon": 21, "cov_method": "blend"},
                       SPEC)
    assert chips[0]["label"] == "panel"
    assert chips[0]["value"] == "orthogonalised"
    assert chips[0]["emphasis"] is True

    raw = runs.chips("run_risk", "incremental", {"spec_id": "abc123"},
                     {**SPEC, "orthogonalized": False})
    assert raw[0]["value"] == "raw"


def test_risk_chips_carry_the_horizon_and_the_covariance_method():
    chips = runs.chips("run_risk", "incremental",
                       {"spec_id": "abc123", "horizon": 63, "cov_method": "ewma"},
                       SPEC)
    assert value_of(chips, "horizon") == "63d"
    assert value_of(chips, "covariance") == "ewma"
    assert "63 trading days" in next(c["hint"] for c in chips if c["label"] == "horizon")


def test_defaults_that_nobody_chose_are_not_shown():
    """A chip for every column would be a JSON dump with rounded corners. Zero ridge,
    equal weighting and no Dimson lags are the defaults and carry no information."""
    chips = runs.chips("run_estimation", "incremental", {"spec_id": "abc123"}, SPEC)
    assert "ridge lambda" not in labels(chips)
    assert "weighting" not in labels(chips)
    assert "Dimson lags" not in labels(chips)
    assert "winsorised" not in labels(chips)


def test_non_default_estimator_settings_do_show():
    chips = runs.chips("run_estimation", "incremental", {"spec_id": "abc123"},
                       {**SPEC, "ridge_lambda": 1.0, "weighting": "ewma",
                        "ewma_halflife": 60, "dimson_lags": 2,
                        "winsor_lo": 0.01, "winsor_hi": 0.99})
    assert value_of(chips, "ridge lambda") == "1.0"
    assert value_of(chips, "weighting") == "ewma"
    assert value_of(chips, "Dimson lags") == "2"
    assert value_of(chips, "winsorised") == "1%-99%"


def test_the_orthogonalisation_mode_of_a_build_is_emphasised():
    chips = runs.chips("build_factors", "full_sample", {"orth_mode": "full_sample"})
    assert chips[0]["label"] == "mode"
    assert chips[0]["emphasis"] is True
    assert "future information" in chips[0]["hint"]


def test_diagnostics_chips_name_the_window_and_the_series_kinds():
    chips = runs.chips("run_diagnostics", "incremental",
                       {"window": 252, "kinds": ["factor", "instrument"]})
    assert value_of(chips, "window") == "252d"
    assert value_of(chips, "series") == "factor, instrument"


def test_scope_arriving_as_a_json_string_is_handled():
    """asyncpg returns JSONB as a dict, psycopg2 as a string. Both reach this."""
    chips = runs.chips("run_diagnostics", "incremental", '{"window": 63}')
    assert value_of(chips, "window") == "63d"


def test_a_malformed_scope_does_not_take_the_panel_down():
    assert runs.chips("run_diagnostics", None, "not json at all") == []
    assert runs.chips("run_diagnostics", None, None) == []


def test_every_chip_hint_is_a_sentence_not_a_restatement():
    chips = runs.chips("run_risk", "incremental",
                       {"spec_id": "abc123", "horizon": 21, "cov_method": "blend"},
                       SPEC)
    for c in chips:
        if c["hint"]:
            assert c["hint"].lower() != c["value"].lower()
            assert len(c["hint"]) > 15, f"{c['label']}: hint adds nothing"


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------

def test_outcome_counts_items_and_rows():
    text = runs.outcome(
        {"job": "build_factors", "status": "succeeded", "rows_out": 200_072,
         "duration_seconds": 12.4},
        {"succeeded": 40})
    assert "40 factors" in text
    assert "200,072 rows" in text
    assert "12 seconds" in text


def test_series_is_not_pluralised_into_seriess():
    text = runs.outcome(
        {"job": "run_diagnostics", "status": "succeeded", "rows_out": 474,
         "duration_seconds": 34},
        {"succeeded": 238})
    assert "238 series" in text
    assert "seriess" not in text


def test_a_single_item_reads_in_the_singular():
    text = runs.outcome(
        {"job": "run_estimation", "status": "succeeded", "rows_out": 1959,
         "duration_seconds": 0.4},
        {"succeeded": 1})
    assert "1 security processed" in text
    assert "under a second" in text


def test_failures_and_skips_are_reported_rather_than_averaged_away():
    text = runs.outcome(
        {"job": "build_factors", "status": "partial", "rows_out": 100,
         "duration_seconds": 5},
        {"succeeded": 38, "failed": 1, "skipped": 1})
    assert "1 failed" in text
    assert "1 skipped" in text


def test_a_failed_run_leads_with_its_error():
    text = runs.outcome(
        {"job": "ingest_yahoo", "status": "failed", "error": "rate limited by Yahoo"},
        {})
    assert text.startswith("Failed:")
    assert "rate limited" in text


def test_a_running_run_says_so_rather_than_reporting_zero():
    text = runs.outcome({"job": "sync_warehouse", "status": "running"}, {})
    assert text == "Still running."


@pytest.mark.parametrize("seconds,expected", [
    (0.2, "under a second"), (34, "34 seconds"), (600, "10 minutes"),
    (7200, "2.0 hours"),
])
def test_durations_read_at_the_right_scale(seconds, expected):
    text = runs.outcome(
        {"job": "sync_warehouse", "status": "succeeded", "rows_out": 1,
         "duration_seconds": seconds},
        {"succeeded": 1})
    assert expected in text


def test_one_second_is_not_reported_as_1_seconds():
    text = runs.outcome(
        {"job": "sync_warehouse", "status": "succeeded", "rows_out": 8638,
         "duration_seconds": 1.2},
        {"succeeded": 8})
    assert "1 second" in text
    assert "1 seconds" not in text
