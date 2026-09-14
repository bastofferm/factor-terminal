"""Tests for the chatbot's grounding machinery.

The corpus and context tests are pure and always run. The tests that actually call
DeepSeek are marked `live` and skipped unless a key is configured and
FACTORS_LIVE_LLM=1 is set — they cost money and need the network, so they should be
a deliberate choice rather than something a routine `pytest` triggers.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.app.ai import context, corpus, prompts, providers

live = pytest.mark.skipif(
    os.environ.get("FACTORS_LIVE_LLM") != "1" or not providers.has_api_key(),
    reason="set FACTORS_LIVE_LLM=1 with a DeepSeek key to run the live calls",
)


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------

def test_corpus_builds_and_is_substantial():
    text = corpus.build()
    assert len(text) > 10_000
    assert "METHODOLOGY REFERENCE" in text


def test_corpus_contains_every_factor():
    """The catalogue is assembled from factor_defs, so a factor added there must
    appear without anyone remembering to update a second document."""
    from backend.pipeline import factor_defs

    text = corpus.build()
    missing = [f["id"] for f in factor_defs.FACTORS if f["id"] not in text]
    assert not missing, f"factors absent from the corpus: {missing}"


def test_corpus_carries_the_reasoning_not_just_the_rules():
    """The point of lifting module docstrings is that they explain *why*. If these
    phrases vanish, the corpus has become a list of parameters and the bot can no
    longer answer the questions it exists for."""
    text = corpus.build()
    for phrase in [
        "Freedman-Diaconis",          # distribution
        "Newey-West",                 # regression
        "Ledoit",                     # covariance
        "unit root",                  # stationarity
        "roll",                       # transforms / commodity decision
        "lookahead",                  # orthogonalisation
    ]:
        assert phrase.lower() in text.lower(), f"corpus lost {phrase!r}"


def test_corpus_carries_the_exact_formula_for_every_factor():
    """Asked how a factor is built, the assistant must be able to answer with the
    arithmetic rather than with the name of a construction method."""
    from backend.pipeline import factor_defs, formula

    text = corpus.build()
    for f in factor_defs.FACTORS:
        first_line = formula.for_factor(f)["construction"]["plain"].splitlines()[0]
        assert first_line in text, f"{f['id']}: formula absent from the corpus"


def test_corpus_distinguishes_the_two_panels():
    """The commonest way to misread a number on any of these pages is to take a raw
    figure for an orthogonalised one. The corpus has to name the distinction, both
    stored columns, and the constraint that ties betas to their own covariance."""
    text = corpus.build()
    for phrase in ["ret_excess", "ret_orth", "dim_model_spec.orthogonalized",
                   "beta' Sigma beta"]:
        assert phrase in text, f"corpus lost {phrase!r}"


def test_corpus_says_which_factors_are_the_same_in_both_panels():
    from backend.pipeline import factor_defs

    text = corpus.build()
    for f in factor_defs.FACTORS:
        if f.get("orth"):
            continue
        block = text[text.index(f"**{f['id']}**"):]
        block = block[:block.index("\n- **")] if "\n- **" in block else block
        assert "identical" in block, f"{f['id']} does not say both panels agree"


def test_corpus_states_its_known_limitations():
    text = corpus.build().lower()
    assert "approximation" in text          # FX carry
    assert "long-only" in text              # style proxies
    assert "mechanical" in text             # MZ attenuation


def test_corpus_is_byte_stable():
    """Context caching only pays if the prefix does not move between calls."""
    assert corpus.build() is corpus.build()


def test_corpus_info_matches_the_text():
    info = corpus.info()
    assert info["chars"] == len(corpus.build())
    assert info["approx_tokens"] > 1000


# ---------------------------------------------------------------------------
# snapshot rendering
# ---------------------------------------------------------------------------

def test_snapshot_renders_the_numbers_it_was_given():
    out = context.render_snapshot({
        "page": "/factors", "factor_id": "cm_broad",
        "stats": {"vol_ann": 0.19412345, "n_obs": 5178},
    })
    assert "/factors" in out
    assert "cm_broad" in out
    assert "0.194123" in out          # rounded, not reformatted away
    assert "5178" in out


def test_snapshot_is_capped():
    huge = {"page": "/factors", "blob": ["x" * 100] * 500}
    out = context.render_snapshot(huge)
    assert len(out) < context.MAX_SNAPSHOT_CHARS + 200
    assert "truncated" in out


def test_empty_snapshot_renders_nothing():
    assert context.render_snapshot(None) == ""
    assert context.render_snapshot({}) == ""


def test_non_finite_values_do_not_reach_the_prompt():
    """NaN is not valid JSON and would break the client mid-stream."""
    out = context.render_snapshot({
        "page": "/risk", "stats": {"a": float("nan"), "b": float("inf"), "c": 1.5}})
    assert "NaN" not in out and "Infinity" not in out
    assert "1.5" in out
    json.loads(out.split("```json")[1].split("```")[0])


# ---------------------------------------------------------------------------
# factor mention matching
# ---------------------------------------------------------------------------

KNOWN = ["eq_global", "eq_em", "cm_broad", "rt_us_level", "sty_value"]


def test_mentioned_factors_matches_on_word_boundaries():
    assert context.mentioned_factors("how is cm_broad built?", KNOWN) == ["cm_broad"]
    assert context.mentioned_factors("compare eq_global and eq_em", KNOWN) == \
        ["eq_global", "eq_em"]


def test_mentioned_factors_does_not_match_loosely():
    """'equity' must not drag in every eq_* factor, and a substring of an id is not
    a mention of it."""
    assert context.mentioned_factors("what about equity in general?", KNOWN) == []
    assert context.mentioned_factors("tell me about global markets", KNOWN) == []


def test_mentioned_factors_is_capped():
    many = [f"f{i}" for i in range(20)]
    question = " ".join(many)
    assert len(context.mentioned_factors(question, many)) <= context.MAX_RETRIEVED_FACTORS


# ---------------------------------------------------------------------------
# prompt contract
# ---------------------------------------------------------------------------

def test_system_prompt_states_the_numeric_contract():
    """These clauses are what stop invented figures. Losing one silently would be a
    real regression, and nothing else in the suite would catch it."""
    s = prompts.SYSTEM
    assert "CANONICAL DATA is correct" in s
    assert "Do not print the conflicting figure" in s
    assert "name the page that shows it" in s
    assert "no tools" in s.lower()


def test_every_page_has_starter_questions():
    for page in ("/", "/factors", "/matrix", "/loadings", "/risk"):
        assert prompts.STARTERS.get(page), f"no starters for {page}"


# ---------------------------------------------------------------------------
# provider safety
# ---------------------------------------------------------------------------

def test_endpoint_refuses_a_foreign_host():
    """A misconfigured base URL must never receive the API key."""
    with pytest.raises(providers.LLMError, match="Refusing"):
        providers._endpoint("https://evil.example.com/v1")
    with pytest.raises(providers.LLMError):
        providers._endpoint("http://api.deepseek.com/v1")   # plain http


def test_endpoint_accepts_the_real_host():
    assert providers._endpoint().startswith("https://api.deepseek.com")


def test_key_is_redacted_from_error_text():
    assert "sk-secret" not in providers._redact("boom sk-secret boom", "sk-secret")


def test_sanitize_strips_stray_markup():
    assert providers.sanitize("<|DSML|>hello") == "hello"
    assert providers.sanitize("plain text") == "plain text"


# ---------------------------------------------------------------------------
# live calls
# ---------------------------------------------------------------------------

@live
@pytest.mark.asyncio
async def test_live_model_returns_content_with_the_full_corpus():
    """The failure this guards against: reasoning tiers spending the whole output
    budget on hidden reasoning and returning nothing."""
    content, finish, usage = await providers.complete([
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "system", "content": corpus.build()},
        {"role": "user", "content": "In one sentence, what does the liveness gate do?"},
    ], max_tokens=300)

    assert content.strip(), f"empty content, finish_reason={finish!r}"
    assert usage.prompt_tokens > 4000


@live
@pytest.mark.asyncio
async def test_live_answer_quotes_the_snapshot_figure():
    canonical = context.render_snapshot({
        "page": "/factors", "factor_id": "cm_broad",
        "stats": {"vol_ann": 0.194, "n_obs": 5178},
    })
    content, _, _ = await providers.complete([
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "system", "content": corpus.build()},
        {"role": "system", "content": prompts.canonical_header() + canonical},
        {"role": "user", "content": "What is this factor's annualised volatility?"},
    ], max_tokens=200)

    assert "19.4" in content


@live
@pytest.mark.asyncio
async def test_live_answer_refuses_a_figure_it_was_not_given():
    canonical = context.render_snapshot({
        "page": "/factors", "factor_id": "cm_broad",
        "stats": {"vol_ann": 0.194},
    })
    content, _, _ = await providers.complete([
        {"role": "system", "content": prompts.SYSTEM},
        {"role": "system", "content": corpus.build()},
        {"role": "system", "content": prompts.canonical_header() + canonical},
        {"role": "user", "content": "What is AAPL's predicted volatility?"},
    ], max_tokens=300)

    lowered = content.lower()
    assert any(p in lowered for p in ("not in view", "not in the", "cannot", "can't",
                                      "no ", "loadings lab", "risk lens")), content
