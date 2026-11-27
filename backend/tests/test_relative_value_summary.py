"""The DeepSeek summary of a relative-value run: the payload, the number check,
and the call, against a scripted provider.

The scripted tests always run. The one that calls DeepSeek is marked `live` and
skipped unless a key is configured and FACTORS_LIVE_LLM=1 is set, as in
test_chat.py.
"""

from __future__ import annotations

import json
import os

import pytest

from backend.app.ai import providers, summarise


def _run() -> dict:
    """A small run in the pipeline's shape."""
    def view(vid, label, long, short, licence, z, stance, score):
        return {
            "id": vid, "label": label, "long": long, "short": short, "note": "note",
            "signals": {
                "value": {"z": z, "raw": 0.0123, "pct": 0.18, "from": "1982-01-31",
                          "label": "spread", "licence": licence},
                "carry": {"z": 0.07},
                "trend": {"z": 1.21, "ret": 0.0525, "vol": 0.06, "above_sma": True},
                "macro": None,
            },
            "score": score, "stance": stance, "change": 0.05, "act": False,
            "tilt": 0.004, "te": 0.0021, "weight_short": 1.1, "pair_vol": 0.05,
            "anchor": {"pair_excess": 0.0112, "per_risk": 0.22},
        }
    return {
        "run_id": "abc", "data_as_of": "2026-10-28",
        "forecast_run": {"run_id": "f1", "as_of": "2026-10-27"},
        "macro": {"regime": "stagflation", "growth": {"z": -0.14}, "inflation": {"z": 0.22},
                  "vix": {"level": 17.31, "z": -0.41}},
        "views": [view("hy_vs_ig", "US high yield vs investment grade", "HYG", "LQD", None,
                       -1.16, "neutral", -0.01),
                  view("eu_vs_us_eq", "European vs US equities, hedged", "VGK", "SPY", "barclays",
                       1.25, "lean overweight", 0.75)],
        "classes": [{"label": "European equities", "fund": "VGK", "expected": 0.0667,
                     "excess": 0.0254, "per_risk": 0.15, "trend_z": 0.3}],
        "credit": {"as_of": "2026-10-28", "hy_oas": 0.0334, "ig_oas": 0.0091,
                   "hy_oas_change_3m": 0.19,
                   "hy": {"pd": 0.037, "excess": 0.0112, "cushion": 0.0032},
                   "hy_low_pd": {"pd": 0.02, "excess": 0.0214},
                   "ig": {"excess": 0.0085, "cushion": 0.00121}},
        "sizing": {"te_total": 0.0123, "te_budget": 0.02},
        "caveats": ["Credit spreads start on 2023-05-15."],
    }


# ---------------------------------------------------------------------------
# payload and numbers
# ---------------------------------------------------------------------------

def test_numbers_are_normalised():
    assert summarise.numbers("+2.30% and −0.62") == ["2.3", "0.62"]
    assert summarise.numbers("1,311 bp") == ["1311"]
    assert summarise.numbers("T10YIE, Q3 and the 12-1 month return") == ["12", "1"]
    assert summarise.numbers("on 2026-10-28") == ["2026", "10", "28"]


def test_payload_formats_figures_and_keeps_licensed_levels_out():
    p = summarise.payload(_run())
    hy, eu = p["views"]
    assert hy["value_z"] == "-1.16" and hy["score"] == "-0.01" and hy["band"] == "hold"
    assert hy["value_level"] == "1.23%"
    # Barclays' CAPE: the z-score and the percentile, never the level.
    assert "value_level" not in eu and eu["value_z"] == "+1.25"
    assert eu["value_percentile"] == "18th"
    assert [summarise._ordinal(p) for p in (0.0, 0.21, 0.93, 0.112, 1.0)] == \
        ["1st", "21st", "93rd", "11th", "100th"]
    assert p["credit"]["hy_excess_after_loss"] == "112 bp"
    assert p["credit"]["hy_cushion_per_year"] == "32 bp"
    assert "macro_z" not in hy                    # a view with no macro signal says nothing
    assert summarise.payload_sha(p) == summarise.payload_sha(summarise.payload(_run()))


def test_unverified_finds_invented_figures_only():
    allowed = summarise.allowed_numbers(summarise.payload(_run()))
    ok = {"headline": "High yield pays 112 bp after losses, with value at -1.16.",
          "views": [{"id": "hy_vs_ig", "read": "Trend +1.21 against value −1.16; hold."}],
          "classes": "", "timing": "Stagflation, growth z -0.14.", "risks": ["Three views."],
          "caveats": []}
    assert summarise.unverified(ok, allowed) == []
    bad = {**ok, "headline": "High yield pays 9.9% after losses and spreads rose 20%."}
    assert summarise.unverified(bad, allowed) == ["9.9", "20"]


def test_validate_orders_views_and_drops_unknown_ids():
    out = summarise.validate({"headline": "h", "views": [
        {"id": "eu_vs_us_eq", "read": "b"}, {"id": "made_up", "read": "x"},
        {"id": "hy_vs_ig", "read": "a"}], "risks": "one", "caveats": ["c1", "c2", "c3", "c4"]},
        ["hy_vs_ig", "eu_vs_us_eq"])
    assert [v["id"] for v in out["views"]] == ["hy_vs_ig", "eu_vs_us_eq"]
    assert out["risks"] == ["one"] and len(out["caveats"]) == 3
    with pytest.raises(summarise.SummaryError):
        summarise.validate({"views": []}, ["hy_vs_ig"])


# ---------------------------------------------------------------------------
# the call, scripted
# ---------------------------------------------------------------------------

GOOD = {"headline": "The scorecard leans toward European equities at +0.75.",
        "views": [{"id": "hy_vs_ig", "read": "Value -1.16 offsets trend +1.21; the band says hold."},
                  {"id": "eu_vs_us_eq", "read": "Value +1.25 drives a lean overweight."}],
        "classes": "European equities lead at +0.15 per unit of risk.",
        "timing": "The regime proxy reads stagflation.",
        "risks": ["A reversal of the trend."], "caveats": ["Credit spreads start on 2023-05-15."]}


@pytest.fixture
def scripted(monkeypatch):
    calls: list[list[dict]] = []
    replies: list[tuple[str, str]] = []

    async def fake(messages, model=None, max_tokens=0, temperature=0.0, timeout=0.0, json_mode=False):
        assert json_mode
        calls.append([dict(m) for m in messages])
        content, finish = replies.pop(0)
        return content, finish, providers.Usage(prompt_tokens=100, completion_tokens=50)

    monkeypatch.setattr(providers, "complete", fake)
    return calls, replies


@pytest.mark.asyncio
async def test_a_clean_reply_is_accepted_first_time(scripted):
    calls, replies = scripted
    replies.append((json.dumps(GOOD), "stop"))
    out = await summarise.summarise(_run(), model="deepseek-chat")
    assert out["attempts"] == 1 and out["usage"]["prompt_tokens"] == 100
    assert [v["id"] for v in out["content"]["views"]] == ["hy_vs_ig", "eu_vs_us_eq"]
    # The model saw the formatted payload, not the run.
    assert '"value_z": "-1.16"' in calls[0][1]["content"]
    assert "0.0123" not in calls[0][1]["content"]


@pytest.mark.asyncio
async def test_an_invented_number_is_sent_back_once(scripted):
    calls, replies = scripted
    replies += [(json.dumps({**GOOD, "headline": "European equities return 7.7% a year."}), "stop"),
                (json.dumps(GOOD), "stop")]
    out = await summarise.summarise(_run())
    assert out["attempts"] == 2 and out["usage"]["completion_tokens"] == 100
    assert "7.7" in calls[1][-1]["content"]
    assert calls[1][-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_a_second_invented_number_is_refused(scripted):
    _, replies = scripted
    bad = json.dumps({**GOOD, "headline": "European equities return 7.7% a year."})
    replies += [(bad, "stop"), (bad, "stop")]
    with pytest.raises(summarise.SummaryError) as exc:
        await summarise.summarise(_run())
    assert exc.value.unverified == ["7.7"]


@pytest.mark.asyncio
async def test_broken_json_gets_a_second_chance(scripted):
    _, replies = scripted
    replies += [("not json", "stop"), (json.dumps(GOOD), "stop")]
    assert (await summarise.summarise(_run()))["attempts"] == 2


@pytest.mark.asyncio
async def test_an_empty_reply_names_the_reasoning_tier_failure(scripted):
    _, replies = scripted
    replies.append(("", "length"))
    with pytest.raises(summarise.SummaryError, match="output budget"):
        await summarise.summarise(_run())


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(
    os.environ.get("FACTORS_LIVE_LLM") != "1" or not providers.has_api_key(),
    reason="set FACTORS_LIVE_LLM=1 with a DeepSeek key to run the live call",
)


@live
@pytest.mark.asyncio
async def test_live_summary_passes_the_number_check():
    out = await summarise.summarise(_run())
    assert out["content"]["headline"]
    assert {v["id"] for v in out["content"]["views"]} <= {"hy_vs_ig", "eu_vs_us_eq"}
