"""DeepSeek's reading of a relative-value run, held to the numbers it was given.

The pipeline decides everything: the signals, the scores, the stance of every
view, the tilts. The model writes the prose around them - which signals drive a
view, where they disagree, what would change the picture - and nothing else. The
division is the model agent's (backend/agent): a language model is good at
saying what a table means and has no business producing the table.

Three things keep it to that.

  * It is shown a compact, already formatted extract of the run (`payload`), not
    the run. Every figure in it is a string as the page prints it, so quoting one
    verbatim is the easy path. Licensed valuation series are not in it: a view
    valued on Barclays' CAPE passes its z-score and percentile, never the level.
  * Its answer must be one JSON object of a fixed shape, in DeepSeek's JSON mode,
    and is validated: unknown view ids are dropped, lengths are capped.
  * Every number in the answer is looked up in the payload (`unverified`). The
    prompts in prompts.py ask the model not to compute figures of its own; here
    that is checked mechanically. A reply with a number the payload does not
    contain is sent back once with the offending numbers named, and refused if
    the second reply still has one. A refused summary is never stored or shown.

Signs are not compared - "fell 0.12" quotes "-0.12" faithfully - and small
integers are allowed, because "the 12-1 month return" and "three views" are
prose, not figures.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

from backend.app.ai import providers

MAX_TOKENS = 1600
TEMPERATURE = 0.1
SMALL_INTEGERS = frozenset(str(i) for i in range(13))

SYSTEM = """\
You write the summary at the top of the Relative Value & Timing page of a
multi-asset factor terminal. Your reader is an experienced financial economist.

You are given DATA, a JSON object computed by the terminal's pipeline. For each
relative-value view it holds four standardised signals (value, carry, trend,
macro; each a z-score clipped at +-3, positive when it favours the long leg),
their composite score, the stance the pipeline assigned, the one-month change in
the score and whether that clears the no-trade band, the suggested tilt, and the
ten-year building-block anchor. It also holds the macro regime proxy, the classes
with the most and least ten-year excess return per unit of risk, the credit
break-even arithmetic, and the caveats of this run.

THE NUMERIC CONTRACT - this matters more than anything else below
- Quote figures exactly as they appear in DATA, character for character, including
  their units. Never compute, round, rescale, convert or combine numbers.
- Every number you write is checked against DATA by a program. A summary
  containing a number that is not in DATA is discarded.
- The stance of each view is the pipeline's. Report it as given; never change it,
  upgrade or soften it, or add a recommendation of your own.
- Attribute everything to the model: "the scorecard leans ...", "the composite
  is ...". These are signals from a model, not forecasts and not advice.

WHAT TO WRITE
- headline: one or two sentences naming the strongest views and what drives them.
- views: one entry per view in DATA, in DATA's order, as {"id": ..., "read": ...}.
  "read" is at most two sentences: which signals drive the score, where they
  disagree (value against trend is the usual case), and whether the band says
  act or hold.
- classes: one or two sentences on the ten-year anchor - which classes offer the
  most and the least expected excess return per unit of risk.
- timing: one or two sentences on the macro regime and on trend against value.
- risks: a JSON array of one to three strings, each a single sentence: what would
  change the picture.
- caveats: a JSON array of one to three strings, each a single sentence: the
  caveats in DATA that matter most for reading this run, shortened.

Plain English. No markdown, no headings, no bullet characters inside strings.
Answer with one JSON object with exactly these keys: headline, views, classes,
timing, risks, caveats.\
"""


class SummaryError(RuntimeError):
    """The model's answer could not be used. `unverified` lists numbers it
    quoted that are not in the data, when that was the reason."""

    def __init__(self, message: str, unverified: list[str] | None = None):
        super().__init__(message)
        self.unverified = unverified or []


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------

def _pct(x, digits: int = 1, signed: bool = False) -> str | None:
    if x is None:
        return None
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def _z(x) -> str | None:
    return None if x is None else f"{x:+.2f}"


def _bp(x) -> str | None:
    return None if x is None else f"{x * 1e4:.0f} bp"


def _ordinal(p) -> str | None:
    """A share of history as an ordinal percentile, floored at the 1st."""
    if p is None:
        return None
    n = max(1, min(100, round(p * 100)))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def payload(run: dict) -> dict:
    """What the model is shown: the run's figures, formatted as the page prints
    them, licensed levels left out."""
    macro = run.get("macro") or {}
    views = []
    for v in run["views"]:
        s = v["signals"]
        val, car, tr, mac = (s.get(k) or {} for k in ("value", "carry", "trend", "macro"))
        anchor = v.get("anchor") or {}
        item = {
            "id": v["id"], "label": v["label"], "long": v["long"], "short": v["short"] or "cash",
            "stance": v["stance"], "score": _z(v["score"]),
            "value_z": _z(val.get("z")), "carry_z": _z(car.get("z")),
            "trend_z": _z(tr.get("z")), "macro_z": _z(mac.get("z")),
            "value_measure": val.get("label"),
            "value_percentile": _ordinal(val.get("pct")),
            "value_history_from": val.get("from"),
            "trend_return_12_1": _pct(tr.get("ret"), 1, signed=True),
            "above_10m_average": tr.get("above_sma"),
            "score_change_1m": _z(v.get("change")),
            "band": "act" if v.get("act") else "hold",
            "tilt": _pct(v.get("tilt"), 1, signed=True),
            "tracking_error": _pct(v.get("te"), 2),
            "anchor_10y_excess": _pct(anchor.get("pair_excess"), 1, signed=True),
            "anchor_per_risk": _z(anchor.get("per_risk")),
            "note": v.get("note"),
        }
        # A level from a licensed series stays out; its z and percentile are derived.
        if val and val.get("licence") != "barclays":
            item["value_level"] = _pct(val.get("raw"), 2)
        views.append({k: x for k, x in item.items() if x is not None})

    classes = [c for c in run.get("classes", []) if c.get("per_risk") is not None]
    def cls(c):
        return {"class": c["label"], "fund": c["fund"], "expected_10y": _pct(c["expected"], 1),
                "excess_10y": _pct(c["excess"], 1, signed=True), "per_risk": _z(c["per_risk"]),
                "trend_z": _z(c.get("trend_z"))}

    credit = run.get("credit") or {}
    hy, low, ig = credit.get("hy") or {}, credit.get("hy_low_pd") or {}, credit.get("ig") or {}
    vix = macro.get("vix") or {}
    out = {
        "data_as_of": run.get("data_as_of"),
        "ten_year_anchor_run": (run.get("forecast_run") or {}).get("as_of"),
        "macro": {"regime": macro.get("regime"),
                  "growth_z": _z((macro.get("growth") or {}).get("z")),
                  "inflation_z": _z((macro.get("inflation") or {}).get("z")),
                  "vix": None if vix.get("level") is None else f"{vix['level']:.1f}",
                  "vix_z": _z(vix.get("z"))},
        "views": views,
        "most_attractive_10y": [cls(c) for c in classes[:4]],
        "least_attractive_10y": [cls(c) for c in classes[-3:]],
        "credit": {k: x for k, x in {
            "as_of": credit.get("as_of"), "hy_oas": _bp(credit.get("hy_oas")),
            "ig_oas": _bp(credit.get("ig_oas")),
            "hy_oas_change_3m": _pct(credit.get("hy_oas_change_3m"), 0, signed=True),
            "hy_default_rate": _pct(hy.get("pd"), 1), "hy_excess_after_loss": _bp(hy.get("excess")),
            "hy_cushion_per_year": _bp(hy.get("cushion")),
            "hy_default_rate_low": _pct(low.get("pd"), 1),
            "hy_excess_after_loss_low": _bp(low.get("excess")),
            "ig_excess_after_loss": _bp(ig.get("excess")),
            "ig_cushion_per_year": _bp(ig.get("cushion")),
        }.items() if x is not None},
        "sizing": {"tracking_error": _pct((run.get("sizing") or {}).get("te_total"), 2),
                   "budget": _pct((run.get("sizing") or {}).get("te_budget"), 1)},
        "caveats": run.get("caveats", []),
    }
    return out


def payload_sha(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# the number check
# ---------------------------------------------------------------------------

# A number not glued to the end of a word ("Q3", "T10YIE"), with an optional sign
# and decimal part; thousands separators are allowed.
_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?")


def _norm(token: str) -> str | None:
    t = token.replace("−", "-").lstrip("+-").replace(",", "")
    try:
        d = Decimal(t)
    except InvalidOperation:
        return None
    return format(d.normalize(), "f")


def numbers(text: str) -> list[str]:
    """Every number in a text, normalised: no sign, no thousands separator, no
    trailing zeros ("+2.30%" -> "2.3")."""
    return [n for n in (_norm(m) for m in _NUMBER.findall(text or "")) if n is not None]


def allowed_numbers(data: dict) -> set[str]:
    return set(numbers(json.dumps(data, ensure_ascii=False))) | SMALL_INTEGERS


def _texts(summary: dict) -> list[str]:
    out = [summary.get("headline", ""), summary.get("classes", ""), summary.get("timing", "")]
    out += [v.get("read", "") for v in summary.get("views", [])]
    out += list(summary.get("risks", [])) + list(summary.get("caveats", []))
    return out


def unverified(summary: dict, allowed: set[str]) -> list[str]:
    """The numbers in a summary that the payload does not contain, in order."""
    seen: list[str] = []
    for text in _texts(summary):
        for n in numbers(text):
            if n not in allowed and n not in seen:
                seen.append(n)
    return seen


# ---------------------------------------------------------------------------
# validation and the call
# ---------------------------------------------------------------------------

def _text(x, limit: int) -> str:
    if not isinstance(x, str):
        return ""
    s = re.sub(r"\s+", " ", x).strip()
    return s if len(s) <= limit else s[:limit - 1].rstrip() + "…"


def validate(obj, view_ids: list[str]) -> dict:
    """The model's JSON in the page's shape: views in the payload's order,
    unknown ids dropped, every string capped."""
    if not isinstance(obj, dict):
        raise SummaryError("the reply is not a JSON object")
    reads: dict[str, str] = {}
    for item in obj.get("views") or []:
        if isinstance(item, dict) and item.get("id") in view_ids:
            reads[item["id"]] = _text(item.get("read"), 360)
    headline = _text(obj.get("headline"), 420)
    if not headline:
        raise SummaryError("the reply has no headline")

    def items(key: str) -> list[str]:
        xs = obj.get(key) or []
        xs = xs if isinstance(xs, list) else [xs]
        return [t for t in (_text(x, 240) for x in xs[:3]) if t]

    return {"headline": headline,
            "views": [{"id": i, "read": reads[i]} for i in view_ids if reads.get(i)],
            "classes": _text(obj.get("classes"), 420), "timing": _text(obj.get("timing"), 420),
            "risks": items("risks"), "caveats": items("caveats")}


async def summarise(run: dict, model: str | None = None) -> dict:
    """Ask DeepSeek for the summary of one run, check it, and return it with the
    usage. Raises SummaryError when the answer fails the checks twice, and
    providers.LLMError when the call itself fails."""
    model = model or providers.DEFAULT_MODEL
    data = payload(run)
    ids = [v["id"] for v in data["views"]]
    allowed = allowed_numbers(data)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "DATA\n```json\n"
                                    + json.dumps(data, indent=1, ensure_ascii=False) + "\n```"},
    ]
    usage = providers.Usage()
    problem, bad = "", []
    for attempt in (1, 2):
        content, finish, u = await providers.complete(messages, model=model, max_tokens=MAX_TOKENS,
                                                      temperature=TEMPERATURE, json_mode=True)
        usage.prompt_tokens += u.prompt_tokens
        usage.completion_tokens += u.completion_tokens
        usage.cached_tokens += u.cached_tokens
        content = providers.sanitize(content or "").strip()
        if not content:
            raise SummaryError(
                "The model returned no text"
                + (" - it ran out of output budget, which the reasoning tiers do; set "
                   "DEEPSEEK_MODEL to a non-reasoning model." if finish == "length" else "."))
        try:
            summary = validate(json.loads(content), ids)
        except (json.JSONDecodeError, SummaryError) as exc:
            problem, bad = f"The reply could not be used: {exc}.", []
        else:
            bad = unverified(summary, allowed)
            if not bad:
                return {"content": summary, "attempts": attempt, "model": model,
                        "payload_sha": payload_sha(data),
                        "usage": {"prompt_tokens": usage.prompt_tokens,
                                  "cached_tokens": usage.cached_tokens,
                                  "completion_tokens": usage.completion_tokens}}
            problem = ("These numbers in the reply are not in DATA: " + ", ".join(bad)
                       + ". Every number must be quoted from DATA verbatim.")
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": problem + " Write the JSON object again, quoting "
                      "only numbers that appear in DATA, or leaving them out."}]
    raise SummaryError(f"Refused after two attempts. {problem}", bad)
