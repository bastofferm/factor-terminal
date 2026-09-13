"""Find out which DeepSeek model actually answers with the real corpus in the prompt.

This exists because of a documented failure mode. The provider registry in
AI_Analyst carries a comment that the v4 reasoning tiers spend the entire token
budget on hidden reasoning when the prompt is large, and return empty content with
finish_reason="length". This project sends a ~5,000-token methodology corpus on
every request, which is exactly that situation.

Rather than pick a model from the documentation and discover the problem through an
empty chat bubble, send the genuine prompt to each candidate and check.

    python -m scripts.probe_deepseek
"""

from __future__ import annotations

import asyncio
import sys
import time

from backend.app.ai import corpus, providers

CANDIDATES = [
    "deepseek-chat",
    "deepseek-flash",
    "deepseek-v4-pro",
]

# A question that cannot be answered from general knowledge — it can only come from
# the corpus — so an answer proves grounding, not recall.
QUESTION = (
    "In two sentences: why does the commodity block use total-return ETFs rather "
    "than the =F continuous futures series?"
)

EXPECTED_TERMS = ("roll", "collateral", "etf")


async def probe(model: str, corpus_text: str) -> dict:
    messages = [
        {"role": "system", "content": "You answer questions about a factor model "
                                      "using only the reference provided."},
        {"role": "system", "content": corpus_text},
        {"role": "user", "content": QUESTION},
    ]
    started = time.monotonic()
    try:
        content, finish, usage = await providers.complete(
            messages, model=model, max_tokens=600)
    except providers.LLMError as exc:
        return {"model": model, "ok": False, "error": str(exc)[:160]}

    elapsed = time.monotonic() - started
    text = providers.sanitize(content)
    lowered = text.lower()
    return {
        "model": model,
        "ok": bool(text.strip()),
        "grounded": sum(t in lowered for t in EXPECTED_TERMS),
        "finish": finish,
        "chars": len(text),
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": usage.cached_tokens,
        "completion_tokens": usage.completion_tokens,
        "seconds": round(elapsed, 1),
        "preview": " ".join(text.split())[:150],
    }


async def main() -> int:
    if not providers.has_api_key():
        print("No DeepSeek API key found.", file=sys.stderr)
        return 1

    corpus_text = corpus.build()
    info = corpus.info()
    print(f"corpus: {info['chars']:,} chars, ~{info['approx_tokens']:,} tokens\n")

    results = []
    for model in CANDIDATES:
        print(f"  probing {model} ...", flush=True)
        results.append(await probe(model, corpus_text))

    print()
    usable = []
    for r in results:
        if not r["ok"]:
            reason = r.get("error") or f"empty content, finish_reason={r.get('finish')!r}"
            print(f"  {r['model']:18s} UNUSABLE  {reason}")
            continue
        usable.append(r)
        print(f"  {r['model']:18s} ok  {r['chars']:4d} chars  "
              f"{r['grounded']}/{len(EXPECTED_TERMS)} key terms  "
              f"{r['seconds']:5.1f}s  "
              f"prompt {r['prompt_tokens']:,} (cached {r['cached_tokens']:,})  "
              f"out {r['completion_tokens']}")
        print(f"  {'':18s}    {r['preview']}")

    print()
    if not usable:
        print("No candidate returned content. The chat feature cannot ship on any "
              "of these models with a corpus this size.", file=sys.stderr)
        return 1

    # Prefer a grounded answer, then a fast one.
    best = sorted(usable, key=lambda r: (-r["grounded"], r["seconds"]))[0]
    print(f"Recommended: DEEPSEEK_MODEL={best['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
