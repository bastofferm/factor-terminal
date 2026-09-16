"""DeepSeek client.

Trimmed from `AI_Analyst/backend/llm_providers.py`, which solves three problems
worth keeping:

  * a host allowlist, so a misconfigured base URL cannot post the API key to an
    arbitrary server,
  * key resolution from the Windows registry as well as the process environment, so
    a key set through System Properties takes effect without restarting uvicorn,
  * redaction of the key from every error string before it can reach a log or an
    HTTP response.

The API is OpenAI-compatible, so this is plain httpx against /chat/completions —
no SDK.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx

BASE_URL = "https://api.deepseek.com/v1"
ALLOWED_HOSTS = frozenset({"api.deepseek.com"})
ENV_KEYS = ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY")

# Overridable because the right answer is empirical, not documented: the reasoning
# tiers can spend their whole output budget on hidden reasoning and return empty
# content on a large prompt, which is this application's normal case. Run
# scripts/probe_deepseek.py against the real corpus before changing it.
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

REQUEST_TIMEOUT = 120.0


class LLMError(RuntimeError):
    """Anything that stops us returning an answer. Never carries the API key."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


def resolve_api_key() -> tuple[str, str]:
    """Return (key, where it came from). Raises if there is none.

    The registry fallback exists because on Windows a key set through System
    Properties is not in an already-running process's environment.
    """
    for name in ENV_KEYS:
        value = os.environ.get(name)
        if value:
            return value, f"env:{name}"

    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in ENV_KEYS:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except OSError:
                    continue
                if value:
                    return str(value), f"registry:{name}"
    except Exception:
        pass

    raise LLMError(
        "No DeepSeek API key found. Set DEEPSEEK_API_KEY in the environment "
        "or in the user environment variables."
    )


def has_api_key() -> bool:
    try:
        resolve_api_key()
        return True
    except LLMError:
        return False


def _endpoint(base_url: str | None = None) -> str:
    url = (base_url or BASE_URL).rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in ALLOWED_HOSTS:
        raise LLMError(
            f"Refusing to send the API key to {url!r}; only "
            f"https://{'/'.join(sorted(ALLOWED_HOSTS))} is allowed."
        )
    return f"{url}/chat/completions"


def _redact(text: str, api_key: str) -> str:
    return text.replace(api_key, "***") if api_key else text


def _payload(messages: list[dict], model: str, max_tokens: int,
             temperature: float, stream: bool) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


async def complete(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 1600,
    temperature: float = 0.2,
    timeout: float = REQUEST_TIMEOUT,
) -> tuple[str, str, Usage]:
    """One blocking completion. Returns (content, finish_reason, usage).

    Used by the probe script and the tests; the chat endpoint streams instead.
    """
    api_key, _ = resolve_api_key()
    model = model or DEFAULT_MODEL
    payload = _payload(messages, model, max_tokens, temperature, stream=False)

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            _endpoint(),
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code != 200:
        raise LLMError(f"DeepSeek returned {r.status_code}: "
                       f"{_redact(r.text[:500], api_key)}")

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason") or ""
    u = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(u.get("prompt_tokens") or 0),
        completion_tokens=int(u.get("completion_tokens") or 0),
        cached_tokens=int(u.get("prompt_cache_hit_tokens") or 0),
    )
    return content, finish, usage


async def stream(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 1600,
    temperature: float = 0.2,
    timeout: float = REQUEST_TIMEOUT,
) -> AsyncIterator[tuple[str, str | None]]:
    """Yield (delta_text, finish_reason) as the answer arrives.

    finish_reason is None until the final chunk. A caller that receives no text at
    all and then finish_reason == "length" has hit the reasoning-tier failure mode:
    the model spent its whole budget on hidden reasoning and returned nothing.
    """
    api_key, _ = resolve_api_key()
    model = model or DEFAULT_MODEL
    payload = _payload(messages, model, max_tokens, temperature, stream=True)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            _endpoint(),
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                raise LLMError(f"DeepSeek returned {r.status_code}: "
                               f"{_redact(body[:500], api_key)}")

            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content") or ""
                finish = choice.get("finish_reason")
                if delta or finish:
                    yield delta, finish


# DeepSeek occasionally emits its tool-call markup as literal text instead of a
# structured tool_calls field. We do not use tools, but stray markup still turns up
# and must not reach the reader.
_MARKUP_MARKERS = ("<|DSML", "<|tool", "|>")


def sanitize(text: str) -> str:
    if not any(m in text for m in _MARKUP_MARKERS):
        return text
    import re

    cleaned = re.sub(r"<\|[^|]*\|?>?", "", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
