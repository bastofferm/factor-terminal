"""Assemble the CANONICAL DATA block: the figures the model is allowed to quote.

Two sources, both deterministic. The frontend posts a snapshot of what it has
actually rendered, and the server adds anything the question names but the snapshot
does not carry. There is no tool loop — the retrieval runs once, before the model
sees the turn, so a question can never trigger an unbounded sequence of queries.

Everything here is read-only against the factors database.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from backend.app import db

# A snapshot is written by our own frontend, but it arrives over HTTP and is pasted
# into a prompt, so it gets the same treatment as any other untrusted input: capped
# in size, and rendered as data rather than interpolated as instructions.
MAX_SNAPSHOT_CHARS = 12_000
MAX_RETRIEVED_FACTORS = 4

# The active page gets the budget above. The other tabs share this one, and no
# single one of them may take more than its slice: seven pages at twelve
# thousand characters each would crowd out the question being asked.
MAX_OTHER_PAGES_CHARS = 12_000
MAX_ONE_OTHER_PAGE_CHARS = 3_000


def _fmt(value: Any) -> Any:
    """Round floats to something a reader would write, leave everything else alone."""
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _fmt(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fmt(v) for v in value[:200]]
    return value


def _as_json(payload: Any, limit: int) -> str:
    text = json.dumps(_fmt(payload), indent=1, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def render_snapshot(snapshot: dict | None) -> str:
    """The figures currently on screen, as a labelled JSON block."""
    if not snapshot:
        return ""
    page = snapshot.get("page") or "unknown"
    body = _as_json({k: v for k, v in snapshot.items() if k != "page"},
                    MAX_SNAPSHOT_CHARS)
    return f"## On screen now — page `{page}`\n\n```json\n{body}\n```"


def render_other_pages(snapshots: list[dict] | None, active_page: str | None) -> str:
    """What the other tabs last drew, each labelled with its page and age.

    Carried so a question asked on one screen can be about a figure seen on
    another, which is most of how anyone actually uses the app. The labels are
    the whole safety property: these numbers were rendered at some earlier
    moment, possibly before a data refresh, and quoting one as though it were on
    screen now would be worse than not having it at all. So each block says
    which page it came from and when, and the prompt is told to say so too.
    """
    if not snapshots:
        return ""

    now = datetime.now(timezone.utc)
    blocks: list[str] = []
    spent = 0
    for snap in snapshots:
        page = snap.get("page") or "unknown"
        if not page or page == active_page:
            continue
        data = snap.get("data") or {}
        if not data:
            continue
        if spent >= MAX_OTHER_PAGES_CHARS:
            blocks.append(f"- `{page}`: rendered, omitted for space")
            continue

        body = _as_json(data, min(MAX_ONE_OTHER_PAGE_CHARS,
                                  MAX_OTHER_PAGES_CHARS - spent))
        spent += len(body)
        blocks.append(f"### `{page}` — {_age(snap.get('captured_at'), now)}\n\n"
                      f"```json\n{body}\n```")

    if not blocks:
        return ""
    return ("## Other tabs, as they were last drawn\n\n"
            "These are not on screen now. Each was rendered at the time shown "
            "and may predate a refresh, so name the page and say when it was "
            "drawn before quoting anything from here.\n\n" + "\n\n".join(blocks))


def _age(captured_at: Any, now: datetime) -> str:
    """How long ago, in words. Unknown stays unknown rather than becoming now."""
    if not captured_at:
        return "time unknown"
    try:
        at = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
    except ValueError:
        return "time unknown"
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    seconds = max(0, int((now - at).total_seconds()))
    if seconds < 90:
        return "seconds ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    if seconds < 86_400:
        return f"{seconds // 3600} h ago"
    return f"{seconds // 86_400} d ago"


async def known_factor_ids() -> list[str]:
    rows = await db.fetch("SELECT factor_id FROM ref_factor WHERE is_active")
    return [r["factor_id"] for r in rows]


def mentioned_factors(question: str, known: list[str]) -> list[str]:
    """Factor ids named in the question.

    Matched on word boundaries against the actual registry rather than by pattern,
    so 'eq_global' matches and 'equity' does not silently pull in seven factors.
    """
    lowered = question.lower()
    hits = [fid for fid in known
            if re.search(rf"\b{re.escape(fid.lower())}\b", lowered)]
    return hits[:MAX_RETRIEVED_FACTORS]


async def factor_brief(factor_id: str) -> str:
    """Construction, coverage and latest diagnostic verdict for one factor."""
    row = await db.fetchrow(
        """
        SELECT f.factor_id, f.name, f.block_id, f.construction,
               f.orthogonalize_against, f.hierarchy_level,
               c.first_date, c.last_date, c.n_obs, c.sd_ann, c.mean_ann
        FROM ref_factor f
        LEFT JOIN (
            SELECT factor_id, min(date) AS first_date, max(date) AS last_date,
                   count(*) AS n_obs,
                   stddev_samp(ret_orth) * sqrt(252) AS sd_ann,
                   avg(ret_orth) * 252 AS mean_ann
            FROM fact_factor_return WHERE ret_orth IS NOT NULL GROUP BY 1
        ) c USING (factor_id)
        WHERE f.factor_id = $1
        """,
        factor_id,
    )
    if not row:
        return ""

    diag = await db.fetchrow(
        """
        SELECT verdict, verdict_reason, flags, window_days, n_obs,
               adf_p, kpss_p, zero_return_share, excess_kurtosis
        FROM fact_series_diagnostics
        WHERE series_key = $1 AND series_type = 'factor'
        ORDER BY window_days DESC, as_of_date DESC LIMIT 1
        """,
        factor_id,
    )

    construction = row["construction"]
    if isinstance(construction, str):
        try:
            construction = json.loads(construction)
        except json.JSONDecodeError:
            construction = {}
    construction = construction or {}

    lines = [
        f"### Factor `{row['factor_id']}` — {row['name']}",
        f"- block: {row['block_id']} (hierarchy level {row['hierarchy_level']})",
        f"- method: `{construction.get('method')}`, "
        f"inputs: {json.dumps(construction.get('inputs') or {}, ensure_ascii=False)}",
    ]
    if row["orthogonalize_against"]:
        lines.append("- orthogonalised against: "
                     + ", ".join(row["orthogonalize_against"]))
    if construction.get("note"):
        lines.append(f"- note recorded at definition: {construction['note']}")
    if row["n_obs"]:
        lines.append(
            f"- coverage: {row['n_obs']:,} observations, "
            f"{row['first_date']} to {row['last_date']}")
        if row["sd_ann"] is not None:
            lines.append(f"- annualised volatility: {row['sd_ann'] * 100:.2f}%")
        if row["mean_ann"] is not None:
            lines.append(f"- annualised mean excess return: {row['mean_ann'] * 100:.2f}%")
    if diag:
        window = "full history" if diag["window_days"] == 0 else f"{diag['window_days']}d window"
        lines.append(f"- stationarity verdict ({window}): **{diag['verdict']}**"
                     + (f" — {diag['verdict_reason']}" if diag["verdict_reason"] else ""))
        if diag["flags"]:
            lines.append(f"- flags: {', '.join(diag['flags'])}")
    return "\n".join(lines)


async def data_freshness() -> str:
    """How current the model is. Relevant to almost any question about a number."""
    row = await db.fetchrow(
        """
        SELECT max(date) AS as_of,
               count(DISTINCT factor_id) AS n_factors
        FROM fact_factor_return WHERE ret_orth IS NOT NULL
        """
    )
    if not row or not row["as_of"]:
        return ""
    dead = await db.fetchval(
        "SELECT count(*) FROM ref_instrument "
        "WHERE role IN ('factor_input','both') AND NOT is_live"
    )
    out = [f"### Model state\n- factor panel runs to {row['as_of']}, "
           f"{row['n_factors']} active factors"]
    if dead:
        out.append(f"- {dead} factor input(s) flagged as no longer live")
    return "\n".join(out)


async def build(question: str, snapshot: dict | None,
                snapshots: list[dict] | None = None) -> tuple[str, list[str]]:
    """Return (canonical block, list of what it contains) for one turn.

    The second element drives the context chip in the UI, so the analyst can see
    what the answer was grounded on rather than having to trust it. Other tabs
    appear in it too: a reader who is told the answer came from four pages needs
    to know which four.
    """
    parts: list[str] = []
    sources: list[str] = []

    state = await data_freshness()
    if state:
        parts.append(state)

    active_page = str(snapshot.get("page") or "") if snapshot else ""
    rendered = render_snapshot(snapshot)
    if rendered:
        parts.append(rendered)
        sources.append(f"page {active_page or '?'}")

    others = render_other_pages(snapshots, active_page)
    if others:
        parts.append(others)
        sources.extend(
            f"{s.get('page')} (other tab)" for s in (snapshots or [])
            if s.get("page") and s.get("page") != active_page and s.get("data"))

    # Anything the question names that the snapshot does not already carry.
    on_screen = str(snapshot.get("factor_id") or "") if snapshot else ""
    try:
        known = await known_factor_ids()
    except Exception:
        known = []
    for fid in mentioned_factors(question, known):
        brief = await factor_brief(fid)
        if brief:
            parts.append(brief)
            sources.append(fid if fid != on_screen else f"{fid} (on screen)")

    if not parts:
        return "", sources
    return "\n\n".join(parts), sources
