"""Where an agent run is written: its trace as it happens, and its result at the end.

The trace and the candidate ledger are written row by row while the search runs,
so a page polling the run sees each step as it is taken. The chosen model is
stored through the pipeline's own writers - the spec, its loadings and beta
covariances, its forecasts and backtests - so every other page reads it exactly
as it reads a specification estimated by hand.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from backend.pipeline import run_estimation as est
from backend.pipeline import run_risk as rr
from backend.pipeline.dbsync import connect


def _json(v: Any) -> str:
    def default(o):
        if isinstance(o, (np.floating,)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (set, tuple)):
            return list(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    def clean(o):
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(x) for k, x in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        return o

    return json.dumps(clean(v), default=default)


def create_run(run_id: str, instrument_id: str, horizon: int, budget: dict) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_run (run_id, instrument_id, horizon_days, status, budget) "
            "VALUES (%s, %s, %s, 'running', %s) ON CONFLICT (run_id) DO NOTHING",
            (run_id, instrument_id, horizon, _json(budget)))


def fail_run(run_id: str, error: str) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE agent_run SET status='failed', finished_at=now(), error=%s "
                    "WHERE run_id=%s", (error[:2000], run_id))


class DbSink:
    """Writes the trace and the ledger as the graph produces them."""

    def step(self, run_id: str, row: dict) -> None:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_step (run_id, seq, node, message, payload) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (run_id, seq) DO NOTHING",
                (run_id, row["seq"], row["node"], row["message"], _json(row.get("payload"))))

    def candidate(self, run_id: str, row: dict) -> None:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_candidate
                    (run_id, candidate_id, round, origin, estimation, risk, spec_hash,
                     metrics, complexity, status, note)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, candidate_id) DO NOTHING
                """,
                (run_id, row["id"], row["round"], row["origin"] or "seed",
                 _json(row["estimation"]), _json(row["risk"]), row["key"],
                 _json(row["metrics"]), row["complexity"],
                 "failed" if row.get("error") else "scored",
                 row.get("error") or row.get("why")))

    def confidence_set(self, run_id: str, included: dict[str, bool],
                       p: dict[str, float], extra: dict[str, dict] | None = None) -> None:
        with connect(autocommit=True) as conn, conn.cursor() as cur:
            for cid, inc in included.items():
                cur.execute(
                    "UPDATE agent_candidate SET in_mcs=%s, mcs_p=%s, "
                    "metrics = COALESCE(metrics, '{}'::jsonb) || %s::jsonb "
                    "WHERE run_id=%s AND candidate_id=%s",
                    (inc, p.get(cid), _json((extra or {}).get(cid) or {}),
                     run_id, cid))


def persist(rt, state: dict) -> None:
    """Store the chosen model and close the run."""
    full = rt.final["full"]
    test = rt.final["test"]
    cand = rt.final["candidate"]
    spec = full["spec"]
    risk = full["risk"]
    inst = rt.data.instrument_id
    run_id = rt.recorder.run_id
    split = rt.split.as_dict()

    from backend.pipeline import run_risk as _rr
    test_bt = _rr.backtest(rt.data.y, full["fc"], rt.horizon, risk,
                           start=rt.split.test[0], end=rt.split.test[-1])

    with connect() as conn, conn.cursor() as cur:
        est.upsert_spec(cur, spec)
        est.write_results(cur, spec.spec_id, inst, full["loadings"], full["meta"],
                          full["covs"])
        rr.write(cur, spec.spec_id, inst, rt.horizon, risk.label(), full["fc"],
                 [("full", full["backtest"]), ("test", test_bt)],
                 in_selection_before=rt.split.test_start)
        cur.execute(
            """
            UPDATE agent_run SET status='succeeded', finished_at=now(), profile=%s,
                validation_start=%s, validation_end=%s, test_start=%s, test_end=%s,
                chosen_candidate=%s, spec_id=%s, cov_method=%s, risk_spec=%s,
                verdict=%s, gates=%s, test_metrics=%s, llm_model=%s
            WHERE run_id=%s
            """,
            (_json(rt.profile), split["validation_start"], split["validation_end"],
             split["test_start"], split["test_end"], state.get("chosen"), spec.spec_id,
             risk.label(), _json({"estimation": cand.estimation.model_dump(mode="json"),
                                  "risk": cand.risk.model_dump(mode="json")}),
             state.get("verdict"), _json(state.get("gates")),
             _json(test.metrics), rt.llm_model, run_id))


# ---------------------------------------------------------------------------
# the pool: what earlier runs found
# ---------------------------------------------------------------------------

def pool_candidates(own_chosen: list[tuple[dict, dict]], peer_chosen: list[tuple[str, dict, dict]],
                    own_members: list[tuple[dict, dict, float | None]],
                    limit: int) -> list[tuple[Any, str]]:
    """Earlier findings as seeds, best evidence first.

    On a single stock the confidence set is wide, so which near-equals happen to
    be proposed decides the winner; two AAPL runs chose different members. Seeding
    each run with what earlier runs found narrows that: the security's own earlier
    choices, then the choices made for securities of the same kind, then the rest
    of its own earlier confidence sets by loss. Every one is re-scored on this
    run's validation span - nothing is carried over but the specification.

    The lead-lag count and the h-day reversion are dropped: the search does not
    choose them, the stage after it does.
    """
    from backend.agent import space as sp

    out: list[tuple[Any, str]] = []
    seen: set[str] = set()

    def take(est_json: dict, risk_json: dict, why: str) -> None:
        if len(out) >= limit:
            return
        try:
            risk = dict(risk_json or {})
            risk.pop("nw_lags", None)
            risk.pop("revert_halflife", None)
            c = sp.Candidate(estimation=sp.EstimationChoice(**(est_json or {})),
                             risk=sp.RiskChoice(**risk))
        except Exception:
            return     # written under an older search space
        if c.key() in seen:
            return
        seen.add(c.key())
        out.append((c, why))

    for e, r in own_chosen:
        take(e, r, "chosen for this security by an earlier run")
    for inst, e, r in peer_chosen:
        take(e, r, f"chosen for {inst}, a security of the same kind")
    for e, r, _ in sorted(own_members, key=lambda m: (m[2] is None, m[2] or 0.0)):
        take(e, r, "in this security's confidence set in an earlier run")
    return out


def load_pool(instrument_id: str, horizon: int, template: str | None,
              limit: int = 8, peers: int = 6) -> list[tuple[Any, str]]:
    """Read the pool for one security from the agent tables."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT risk_spec -> 'estimation', risk_spec -> 'risk'
            FROM agent_run
            WHERE instrument_id = %s AND horizon_days = %s AND status = 'succeeded'
              AND risk_spec IS NOT NULL
            ORDER BY started_at DESC LIMIT 4
            """, (instrument_id, horizon))
        own_chosen = [(e, r) for e, r in cur.fetchall()]
        peer_chosen: list = []
        if template:
            cur.execute(
                """
                SELECT DISTINCT ON (instrument_id) instrument_id,
                       risk_spec -> 'estimation', risk_spec -> 'risk'
                FROM agent_run
                WHERE instrument_id <> %s AND horizon_days = %s AND status = 'succeeded'
                  AND risk_spec IS NOT NULL AND profile ->> 'template' = %s
                ORDER BY instrument_id, started_at DESC
                """, (instrument_id, horizon, template))
            peer_chosen = cur.fetchall()[:peers]
        cur.execute(
            """
            SELECT c.estimation, c.risk, (c.metrics ->> 'common_loss')::float
            FROM agent_candidate c JOIN agent_run r USING (run_id)
            WHERE r.instrument_id = %s AND r.horizon_days = %s AND r.status = 'succeeded'
              AND c.in_mcs
            ORDER BY r.started_at DESC LIMIT 60
            """, (instrument_id, horizon))
        own_members = cur.fetchall()
    return pool_candidates(own_chosen, peer_chosen, own_members, limit)


def finish_without_model(run_id: str, state: dict, profile: dict, llm_model: str | None) -> None:
    """Close a run whose search produced nothing to store."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE agent_run SET status='succeeded', finished_at=now(), profile=%s, "
            "verdict=%s, gates=%s, test_metrics=%s, llm_model=%s WHERE run_id=%s",
            (_json(profile), state.get("verdict"), _json(state.get("gates")),
             _json(state.get("test_metrics")), llm_model, run_id))
