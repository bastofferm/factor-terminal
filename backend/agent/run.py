"""Run the model agent for one or more securities.

    python -m backend.agent.run --instrument US:AAPL [--horizon 21] [--rounds 5]
                                [--max-candidates 40] [--no-llm] [--dry-run]
                                [--no-pool]

Each run searches the specification space on a validation span, takes the
simplest model in the confidence set to a held-out test span once, and records
a verdict. With DEEPSEEK_API_KEY set the language model drives the search; with
--no-llm, or without a key, the rule-based critic does. --dry-run reads from the
database but writes nothing. Unless --no-pool, the seed also carries what earlier
runs chose for this security and for securities of the same kind.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from backend.agent import data as ad
from backend.agent import graph as ag
from backend.agent import store
from backend.pipeline.dbsync import connect, etl_run, mark_item_done, run_failed

JOB = "model_agent"


def make_proposer(use_llm: bool):
    if not use_llm:
        return None
    from backend.app.ai import providers
    if not providers.has_api_key():
        return None
    from backend.agent.llm import DeepSeekProposer
    return DeepSeekProposer()


def run_one(instrument_id: str, horizon: int = 21, budget: ag.Budget | None = None,
            use_llm: bool = True, run_id: str | None = None, write: bool = True,
            quiet: bool = False, pool: bool = True, objective: str = "daily") -> dict:
    """Search, test and (unless `write` is false) store one security's model."""
    budget = budget or ag.Budget()
    run_id = run_id or str(uuid.uuid4())
    if write:
        store.create_run(run_id, instrument_id, horizon,
                         {**budget.__dict__, "objective": objective})
    try:
        with connect() as conn, conn.cursor() as cur:
            data = ad.load(cur, instrument_id)
        pooled = []
        if pool and budget.max_pooled > 0:
            # Classified on the data before the test span, as the profile is.
            from backend.agent.evaluate import make_split
            try:
                template = ad.classify(data, make_split(data.y.index, horizon).test_start)
            except ValueError:
                template = None
            pooled = store.load_pool(instrument_id, horizon, template, budget.max_pooled)
        proposer = make_proposer(use_llm)
        out = ag.run(data, horizon=horizon, budget=budget, proposer=proposer,
                     run_id=run_id, sink=store.DbSink() if write else None,
                     persist=store.persist if write else None, pool=pooled,
                     objective=objective)
        st = out["state"]
        if write and not out["runtime"].final:
            store.finish_without_model(run_id, st, out["runtime"].profile,
                                       out["runtime"].llm_model)
        if not quiet:
            _print(out)
        return out
    except Exception as exc:
        if write:
            store.fail_run(run_id, f"{type(exc).__name__}: {exc}")
        raise


def _print(out: dict) -> None:
    for s in out["steps"]:
        print(f"  [{s['node']:10s}] {s['message']}")
    st = out["state"]
    print(f"\n  verdict: {st.get('verdict')}  chosen: {st.get('chosen')}  "
          f"candidates: {len(st.get('ledger') or [])}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the model agent")
    ap.add_argument("--instrument", action="append", required=True)
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--per-round", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=1800.0)
    ap.add_argument("--no-llm", action="store_true", help="rule-based critic only")
    ap.add_argument("--dry-run", action="store_true", help="write nothing to the database")
    ap.add_argument("--no-pool", action="store_true",
                    help="do not seed with candidates from earlier runs")
    ap.add_argument("--objective", choices=("daily", "horizon", "both"), default="daily",
                    help="what the confidence set is computed on: the next day's "
                         "forecast, the h-day forecast, or both")
    args = ap.parse_args()

    budget = ag.Budget(max_rounds=args.rounds, max_candidates=args.max_candidates,
                       per_round=args.per_round, seconds=args.seconds)
    if args.dry_run:
        for inst in args.instrument:
            run_one(inst, args.horizon, budget, not args.no_llm, write=False,
                    pool=not args.no_pool, objective=args.objective)
        return 0

    with etl_run(JOB, scope={"n_instruments": len(args.instrument),
                             "horizon": args.horizon, "llm": not args.no_llm}) as run_id:
        for inst in args.instrument:
            try:
                out = run_one(inst, args.horizon, budget, not args.no_llm,
                              pool=not args.no_pool, objective=args.objective)
                mark_item_done(run_id, JOB, inst, "succeeded",
                               rows_out=len(out["state"].get("ledger") or []))
            except Exception as exc:
                mark_item_done(run_id, JOB, inst, "failed", error=str(exc))
                print(f"  {inst:12s} FAILED: {exc}", file=sys.stderr)
    return 1 if run_failed(run_id) else 0


if __name__ == "__main__":
    raise SystemExit(main())
