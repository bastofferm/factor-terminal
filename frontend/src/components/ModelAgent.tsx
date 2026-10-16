"use client";

/**
 * The model agent's panel: start a search, watch it, read the verdict.
 *
 * The agent replaces the hand-set estimation controls. It fits candidate
 * specifications on a validation span - a language model proposing which to try,
 * code fitting and scoring every one - takes the simplest model in the confidence
 * set to a held-out test span once, and gives a verdict. This panel is the whole
 * of that made visible: the trace as it happens, every candidate it tried with
 * the numbers it was judged on, and the verdict with the reason for each gate.
 */

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { Panel } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { MetricCard } from "@/components/MetricCard";
import { AGENT_METRICS as M, GATE_METRIC } from "@/lib/agentMetrics";
import { AgentCandidate, AgentObjective, AgentRunDetail, api, num, pct, pval } from "@/lib/api";
import { agent as agentVerdict, backtest } from "@/lib/verdict";

const VERDICT_LABEL: Record<string, string> = {
  PASS: "Pass",
  PASS_WITH_CAVEATS: "Pass with caveats",
  FAIL: "Fail",
  INSUFFICIENT_DATA: "Insufficient data",
};

/** One line per candidate, in the terms the agent's own trace uses. */
export function candidateSummary(c: Pick<AgentCandidate, "estimation" | "risk">): string {
  const e = c.estimation ?? {}, r = c.risk ?? {};
  const fs = Array.isArray(e.factor_set) ? `${e.factor_set.length} factors` : e.factor_set;
  const bits = [e.panel, fs, `${e.window_days}d`, e.estimator];
  if (e.ridge_lambda) bits.push(`λ=${e.ridge_lambda}`);
  if (e.weighting === "ewma") bits.push(`hl${e.ewma_halflife}`);
  if (e.dimson_lags) bits.push(`dimson${e.dimson_lags}:${e.dimson_set}`);
  if (e.hac_lags !== undefined && e.hac_lags !== "auto") bits.push(`hac${e.hac_lags}`);
  if (e.winsorize) bits.push("winsor");
  if (e.sector === "own") bits.push("+sector");
  const risk = [`${r.cov_method}/${r.halflife}/${r.cov_window}`];
  if (r.cov_method === "blend") risk.push(`a=${r.ewma_weight}`);
  if (r.specific === "ewma") risk.push(`spec-ewma${r.specific_halflife}`);
  if (r.debias) risk.push("debias");
  if (r.refresh === "daily") risk.push("daily");
  if (r.var_dist === "t") risk.push("t-VaR");
  if (r.nw_lags) risk.push(`lead-lag${r.nw_lags}`);
  if (r.revert_halflife) risk.push(`revert${r.revert_halflife}`);
  return `${bits.join(" ")} | ${risk.join(" ")}`;
}

// The card typesets its formulas with KaTeX; it loads on the click that opens it.
const CandidateCard = dynamic(
  () => import("@/components/CandidateCard").then((m) => m.CandidateCard),
  { ssr: false });

const OBJECTIVE_LABEL: Record<AgentObjective, string> = {
  daily: "Next day",
  horizon: "Horizon",
  both: "Both",
};

const TONE_TEXT = { good: "text-pass", warn: "text-warn", bad: "text-fail", neutral: "text-navy" };
const STAGE_TEXT: Record<string, string> = {
  chosen: "font-semibold text-pass", eligible: "text-muted", uncalibrated: "text-warn",
  outside: "text-muted", failed: "text-fail",
};
const TONE_DOT = { good: "bg-pass", warn: "bg-warn", bad: "bg-fail", neutral: "bg-line" };

export function ModelAgent({
  instrument,
  onModel,
}: {
  instrument: string;
  /** Called with the chosen spec whenever a finished run is on screen. */
  onModel: (specId: string | null, run: AgentRunDetail | null) => void;
}) {
  const [llm, setLlm] = useState<{ llm_available: boolean; model: string } | null>(null);
  const [useLlm, setUseLlm] = useState(true);
  const [horizon, setHorizon] = useState(21);
  const [rounds, setRounds] = useState(5);
  const [maxCandidates, setMaxCandidates] = useState(40);
  const [objective, setObjective] = useState<AgentObjective>("daily");
  const [detail, setDetail] = useState<AgentRunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const traceEnd = useRef<HTMLDivElement>(null);
  const [card, setCard] = useState<string | null>(null);
  const announced = useRef<string | null>(null);

  useEffect(() => {
    api.agentStatus().then((s) => { setLlm(s); setUseLlm(s.llm_available); })
      .catch(() => setLlm({ llm_available: false, model: "" }));
  }, []);

  // The latest run for the security, whatever its state.
  useEffect(() => {
    setDetail(null); setError(null); announced.current = null;
    api.agentLatest(instrument)
      .then(setDetail)
      .catch(() => { setDetail(null); onModel(null, null); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instrument]);

  const running = detail?.run.status === "running";

  // Poll while running. The trace and the ledger are written as the search goes,
  // so each poll shows progress rather than nothing until the end.
  useEffect(() => {
    if (!running || !detail) return;
    const id = window.setInterval(() => {
      api.agentGet(detail.run.run_id).then(setDetail).catch(() => {});
    }, 2000);
    return () => window.clearInterval(id);
  }, [running, detail?.run.run_id]);

  useEffect(() => {
    if (!running || !detail) return;
    const started = new Date(detail.run.started_at).getTime();
    const id = window.setInterval(
      () => setElapsed(Math.max(0, Math.round((Date.now() - started) / 1000))), 1000);
    return () => window.clearInterval(id);
  }, [running, detail?.run.started_at]);

  // Hand the chosen model to the page once per finished run.
  useEffect(() => {
    if (!detail || detail.run.status === "running") return;
    if (announced.current === detail.run.run_id) return;
    announced.current = detail.run.run_id;
    onModel(detail.run.spec_id, detail);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detail]);

  useEffect(() => {
    if (running) traceEnd.current?.scrollIntoView({ block: "nearest" });
  }, [detail?.steps.length, running]);

  const start = () => {
    setError(null);
    api.agentRun({ instrument_id: instrument, horizon_days: horizon, rounds,
                   max_candidates: maxCandidates, use_llm: useLlm && !!llm?.llm_available,
                   objective })
      .then((r) => api.agentGet(r.run_id))
      .then((d) => { announced.current = null; setDetail(d); })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const run = detail?.run;
  // The objective the shown run was chosen on, not the one set for the next run.
  const runObjective: AgentObjective =
    (detail?.selection?.funnel?.objective ?? run?.budget?.objective ?? "daily") as AgentObjective;
  const usesHorizon = runObjective !== "daily";
  // Ranked on the loss the confidence set used, over the dates every candidate has:
  // each candidate's own loss covers its own days, and a longer covariance window
  // starts later, so those are not comparable.
  const ranked = useMemo(() => {
    const c = [...(detail?.candidates ?? [])];
    const loss = (x: AgentCandidate) => usesHorizon
      ? x.metrics?.common_horizon_loss ?? x.metrics?.horizon_loss ?? Infinity
      : x.metrics?.common_loss ?? x.metrics?.daily_loss ?? Infinity;
    return c.sort((a, b) => loss(a) - loss(b));
  }, [detail?.candidates, usesHorizon]);

  const verdict = agentVerdict.verdict(run?.verdict);
  const tm = run?.test_metrics ?? {};
  const chosen = detail?.candidates.find((c) => c.candidate_id === run?.chosen_candidate);
  const bench = backtest.benchmark(tm.dm_t_own_ewma as number, tm.dm_p_own_ewma as number);

  return (
    <Panel
      title="Model agent"
      caption={
        <>
          Chooses this security&apos;s model instead of you. Candidates are fitted and
          scored on a validation span; {useLlm && llm?.llm_available
            ? <>a language model ({llm.model}) proposes what to try next, </>
            : <>a rule-based critic proposes what to try next, </>}
          code computes every number, and the simplest model the data cannot tell apart
          from the best is taken to a held-out test span the search never saw - once.
        </>
      }
    >
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="label">Horizon</label>
          <select className="field mt-1 w-[110px]" value={horizon} disabled={running}
                  onChange={(e) => setHorizon(Number(e.target.value))}>
            {[5, 21, 63].map((h) => <option key={h} value={h}>{h} days</option>)}
          </select>
        </div>
        <div>
          <label className="label">Refinement rounds</label>
          <input type="number" min={0} max={10} className="field mt-1 w-[90px]" value={rounds}
                 disabled={running} onChange={(e) => setRounds(Number(e.target.value))} />
        </div>
        <div>
          <label className="label">Max candidates</label>
          <input type="number" min={4} max={80} className="field mt-1 w-[90px]"
                 value={maxCandidates} disabled={running}
                 onChange={(e) => setMaxCandidates(Number(e.target.value))} />
        </div>
        <div>
          <label className="label inline-flex items-center">
            Optimise for<InfoTip metric={M.objective} label="Optimise for" />
          </label>
          <select className="field mt-1 w-[150px]" value={objective} disabled={running}
                  onChange={(e) => setObjective(e.target.value as AgentObjective)}>
            <option value="daily">Next day</option>
            <option value="horizon">Horizon ({horizon} days)</option>
            <option value="both">Both</option>
          </select>
        </div>
        <label className="flex items-center gap-2 pb-1.5 text-[12px]"
               title={llm?.llm_available ? "" : "No DeepSeek API key is configured"}>
          <input type="checkbox" checked={useLlm && !!llm?.llm_available}
                 disabled={running || !llm?.llm_available}
                 onChange={(e) => setUseLlm(e.target.checked)} />
          Language model drives the search
        </label>
        <button className="btn ml-auto min-w-[170px]" onClick={start} disabled={running}>
          {running ? `Searching… ${elapsed}s` : run ? "Run the agent again" : "Run the model agent"}
        </button>
      </div>

      {error && (
        <div className="mt-3 rounded border border-fail/30 bg-fail/5 p-2 text-[12px] text-fail">
          {error}
        </div>
      )}
      {run?.status === "failed" && (
        <div className="mt-3 rounded border border-fail/30 bg-fail/5 p-2 text-[12px] text-fail">
          The run failed: {run.error}
        </div>
      )}

      {run && run.status !== "running" && run.verdict && (
        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
          <div>
            <div className="flex items-baseline gap-3">
              <span className={`text-[22px] font-semibold ${TONE_TEXT[verdict.tone]}`}>
                {VERDICT_LABEL[run.verdict] ?? run.verdict}
              </span>
              <span className="text-[11px] text-muted">{verdict.reason}</span>
            </div>
            {chosen && (
              <p className="mt-1 font-mono text-[11px] text-navy">
                {chosen.candidate_id}: {candidateSummary(run.risk_spec ?? chosen)}
              </p>
            )}
            {run.risk_spec?.risk && (
              <p className="mt-1 text-[11px] text-muted">
                {run.risk_spec.risk.nw_lags
                  ? <>The h-day forecast adds {run.risk_spec.risk.nw_lags} lag
                      {run.risk_spec.risk.nw_lags > 1 ? "s" : ""} of cross-day factor
                      covariance, chosen after the search on the validation span&apos;s
                      h-day loss.</>
                  : <>No lead-lag adjustment: on the validation span&apos;s h-day returns
                      no lag count beat none by more than one standard error.</>}
                {run.risk_spec.risk.revert_halflife
                  ? <> It reverts toward the long-run covariance, a shock decaying with
                      a {run.risk_spec.risk.revert_halflife}-day half-life, chosen the
                      same way.</>
                  : <> It does not revert toward the long run: no reversion beat none by
                      more than one standard error.</>}
                {run.profile?.sector_factor && (
                  run.risk_spec.estimation?.sector === "own"
                    ? <> Uses its sector factor, {run.profile.sector_factor}.</>
                    : <> Its sector factor, {run.profile.sector_factor}, was not chosen.</>
                )}
              </p>
            )}
            <p className="mt-1 text-[11px] text-muted">
              Judged on {run.test_start} to {run.test_end}, after the search ended.
              Selection used {run.validation_start} to {run.validation_end}.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-2 overflow-visible sm:grid-cols-4">
              <MetricCard label="Monthly QLIKE (test)" value={num(tm.qlike as number, 3)}
                          sub={`own EWMA ${num(tm.qlike_own_ewma as number, 3)}`}
                          tone={bench.tone} metric={M.qlikeMonthly} />
              <MetricCard label="vs own EWMA, p" value={pval(tm.dm_p_own_ewma as number)}
                          sub={bench.reason} tone={bench.tone} metric={M.dmMonthly} />
              <MetricCard label="Bias (mean ratio)" value={num(tm.mean_bias as number, 2)}
                          sub={backtest.bias(tm.mean_bias as number).reason}
                          tone={backtest.bias(tm.mean_bias as number).tone}
                          metric={M.meanBias} align="right" />
              <MetricCard label="VaR 99% breaches"
                          value={`${tm.exceptions_99 ?? "—"}/${num(tm.expected_99 as number, 0)}`}
                          sub={backtest.kupiec(tm.kupiec_p_99 as number).reason}
                          tone={backtest.kupiec(tm.kupiec_p_99 as number).tone}
                          metric={M.kupiec99} align="right" />
            </div>
          </div>
          <ul className="space-y-1">
            {(run.gates ?? []).map((g) => {
              const a = agentVerdict.gate(g.status);
              return (
                <li key={g.name} className="flex gap-2 text-[11.5px] leading-snug">
                  <span className={`mt-[5px] inline-block h-[7px] w-[7px] shrink-0 rounded-full ${TONE_DOT[a.tone]}`} />
                  <span><b className="font-semibold">{g.name}</b>
                    <InfoTip metric={M[GATE_METRIC[g.name]]} label={g.name} />{" "}
                    <span className="text-muted">— {g.detail}</span></span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {ranked.length > 0 && (
        <div className="mt-4">
          <h3 className="mb-1 text-2xs font-semibold uppercase tracking-label text-muted">
            Candidates, best {usesHorizon ? `${run?.horizon_days ?? horizon}-day` : "daily"}{" "}
            loss first
          </h3>
          {detail?.selection?.funnel && run?.status !== "running" && (
            <p className="mb-1.5 text-[11.5px] leading-snug">
              How the model was chosen:{" "}
              <b>{detail.selection.funnel.tried}</b> candidates tried →{" "}
              <b>{detail.selection.funnel.in_set}</b> not significantly worse than the best
              on {runObjective === "daily" ? "the daily loss"
                : runObjective === "horizon" ? `the ${run?.horizon_days}-day loss`
                : `the daily and the ${run?.horizon_days}-day loss`}
              <InfoTip metric={M.mcs} /> (optimised for:{" "}
              {OBJECTIVE_LABEL[runObjective]}<InfoTip metric={M.objective} />) →{" "}
              {detail.selection.funnel.screen_applied
                ? <><b>{detail.selection.funnel.calibrated}</b> of those calibrated on
                    validation<InfoTip metric={M.calibrated} /></>
                : <>none calibrated, so the screen did not apply</>} →{" "}
              <b>{detail.selection.funnel.chosen}</b>, the one with the fewest effective
              parameters<InfoTip metric={M.params} />.{" "}
              <span className="text-muted">Click a row for the candidate in full.</span>
            </p>
          )}
          {detail?.selection?.rule_note && (
            <p className="mb-1.5 text-[11px] text-warn">{detail.selection.rule_note}</p>
          )}
          <div className="max-h-[300px] overflow-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="th">#</th><th className="th">From</th>
                  <th className="th text-left">Specification</th>
                  <Th label={runObjective === "horizon" ? "Daily loss" : "Daily loss ▸"}
                      metric={M.selectionLoss} />
                  <Th label={usesHorizon ? "h-day loss ▸" : "h-day loss"}
                      metric={M.horizonLoss} />
                  <Th label="vs EWMA p" metric={M.dmDaily} />
                  <Th label="QLIKE (mo.)" metric={M.qlikeMonthly} />
                  <Th label="Bias" metric={M.meanBias} />
                  <Th label="MZ slope" metric={M.mzSlope} />
                  <Th label="VaR99 p" metric={M.kupiec99} />
                  <Th label="OOS R²" metric={M.oosR2} />
                  <Th label="Params" metric={M.params} />
                  <Th label="In set" metric={M.mcs} />
                  <Th label="Calib." metric={M.calibrated} />
                  <th className="th text-left">Selection</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((c) => {
                  const m = c.metrics ?? {};
                  const isChosen = c.candidate_id === run?.chosen_candidate;
                  const dmT = m.dm_t_daily_own_ewma, dmP = m.dm_p_daily_own_ewma;
                  const b = backtest.benchmark(dmT, dmP);
                  return (
                    <tr key={c.candidate_id}
                        className={`cursor-pointer border-t border-lineSoft hover:bg-navy/[0.03]
                                    ${isChosen ? "bg-navy/5" : ""}`}
                        onClick={() => setCard(c.candidate_id)}
                        title="Click for the candidate in full">
                      <td className={`cell ${isChosen ? "font-semibold text-navy" : ""}`}>
                        {c.candidate_id}{isChosen ? " ★" : ""}
                      </td>
                      <td className="cell text-muted">{c.origin}</td>
                      <td className="cell max-w-[360px] truncate text-left font-mono text-[10.5px]">
                        {c.status === "failed" ? <span className="text-fail">{c.note}</span>
                          : candidateSummary(c)}
                      </td>
                      <td className={`cell ${runObjective !== "horizon" ? "font-semibold" : ""}`}>
                        {num(m.common_loss ?? m.daily_loss, 4)}</td>
                      <td className={`cell ${usesHorizon ? "font-semibold" : ""}`}>
                        {num(m.common_horizon_loss ?? m.horizon_loss, 3)}</td>
                      <td className={`cell ${TONE_TEXT[b.tone]}`}>{pval(dmP)}</td>
                      <td className="cell">{num(m.qlike, 3)}</td>
                      <td className="cell">{num(m.mean_bias, 2)}</td>
                      <td className="cell">{num(m.mz_beta, 2)}</td>
                      <td className="cell">{pval(m.kupiec_p_99)}</td>
                      <td className="cell">{pct(m.oos_r2)}</td>
                      <td className="cell text-muted">{num(c.complexity, 1)}</td>
                      <td className="cell">{c.in_mcs ? "✓" : ""}</td>
                      <td className="cell">{c.calibrated == null ? "" : c.calibrated ? "✓"
                        : <span className="text-warn">✗</span>}</td>
                      <td className={`cell whitespace-nowrap text-left ${STAGE_TEXT[c.stage ?? ""] ?? ""}`}>
                        {c.stage_label ?? ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {card && detail && (
        <CandidateCard runId={detail.run.run_id} candidateId={card}
                       onClose={() => setCard(null)} />
      )}

      {(detail?.steps.length ?? 0) > 0 && (
        <details className="mt-3" open={running}>
          <summary className="cursor-pointer text-2xs font-semibold uppercase tracking-label text-muted">
            Trace ({detail!.steps.length} steps)
          </summary>
          <ol className="mt-1 max-h-[260px] space-y-0.5 overflow-auto">
            {detail!.steps.map((s) => (
              <li key={s.seq} className="flex gap-2 text-[11px] leading-snug">
                <span className="w-[72px] shrink-0 font-mono text-muted">{s.node}</span>
                <span>{s.message}</span>
              </li>
            ))}
            <div ref={traceEnd} />
          </ol>
        </details>
      )}
    </Panel>
  );
}

/** A column header with its definition one hover away. */
function Th({ label, metric }: { label: string; metric: any }) {
  return (
    <th className="th whitespace-nowrap">
      <span className="inline-flex items-center">{label}<InfoTip metric={metric} label={label} /></span>
    </th>
  );
}
