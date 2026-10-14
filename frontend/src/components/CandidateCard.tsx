"use client";

/**
 * One candidate of the model agent, described in full.
 *
 * A leaderboard row is a specification string and nine numbers. Read alone it
 * answers none of the questions it raises: what the model actually does, how much
 * data its estimates rest on, and why it was or was not chosen. This card answers
 * them, in that order, from what the run stored - the selection is replayed by
 * backend/agent/explain.py, and every number carries its definition
 * (lib/agentMetrics.ts) one hover away.
 */

import { ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Formulas, NoiseBlock, estimationLines, riskLines } from "@/components/CandidateFormulas";
import { InfoTip } from "@/components/InfoTip";
import { AGENT_METRICS as M } from "@/lib/agentMetrics";
import { AgentCandidateDetail, api, num, pct, pval } from "@/lib/api";
import { blockStyle } from "@/lib/blocks";
import { backtest } from "@/lib/verdict";

type Tone = "good" | "warn" | "bad" | "neutral";
const DOT: Record<Tone, string> = {
  good: "bg-pass", warn: "bg-warn", bad: "bg-fail", neutral: "bg-line",
};
const STAGE_TONE: Record<string, Tone> = {
  chosen: "good", eligible: "neutral", uncalibrated: "warn", outside: "bad", failed: "bad",
};

const ORIGIN: Record<string, string> = {
  seed: "One of the fixed starting points: the original model, and one departure "
      + "along each axis the methodology review found to matter.",
  llm: "Proposed by the language model after reading every earlier candidate's "
     + "validation results and the critic's hints.",
  critic: "Proposed by the rule-based critic, from what the leading candidate's "
        + "diagnostics pointed to.",
  pool: "Carried in from an earlier run - chosen for this security or for a "
      + "security of the same kind, or in its earlier confidence set - and "
      + "re-scored here on this run's validation span.",
};

const ESTIMATOR: Record<string, string> = {
  ols: "Ordinary least squares: every day in the window counts by its weight, "
     + "large and small residuals alike.",
  huber: "Huber regression: residuals beyond 1.345 standard deviations are "
       + "down-weighted, so a handful of extreme days cannot pull the betas.",
  ridge: "Ridge regression: least squares plus a penalty on the size of the betas, "
       + "which shrinks them toward zero. Less estimation noise, at the cost of "
       + "betas - and so systematic risk - that are biased down.",
};

const COV: Record<string, string> = {
  sample: "the equally weighted sample covariance of the factor returns",
  ewma: "an exponentially weighted covariance: recent days count more",
  ledoit_wolf: "the sample covariance shrunk toward a constant-correlation target "
             + "(Ledoit-Wolf), which trades a little bias for much less noise",
  blend: "a blend of an exponentially weighted covariance and a Ledoit-Wolf "
       + "shrunk long-run covariance",
};

export function CandidateCard({
  runId,
  candidateId,
  onClose,
}: {
  runId: string;
  candidateId: string;
  onClose: () => void;
}) {
  const [d, setD] = useState<AgentCandidateDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setD(null); setError(null);
    api.agentCandidate(runId, candidateId).then(setD)
      .catch((e) => setError(String(e.message ?? e)));
  }, [runId, candidateId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const c = d?.candidate;
  const e = c?.estimation ?? {};
  const r = c?.risk ?? {};
  const m = c?.metrics ?? {};
  const stageTone = STAGE_TONE[d?.stage.stage ?? ""] ?? "neutral";

  // Into document.body: the agent's panel animates in with a transform, and a
  // transformed ancestor would make this overlay fixed to the panel rather than
  // to the window.
  if (typeof document === "undefined") return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/20 p-6
                    backdrop-blur-[1px]"
         onClick={onClose} role="dialog" aria-modal="true"
         aria-label={`Candidate ${candidateId}`}>
      <div className="enter mt-6 max-h-[88vh] w-full max-w-[820px] overflow-y-auto rounded
                      border border-line bg-panel shadow-xl"
           onClick={(ev) => ev.stopPropagation()}>
        <header className="sticky top-0 z-10 flex items-start justify-between gap-4 border-b
                           border-line bg-panel/95 px-5 py-3 backdrop-blur">
          <div>
            <div className="flex items-baseline gap-2">
              <span className="font-mono text-[14px] font-semibold text-navy">
                {candidateId}{d?.chosen === candidateId ? " ★" : ""}
              </span>
              {c && <span className="rounded border border-line px-1.5 py-[1px] text-[10px]
                                   uppercase tracking-label text-muted">{c.origin}</span>}
              {c && <span className="text-[11px] text-muted">round {c.round}</span>}
            </div>
            {d && <div className="mt-0.5 text-[12px] text-muted">
              {d.run.instrument_id} · validation {d.run.validation_start} to{" "}
              {d.run.validation_end}
            </div>}
          </div>
          <button className="btn-ghost shrink-0" onClick={onClose}>Close</button>
        </header>

        {error && <div className="p-5 text-[12px] text-fail">{error}</div>}
        {!d && !error && <div className="p-5 text-[12px] text-muted">Loading…</div>}

        {d && c && (
          <div className="space-y-5 px-5 py-4 text-[12px]">
            {/* --- the selection ------------------------------------------- */}
            <section>
              <H>Where it stood in the selection</H>
              <div className={`rounded border px-3 py-2 ${
                stageTone === "good" ? "border-pass/40 bg-pass/5"
                  : stageTone === "bad" ? "border-fail/30 bg-fail/5"
                  : stageTone === "warn" ? "border-warn/40 bg-warn/5"
                  : "border-line bg-white"}`}>
                <div className="font-semibold capitalize text-navy">{d.stage.label}</div>
                <p className="mt-0.5 leading-snug">{d.stage.text}</p>
              </div>
              <Funnel d={d} />
              {d.rule_note && <p className="mt-1.5 text-[11px] text-warn">{d.rule_note}</p>}
            </section>

            <section>
              <H>Why it was tried</H>
              <p className="leading-snug text-muted">{ORIGIN[c.origin] ?? c.origin}</p>
              {c.note && c.status !== "failed" && (
                <p className="mt-1 leading-snug">“{c.note}”</p>
              )}
            </section>

            {/* --- the estimation half ---------------------------------------- */}
            <section>
              <H>How the betas are estimated</H>
              <Rows>
                <Row k="Factor panel">
                  {e.panel === "raw"
                    ? "Raw factor returns. Loadings read directly (\"beta 1.0 to global "
                      + "equity\"), at the cost of strongly overlapping regressors."
                    : "Orthogonalised factor returns: each factor residualised against "
                      + "the blocks above it, so the regressors overlap little and each "
                      + "loading is incremental."}
                </Row>
                <Row k="Factors">
                  {d.factors.length} factors -{" "}
                  {Array.isArray(e.factor_set) ? "an explicit list"
                    : e.factor_set === "template"
                      ? `the ${d.run.template ?? "asset-class"} template`
                      : "every core factor"}
                  {e.sector === "own" && <>, plus its sector factor {d.run.sector_factor}</>}.
                  <div className="mt-1 flex flex-wrap gap-1">
                    {d.factors.map((f) => {
                      const bs = blockStyle(f.block_id);
                      return (
                        <span key={f.factor_id} title={f.name}
                              className="rounded px-1 py-[1px] font-mono text-[10px]"
                              style={{ background: bs.tint, color: bs.colour }}>
                          {f.factor_id}
                        </span>
                      );
                    })}
                  </div>
                </Row>
                <Row k="Estimation window">
                  {e.window_days} trading days (≈ {num(e.window_days / 252, 1)} years). A
                  new regression every 21 trading days, each on the {e.window_days} days
                  before it.
                </Row>
                <Row k="Weighting" metric={M.effectiveObs}>
                  {e.weighting === "ewma"
                    ? <>Exponential: a day&apos;s weight halves every {e.ewma_halflife}{" "}
                        days, worth ≈ {num(d.effective.regression, 0)} equally weighted
                        days.</>
                    : <>Equal: every day in the window counts the same - {e.window_days}{" "}
                        effective observations.</>}
                  {" "}That is ≈ {num(d.effective.regression / Math.max(d.factors.length, 1), 0)}{" "}
                  observations per factor.
                </Row>
                <Row k="Estimator">
                  {ESTIMATOR[e.estimator] ?? e.estimator}
                  {e.estimator === "ridge" && <> Penalty λ = {e.ridge_lambda}.</>}
                </Row>
                <Row k="Lead-lag in the betas">
                  {e.dimson_lags
                    ? <>Dimson, {e.dimson_lags} lag{e.dimson_lags > 1 ? "s" : ""} on{" "}
                        {e.dimson_set === "all" ? "every factor"
                          : "the equity, style, FX and sector factors"}: yesterday&apos;s
                        factor moves are added as regressors, for a security that reacts
                        to them a day late.</>
                    : "None: the security is regressed on same-day factor returns only."}
                </Row>
                <Row k="Standard errors">
                  Newey-West, {e.hac_lags === "auto" || e.hac_lags == null
                    ? "with the automatic lag rule" : `${e.hac_lags} lags`}. Changes the
                  standard errors of the betas, never the forecast.
                </Row>
                <Row k="Outliers">
                  {e.winsorize ? "Returns clipped at their 1% and 99% quantiles before "
                                 + "estimation." : "Returns used as they are."}
                </Row>
                <Row k="Effective parameters" metric={M.params}>
                  {num(c.complexity, 1)} per regression
                  {d.parsimony != null && c.complexity != null
                    && Math.abs(d.parsimony - c.complexity) > 0.05
                    && <>; {num(d.parsimony, 1)} in the simplicity comparison, its sector
                         factor not charged</>}.
                </Row>
              </Rows>
              <Formulas lines={estimationLines(e, d.factors.length, d.effective.regression)} />
            </section>

            {/* --- the risk half ------------------------------------------------ */}
            <section>
              <H>How the risk forecast is built</H>
              <Rows>
                <Row k="Systematic risk">
                  β′Σβ: the betas above, held until the next regression, against the
                  factor covariance Σ.
                  {r.debias && <> The part that is only the betas&apos; estimation noise,
                    tr(ΣV), is subtracted.</>}
                </Row>
                <Row k="Factor covariance Σ" metric={M.effectiveObs}>
                  {COV[r.cov_method] ?? r.cov_method}
                  {(r.cov_method === "ewma" || r.cov_method === "blend")
                    && <>, half-life {r.halflife} days</>}, over the last {r.cov_window}{" "}
                  days{r.cov_method === "blend"
                    && <>, weighted {Math.round((r.ewma_weight ?? 0.6) * 100)}% to the
                        exponential part</>}.{" "}
                  <CovEff d={d} />
                </Row>
                <Row k="Specific risk">
                  {r.specific === "ewma"
                    ? <>Tracked out of sample: an exponential average (half-life{" "}
                        {r.specific_halflife} days, ≈ {num(d.effective.specific, 0)}{" "}
                        effective days) of the squared residuals of the betas in force,
                        each less the part due to beta noise.</>
                    : "The residual volatility of the latest regression window."}{" "}
                  Floored at 2% a year.
                </Row>
                <Row k="Refresh">
                  {r.refresh === "daily"
                    ? "Daily: the covariance and specific risk are re-estimated every "
                      + "day; the betas are held between regressions."
                    : "At each regression: the forecast is issued every 21 days and held "
                      + "until the next."}
                </Row>
                <Row k="Value at risk">
                  {r.var_dist === "t"
                    ? "Student t, its degrees of freedom fitted to the trailing returns "
                      + "divided by their own volatility - fatter tails than the normal."
                    : "Normal."}
                </Row>
                <Row k="Multi-day risk" metric={M.horizonLoss}>
                  {r.nw_lags
                    ? <>Adds {r.nw_lags} lag{r.nw_lags > 1 ? "s" : ""} of cross-day factor
                        covariance to the {d.run.horizon_days}-day forecast.</>
                    : <>The {d.run.horizon_days}-day forecast is the daily one scaled by
                        √{d.run.horizon_days}{e.dimson_lags ? ", with the betas summed "
                          + "over lags" : ""}. The lead-lag count is chosen after the
                        search, for the winner only.</>}
                </Row>
              </Rows>
              <Formulas lines={riskLines(e, r, d.factors.length, d.run.horizon_days)} />
              <NoiseBlock
                n={d.effective.covariance.effective ?? d.effective.covariance.effective_ewma ?? 0}
                K={d.factors.length}
                longWindow={d.effective.covariance.method === "blend"
                  ? d.effective.covariance.effective_long : undefined} />
            </section>

            {/* --- the numbers ----------------------------------------------------- */}
            <section>
              <H>Validation results</H>
              {c.status === "failed"
                ? <p className="text-fail">{c.note}</p>
                : <Results d={d} m={m} />}
              <p className="mt-2 text-[11px] text-muted">
                All figures are on the validation span, {d.run.validation_start} to{" "}
                {d.run.validation_end}. The test span, from {d.run.test_start}, is
                opened once, for the chosen model only, and appears only in the
                verdict. Hover a ? for the definition, computation and null hypothesis.
              </p>
            </section>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

function H({ children }: { children: ReactNode }) {
  return <h3 className="mb-1.5 text-2xs font-semibold uppercase tracking-label text-muted">
    {children}</h3>;
}

function Rows({ children }: { children: ReactNode }) {
  return <dl className="divide-y divide-lineSoft rounded border border-line bg-white">
    {children}</dl>;
}

function Row({ k, metric, children }: { k: string; metric?: any; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[150px_minmax(0,1fr)] gap-3 px-3 py-1.5">
      <dt className="flex items-start font-semibold text-navy">
        {k}{metric && <InfoTip metric={metric} label={k} />}
      </dt>
      <dd className="leading-snug">{children}</dd>
    </div>
  );
}

function CovEff({ d }: { d: AgentCandidateDetail }) {
  const k = d.factors.length;
  const entries = (k * (k + 1)) / 2;
  const cv = d.effective.covariance;
  const eff = cv.effective ?? cv.effective_ewma ?? 0;
  return (
    <>
      Worth ≈ {num(eff, 0)} equally weighted days
      {cv.effective_long != null && <> in the exponential part and {num(cv.effective_long, 0)}{" "}
        in the long-run part</>}, for a {k}-factor matrix with {entries} distinct entries
      {eff > 0 && eff < 3 * k && <> - few observations per entry, which is why the
        estimate is shrunk or blended</>}.
    </>
  );
}

function Funnel({ d }: { d: AgentCandidateDetail }) {
  const f = d.funnel;
  const step = (n: number | string, t: string, info?: any) => (
    <span className="whitespace-nowrap">
      <b className="text-navy">{n}</b> {t}{info && <InfoTip metric={info} />}
    </span>
  );
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-[11px] text-muted">
      {step(f.tried, "tried")} <span>→</span>
      {step(f.in_set, `not significantly worse than the best on ${
        f.objective === "horizon" ? "the h-day loss"
          : f.objective === "both" ? "the daily and the h-day loss" : "the daily loss"}`,
        M.mcs)} <span>→</span>
      {f.screen_applied ? step(f.calibrated, "of those calibrated", M.calibrated)
        : <span>none calibrated, screen not applied</span>} <span>→</span>
      {step(f.chosen ?? "-", "the simplest", M.params)}
    </div>
  );
}

/** "56 against 42 expected · ", or a note where this run did not record the count. */
function breaches(x: number | null | undefined, expected: number | null | undefined) {
  if (x == null || expected == null) return "count not recorded for this run · ";
  return `${x} against ${num(expected, 0)} expected · `;
}

function Results({ d, m }: { d: AgentCandidateDetail; m: Record<string, number | null> }) {
  const c = d.candidate;
  const n = m.n_forecasts ?? null;
  const band = n && n > 2 ? Math.sqrt(2 / n) : null;
  const pTest = (p: number | null | undefined, h0good = true): Tone =>
    p == null ? "neutral" : (p >= 0.05) === h0good ? "good" : "bad";
  const dmDaily = backtest.benchmark(m.dm_t_daily_own_ewma as number,
                                     m.dm_p_daily_own_ewma as number);
  const dmMonthly = backtest.benchmark(m.dm_t_own_ewma as number, m.dm_p_own_ewma as number);
  const bs = m.bias_stat;
  const groups: Array<[string, Array<[string, any, ReactNode, Tone]>]> = [
    ["How it was chosen", [
      ["Daily loss (common days)", M.selectionLoss,
        <>{num(m.common_loss ?? m.daily_loss, 4)}
          {d.funnel.objective !== "horizon" && <span className="text-muted"> - used for
            selection</span>}</>, "neutral"],
      ["h-day loss (common dates)", M.horizonLoss,
        <>{num(m.common_horizon_loss ?? m.horizon_loss, 3)}
          {(d.funnel.objective === "horizon" || d.funnel.objective === "both")
            && <span className="text-muted"> - used for selection</span>}</>, "neutral"],
      ["In the confidence set", M.mcs,
        <>{c.in_mcs ? "yes" : "no"}{c.mcs_p != null && <> · MCS p {pval(c.mcs_p)}</>}</>,
        c.in_mcs ? "good" : "bad"],
      ["Calibrated", M.calibrated,
        c.calibrated == null ? "—"
          : c.calibrated ? "yes"
          : <>no - {(d.stage.calibration_failures ?? []).join("; ")}</>,
        c.calibrated == null ? "neutral" : c.calibrated ? "good" : "warn"],
      ["Effective parameters", M.params, num(c.complexity, 1), "neutral"],
    ]],
    ["Against simpler forecasts", [
      ["Daily loss vs own EWMA", M.dmDaily,
        <>{num(m.daily_loss, 4)} against {num(m.daily_loss_own_ewma, 4)} · DM p{" "}
          {pval(m.dm_p_daily_own_ewma)} ({dmDaily.reason})</>, dmDaily.tone as Tone],
      ["Monthly QLIKE vs own EWMA", M.dmMonthly,
        <>{num(m.qlike, 3)} against {num(m.qlike_own_ewma, 3)} · DM p{" "}
          {pval(m.dm_p_own_ewma)} ({dmMonthly.reason})</>, dmMonthly.tone as Tone],
      ["Monthly QLIKE vs CAPM", M.qlikeCapm,
        <>{num(m.qlike, 3)} against {num(m.qlike_capm, 3)}
          {m.dm_p_capm != null && <> · DM p {pval(m.dm_p_capm)}</>}</>, "neutral"],
    ]],
    ["Calibration", [
      ["Bias (mean ratio)", M.meanBias, num(m.mean_bias, 2), "neutral"],
      ["Bias statistic", M.biasStatistic,
        <>{num(bs, 3)}{band != null && <> against 1 ± {num(band, 3)}</>}</>,
        bs == null || band == null ? "neutral" : Math.abs(bs - 1) <= band ? "good" : "bad"],
      ["Daily dispersion", M.zStd, num(m.z_std, 3), "neutral"],
      ["Mincer-Zarnowitz slope", M.mzSlope, num(m.mz_beta, 2), "neutral"],
      ["Mincer-Zarnowitz joint", M.mzJoint, <>p {pval(m.mz_joint_p)}</>,
        m.mz_joint_p == null ? "neutral" : m.mz_joint_p >= 0.05 ? "good" : "warn"],
      ["Mincer-Zarnowitz R²", M.mzR2, num(m.mz_r2, 3), "neutral"],
      ["VaR 95% breaches", M.kupiec95,
        <>{breaches(m.exceptions_95, m.expected_95)}Kupiec p {pval(m.kupiec_p_95)}</>,
        pTest(m.kupiec_p_95)],
      ["VaR 99% breaches", M.kupiec99,
        <>{breaches(m.exceptions_99, m.expected_99)}Kupiec p {pval(m.kupiec_p_99)}</>,
        pTest(m.kupiec_p_99)],
      ["VaR 95% clustering", M.christoffersen95, <>Christoffersen p{" "}
        {pval(m.christoffersen_p_95)}</>, pTest(m.christoffersen_p_95)],
      ["Forecasts scored", M.nForecasts, num(n, 0), "neutral"],
    ]],
    ["The regressions", [
      ["Out-of-sample R²", M.oosR2, pct(m.oos_r2), "neutral"],
      ["Adjusted R² (in sample)", M.adjR2, pct(m.adj_r2), "neutral"],
      ["Residual autocorrelation", M.residAutocorr,
        <>{pct(m.resid_autocorr_share)} of windows reject Ljung-Box</>, "neutral"],
      ["Condition number, worst window", M.condMax, num(m.cond_max, 0),
        m.cond_max == null ? "neutral" : m.cond_max > 200 ? "bad" : "good"],
      ["Maximum VIF, median window", M.maxVif, num(m.max_vif_median, 1), "neutral"],
      ["Loading change per refit", M.betaShift, num(m.beta_shift_l1_median, 2), "neutral"],
    ]],
  ];
  return (
    <div className="space-y-3">
      {groups.map(([title, rows]) => (
        <div key={title}>
          <div className="mb-1 text-[11px] font-semibold text-navy">{title}</div>
          <table className="w-full border-collapse rounded border border-line bg-white">
            <tbody>
              {rows.map(([label, metric, value, tone]) => (
                <tr key={label} className="border-t border-lineSoft first:border-t-0">
                  <td className="w-[230px] px-3 py-1 align-top">
                    <span className="inline-flex items-center">{label}
                      <InfoTip metric={metric} label={label} /></span>
                  </td>
                  <td className="px-3 py-1 align-top">
                    <span className="inline-flex items-start gap-2">
                      <span className={`mt-[5px] inline-block h-[7px] w-[7px] shrink-0
                                        rounded-full ${DOT[tone]}`} />
                      <span className="tabular-nums">{value ?? "—"}</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
