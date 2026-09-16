"use client";

/**
 * Risk Lens: is the model's predicted risk for this security actually right?
 *
 * Every forecast plotted here used loadings and a covariance matrix estimated on
 * data ending at or before its own date; the realised volatility it is scored
 * against covers the following horizon. That lag is what makes the comparison mean
 * anything, and it is enforced in the pipeline rather than assumed here.
 */

import { useEffect, useState } from "react";
import { Chart, PALETTE, Panel } from "@/components/Chart";
import { api, num, pct, pval, RiskResult, Spec } from "@/lib/api";
import { MetricCard, MetricStrip, alignFor } from "@/components/MetricCard";
import { METRICS } from "@/lib/metrics";
import { SecuritySearch } from "@/components/SecuritySearch";
import { backtest } from "@/lib/verdict";
import { usePublishSnapshot } from "@/lib/chat-context";

const HORIZONS = [
  [5, "1 week"], [21, "1 month"], [63, "3 months"], [126, "6 months"],
] as const;

export default function RiskPage() {
  const [specs, setSpecs] = useState<Spec[]>([]);
  const [specId, setSpecId] = useState("");
  const [instrument, setInstrument] = useState("US:AAPL");
  const [horizon, setHorizon] = useState(21);
  const [covMethod, setCovMethod] = useState("blend");

  const [result, setResult] = useState<RiskResult | null>(null);
  const [std, setStd] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Default to a spec that will actually return something for this security.
   *
   * The newest spec is often one estimated for a different name, and choosing it
   * lands the reader on an error where a working page was expected. Preference
   * order: covers this security, then has any loadings at all, then whatever is
   * newest.
   */
  useEffect(() => {
    api.specs(instrument).then((s) => {
      setSpecs(s);
      if (!s.length) return;
      const best =
        s.find((x: any) => x.covers_instrument) ??
        s.find((x: any) => (x.n_instruments ?? 0) > 0) ??
        s[0];
      setSpecId(best.spec_id);
    }).catch(() => {});
  }, [instrument]);

  const run = () => {
    if (!specId) return;
    setBusy(true); setError(null); setResult(null);
    api.securityRisk({
      instrument_id: instrument, spec_id: specId,
      horizon_days: horizon, cov_method: covMethod,
    })
      .then((r) => {
        setResult(r);
        return api.standardized(specId, instrument,
          { horizon_days: horizon, cov_method: covMethod }).catch(() => null);
      })
      .then(setStd)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  useEffect(() => { if (specId) run(); }, [specId]);

  const bt = result?.backtest as any;
  const reliable = (result?.forecasts ?? []).filter((f) => f.is_reliable);
  const scored = reliable.filter((f) => f.sigma_realized_ann !== null);

  const latest = reliable[reliable.length - 1];
  usePublishSnapshot(
    result
      ? {
          instrument_id: result.instrument_id,
          spec_id: result.spec_id,
          horizon_days: result.horizon_days,
          cov_method: result.cov_method,
          n_forecasts: result.n_forecasts,
          n_unreliable: result.n_unreliable,
          backtest: result.backtest,
          interpretation: result.interpretation,
          latest_forecast: latest
            ? {
                as_of: latest.as_of_date,
                sigma_pred_ann: latest.sigma_pred_ann,
                sigma_factor_ann: latest.sigma_factor_ann,
                sigma_specific_ann: latest.sigma_specific_ann,
                factor_risk_share: latest.factor_risk_share,
              }
            : undefined,
          standardized: std
            ? { std: std.std, skew: std.skew,
                excess_kurtosis: std.excess_kurtosis, n: std.n }
            : undefined,
        }
      : null
  );

  return (
    <div className="space-y-4">
      <Panel title="Forecast">
        <div className="flex flex-wrap items-end gap-3">
          {/*
            The same catalogue search as the Loadings Lab. Risk scoring needs
            loadings to exist first, so a name picked here that has none will be
            told so - but it should at least be pickable by name rather than only
            by an exact id typed from memory.
          */}
          <div className="w-56">
            <label className="label">Security</label>
            <SecuritySearch value={instrument} onChange={setInstrument} />
          </div>
          <div>
            <label className="label">Model spec</label>
            <select className="field mt-1 w-72" value={specId}
                    onChange={(e) => setSpecId(e.target.value)}>
              {specs.map((s) => (
                <option key={s.spec_id} value={s.spec_id}>
                  {s.name} · {s.n_instruments} instruments · {s.spec_id.slice(0, 8)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label">Horizon</label>
            <select className="field mt-1 w-32" value={horizon}
                    onChange={(e) => setHorizon(Number(e.target.value))}>
              {HORIZONS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
          <div>
            <label className="label">Covariance</label>
            <select className="field mt-1 w-44" value={covMethod}
                    onChange={(e) => setCovMethod(e.target.value)}>
              <option value="blend">Blend</option>
              <option value="ledoit_wolf">Ledoit-Wolf</option>
              <option value="ewma">EWMA</option>
              <option value="sample">Sample</option>
            </select>
          </div>
          <button className="btn" onClick={run} disabled={busy || !specId}>
            {busy ? "Scoring…" : "Run backtest"}
          </button>
        </div>
        <p className="mt-2 text-[11px] text-muted">
          Loadings must exist for this security and spec. Estimate them first in the
          Loadings Lab if the request comes back empty.
        </p>
      </Panel>

      {error && (
        <div className="rounded border border-fail/30 bg-fail/5 p-3 text-[12px] text-fail">
          {error}
        </div>
      )}

      {(result?.interpretation?.length ?? 0) > 0 && (
        <Panel title="What the tests say">
          <ul className="space-y-1.5">
            {result!.interpretation.map((line, i) => (
              <li key={i} className="flex gap-2 text-[12px] leading-snug">
                <span className="text-navy3">—</span>
                <span>{line}</span>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      {/*
        Every tone comes from lib/verdict.ts. Two of these deserve amber rather than
        red and would get red from a naive rule: an MZ slope below 1 is partly
        mechanical at this horizon, and a rejected joint test is evidence rather
        than a failed model.
      */}
      {bt && (
        <MetricStrip>
          {([
            ["Forecasts scored", bt.n_forecasts?.toLocaleString(),
             result!.n_unreliable ? `${result!.n_unreliable} excluded` : undefined,
             "neutral", METRICS.forecastsScored],
            ["Bias ratio", num(bt.mean_bias, 3),
             backtest.bias(bt.mean_bias).reason, backtest.bias(bt.mean_bias).tone,
             METRICS.biasRatio],
            ["Bias statistic", num(bt.z_std, 3),
             backtest.zStd(bt.z_std).reason, backtest.zStd(bt.z_std).tone,
             METRICS.biasStatistic],
            ["MZ slope", num(bt.mz_beta, 2),
             backtest.mzSlope(bt.mz_beta).reason, backtest.mzSlope(bt.mz_beta).tone,
             METRICS.mzSlope],
            ["MZ joint p", pval(bt.mz_joint_p),
             backtest.mzJoint(bt.mz_joint_p).reason,
             backtest.mzJoint(bt.mz_joint_p).tone, METRICS.mzJoint],
            ["MZ R²", num(bt.mz_r2, 3),
             backtest.mzR2(bt.mz_r2).reason, backtest.mzR2(bt.mz_r2).tone,
             METRICS.mzR2],
            ["VaR 95%", `${bt.exceptions_95} / ${num(bt.expected_95, 0)}`,
             `Kupiec p ${pval(bt.kupiec_p_95)}`,
             backtest.kupiec(bt.kupiec_p_95).tone, METRICS.var95],
            ["VaR 99%", `${bt.exceptions_99} / ${num(bt.expected_99, 0)}`,
             `Kupiec p ${pval(bt.kupiec_p_99)}`,
             backtest.kupiec(bt.kupiec_p_99).tone, METRICS.var99],
            ["Clustering", pval(bt.christoffersen_p_95),
             backtest.christoffersen(bt.christoffersen_p_95).reason,
             backtest.christoffersen(bt.christoffersen_p_95).tone,
             METRICS.clustering],
          ] as const).map(([label, value, sub, tone, metric], i) => (
            <MetricCard key={label} label={label} value={value} sub={sub}
                        tone={tone} metric={metric} align={alignFor(i)} />
          ))}
        </MetricStrip>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        {scored.length > 0 && (
          <Panel
            title="Predicted versus realised volatility"
            caption="Predicted uses only information available at each date; realised covers the following horizon. The two should track, not merely average out."
          >
            <Chart
              height={300}
              data={[
                { x: scored.map((f) => f.as_of_date), y: scored.map((f) => f.sigma_pred_ann),
                  type: "scatter", mode: "lines", name: "predicted",
                  line: { color: PALETTE[0], width: 1.5 } },
                { x: scored.map((f) => f.as_of_date),
                  y: scored.map((f) => f.sigma_realized_ann),
                  type: "scatter", mode: "lines", name: "realised (forward)",
                  line: { color: PALETTE[1], width: 1.5 } },
              ]}
              layout={{ yaxis: { title: "annualised volatility", tickformat: ".0%" } }}
            />
          </Panel>
        )}

        {scored.length > 0 && (
          <Panel
            title="Bias ratio through time"
            caption="Realised divided by predicted. Persistent excursions above 1 mean the model is systematically under-forecasting, which no amount of averaging will reveal."
          >
            <Chart
              height={300}
              data={[
                { x: scored.map((f) => f.as_of_date), y: scored.map((f) => f.bias_ratio),
                  type: "scatter", mode: "lines", name: "realised ÷ predicted",
                  line: { color: PALETTE[3], width: 1.2 } },
                { x: [scored[0].as_of_date, scored[scored.length - 1].as_of_date],
                  y: [1, 1], type: "scatter", mode: "lines", name: "calibrated",
                  line: { color: "#9AA0AE", width: 1, dash: "dash" } },
              ]}
              layout={{ yaxis: { title: "ratio" } }}
            />
          </Panel>
        )}

        {scored.length > 0 && (
          <Panel
            title="Mincer-Zarnowitz"
            caption="Realised variance against predicted variance. A correct forecast sits on the 45-degree line. A slope below 1 is partly mechanical when volatility moves faster than the horizon measures."
          >
            <Chart
              height={300}
              numericX
              data={[
                { x: scored.map((f) => f.sigma_pred_ann ** 2),
                  y: scored.map((f) => (f.sigma_realized_ann ?? 0) ** 2),
                  type: "scatter", mode: "markers", name: "windows",
                  marker: { color: PALETTE[0], size: 5, opacity: 0.55 } },
                ...(bt ? [{
                  x: scored.map((f) => f.sigma_pred_ann ** 2),
                  y: scored.map((f) => bt.mz_alpha + bt.mz_beta * f.sigma_pred_ann ** 2),
                  type: "scatter" as const, mode: "lines" as const,
                  name: `fit (slope ${num(bt.mz_beta, 2)})`,
                  line: { color: PALETTE[1], width: 1.5 },
                }] : []),
                { x: scored.map((f) => f.sigma_pred_ann ** 2),
                  y: scored.map((f) => f.sigma_pred_ann ** 2),
                  type: "scatter", mode: "lines", name: "45°",
                  line: { color: "#9AA0AE", width: 1, dash: "dash" } },
              ]}
              layout={{
                xaxis: { title: "predicted variance" },
                yaxis: { title: "realised variance" },
                hovermode: "closest",
              }}
            />
          </Panel>
        )}

        {std && (
          <Panel
            title="Standardised returns"
            caption={`Returns divided by the forecast in force at the time. Standard deviation ${num(std.std, 3)} against a target of 1; excess kurtosis ${num(std.excess_kurtosis, 2)}. Fat tails that survive standardisation are real tail risk, not a volatility-timing failure.`}
          >
            <Chart
              height={300}
              numericX
              data={[
                { x: std.bin_centres, y: std.density, type: "bar", name: "standardised returns",
                  marker: { color: PALETTE[0], opacity: 0.5 } },
                { x: std.bin_centres, y: std.normal_pdf, type: "scatter", mode: "lines",
                  name: "standard normal", line: { color: PALETTE[1], width: 1.5 } },
              ]}
              layout={{ bargap: 0, xaxis: { title: "z" }, yaxis: { title: "density" },
                        hovermode: "closest" }}
            />
          </Panel>
        )}

        {reliable.length > 0 && (
          <Panel
            title="Risk decomposition"
            caption="Factor risk against specific risk. A security whose risk is mostly specific is one the factor model has little to say about — which is itself worth knowing."
          >
            <Chart
              height={300}
              data={[
                { x: reliable.map((f) => f.as_of_date),
                  y: reliable.map((f) => f.sigma_factor_ann), type: "scatter",
                  mode: "lines", name: "factor", stackgroup: "one",
                  line: { color: PALETTE[0], width: 0 }, fillcolor: PALETTE[0] + "CC" },
                { x: reliable.map((f) => f.as_of_date),
                  y: reliable.map((f) => f.sigma_specific_ann), type: "scatter",
                  mode: "lines", name: "specific", stackgroup: "one",
                  line: { color: PALETTE[4], width: 0 }, fillcolor: PALETTE[4] + "CC" },
              ]}
              layout={{ yaxis: { title: "annualised volatility", tickformat: ".0%" } }}
            />
          </Panel>
        )}

        {std && (
          <Panel
            title="Quantile-quantile of standardised returns"
            caption="If the model were perfectly calibrated with normal innovations these would lie on the 45-degree line. Departure at the ends is what the VaR exception counts are picking up."
          >
            <Chart
              height={300}
              numericX
              data={[
                { x: std.qq_normal, y: std.qq_sample, type: "scatter", mode: "markers",
                  name: "standardised", marker: { color: PALETTE[0], size: 4, opacity: 0.6 } },
                { x: std.qq_normal, y: std.qq_normal, type: "scatter", mode: "lines",
                  name: "45°", line: { color: "#9AA0AE", width: 1, dash: "dash" } },
              ]}
              layout={{ xaxis: { title: "normal quantile" }, yaxis: { title: "sample quantile" },
                        hovermode: "closest" }}
            />
          </Panel>
        )}
      </div>

      {result && result.n_unreliable > 0 && (
        <Panel
          title="Excluded forecasts"
          caption="Built on ill-conditioned estimation windows, where the loading vector is not identified well enough to forecast with. Shown rather than deleted, because suppressing the model's own output would hide the weakness."
        >
          <div className="max-h-[240px] overflow-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="th">As of</th><th className="th">Predicted vol</th>
                  <th className="th">Condition no.</th><th className="th">Max VIF</th>
                  <th className="th">Reason</th>
                </tr>
              </thead>
              <tbody>
                {result.forecasts.filter((f) => !f.is_reliable).map((f) => (
                  <tr key={f.as_of_date} className="border-t border-lineSoft">
                    <td className="cell">{f.as_of_date}</td>
                    <td className="cell text-fail">{pct(f.sigma_pred_ann)}</td>
                    <td className="cell">{num(f.condition_number, 0)}</td>
                    <td className="cell">{num(f.max_vif, 0)}</td>
                    <td className="cell font-sans text-muted">{f.unreliable_reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}
