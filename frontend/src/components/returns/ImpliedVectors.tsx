"use client";

/**
 * Return vectors: the building blocks beside the rear-view mirror and the
 * vectors implied by reference portfolios, all as arithmetic excess returns
 * over cash - the units an optimiser takes.
 *
 * The implied vectors are not forecasts. They are what a market-cap, a 1/N or a
 * minimum-variance portfolio would have to believe to be optimal, and they are
 * where a doubtful forecast can be shrunk toward. Hand-entered capital market
 * assumptions (J.P. Morgan, Invesco, an in-house view) join the building blocks
 * as further "conviction" vectors once all twelve classes are filled in.
 */

import { useEffect, useMemo, useState } from "react";
import { Chart, Panel } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { ExternalCma, ReturnForecast, api, num, pct } from "@/lib/api";
import { RETURN_METRICS as RM } from "@/lib/returnMetrics";
import { classLabels, riskyKeys, vectorColour, vectorOrder } from "@/components/returns/shared";

const GROUP_LABEL: Record<string, string> = {
  conviction: "conviction", concern: "concern", reference: "reference", blend: "blend",
};

export function ImpliedVectors({ result }: { result: ReturnForecast }) {
  const labels = classLabels(result);
  const keys = riskyKeys(result);
  const order = vectorOrder(result);

  const data = useMemo(() => order.map((k, i): Plotly.Data => {
    const v = result.vectors[k];
    const isLine = k === "blend";
    return {
      type: isLine ? "scatter" : "bar",
      mode: isLine ? "markers" : undefined,
      name: v.label,
      x: keys.map((c) => labels[c]),
      y: v.values,
      marker: isLine
        ? { symbol: "diamond", size: 9, color: "#F5F4F0", line: { color: vectorColour(k, i), width: 1.6 } }
        : { color: vectorColour(k, i) },
      hovertemplate: `${v.label}: %{y:.2%}<extra>%{x}</extra>`,
    };
  }), [order, result, keys, labels]);

  return (
    <div className="space-y-4">
      <Panel
        title="Return vectors"
        caption={<>Arithmetic annual excess returns over cash. The implied vectors are μ = λΣw with
          λ = <b>{num(result.lambda.value, 2)}</b>, set so that {result.lambda.basis}
          <InfoTip metric={RM.lambda} />. The minimum-variance vector is nearly flat by construction:
          every asset it holds gets the same implied return<InfoTip metric={RM.implied} />.</>}
      >
        <Chart data={data} height={420}
               layout={{ barmode: "group", hovermode: "closest", yaxis: { tickformat: ".0%" },
                         xaxis: { tickangle: -30 },
                         legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0 },
                         margin: { t: 50, b: 120 } }} />
      </Panel>

      <Panel title="Vectors by class" caption="Weights are each vector's effective share of the blend.">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="th">Class</th>
                {order.map((k) => (
                  <th key={k} className="th text-right">
                    <span style={{ color: vectorColour(k) }}>■</span> {result.vectors[k].label}
                    <div className="font-normal normal-case tracking-normal">
                      {GROUP_LABEL[result.vectors[k].group]}
                      {result.vectors[k].weight != null && <> · {pct(result.vectors[k].weight, 0)}</>}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {keys.map((c, i) => (
                <tr key={c} className="border-t border-lineSoft">
                  <td className="cell font-sans text-[11.5px]">{labels[c]}</td>
                  {order.map((k) => (
                    <td key={k} className={"cell text-right " + (k === "blend" ? "font-semibold text-navy" : "")}>
                      {pct(result.vectors[k].values[i], 2)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-[11px] text-muted">
          Composite funds are mapped onto the regional classes by a style regression on the last 36
          months: {Object.entries(result.composites).map(([t, w]) => (
            <span key={t} className="mr-2">
              <b>{t}</b> {Object.entries(w).map(([k, v]) => `${labels[k]?.split(" ")[0]} ${pct(v, 0)}`).join(", ")}.
            </span>))}
        </p>
      </Panel>

      <ExternalCmas keys={keys} labels={labels} />
    </div>
  );
}

/**
 * Hand-entered capital market assumptions. Typed in, never fetched: the
 * published numbers are the publishers', and a vector only joins the blend once
 * every class has a value.
 */
function ExternalCmas({ keys, labels }: { keys: string[]; labels: Record<string, string> }) {
  const [rows, setRows] = useState<ExternalCma[]>([]);
  const [draft, setDraft] = useState<{ source: string; as_of: string; note: string;
                                       values: Record<string, string> }>(
    { source: "", as_of: new Date().toISOString().slice(0, 10), note: "", values: {} });
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = () => api.returnsExternal().then(setRows).catch((e) => setErr(String(e.message ?? e)));
  useEffect(() => { load(); }, []);

  const save = async () => {
    setErr(null); setMsg(null);
    const values: Record<string, number> = {};
    for (const k of keys) {
      const s = (draft.values[k] ?? "").trim();
      if (s !== "") values[k] = Number(s) / 100;
    }
    if (!draft.source.trim() || !Object.keys(values).length) {
      setErr("A source name and at least one value are needed."); return;
    }
    try {
      const r = await api.returnsExternalSave({ source: draft.source.trim(), as_of: draft.as_of,
                                               note: draft.note || null, values });
      setMsg(r.complete ? "Stored. Run the forecast again to include it as a conviction vector."
                        : `Stored ${r.stored} classes. It joins the blend once all ${keys.length} are filled.`);
      load();
    } catch (e: any) { setErr(String(e.message ?? e)); }
  };

  const remove = async (r: ExternalCma) => {
    await api.returnsExternalRemove(r.source, r.as_of).catch((e) => setErr(String(e.message ?? e)));
    load();
  };

  return (
    <Panel
      title="External capital market assumptions"
      caption="Compound annual returns from a published source, typed in by hand (J.P. Morgan LTCMA, Invesco CMA, an in-house view). Stored with their source and date; used as conviction vectors next to the building blocks."
    >
      {rows.length > 0 && (
        <table className="mb-3 w-full border-collapse">
          <thead><tr><th className="th">Source</th><th className="th">As of</th>
            <th className="th">Classes</th><th className="th">Note</th><th className="th" /></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.source + r.as_of} className="border-t border-lineSoft">
                <td className="cell font-sans">{r.source}</td>
                <td className="cell">{r.as_of}</td>
                <td className="cell">{Object.keys(r.values).length} / {keys.length}</td>
                <td className="cell font-sans text-muted">{r.note}</td>
                <td className="cell text-right">
                  <button className="btn-ghost px-1.5 py-0" onClick={() => remove(r)} title="Remove">✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <details>
        <summary className="cursor-pointer text-[12px] text-navy">Add a source</summary>
        <div className="mt-2 flex flex-wrap gap-3">
          <div><label className="label">Source</label>
            <input className="field mt-1 w-56" value={draft.source} placeholder="e.g. JPM LTCMA 2026"
                   onChange={(e) => setDraft({ ...draft, source: e.target.value })} /></div>
          <div><label className="label">As of</label>
            <input type="date" className="field mt-1 w-40" value={draft.as_of}
                   onChange={(e) => setDraft({ ...draft, as_of: e.target.value })} /></div>
          <div className="min-w-[240px] flex-1"><label className="label">Note</label>
            <input className="field mt-1" value={draft.note}
                   onChange={(e) => setDraft({ ...draft, note: e.target.value })} /></div>
        </div>
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          {keys.map((k) => (
            <div key={k}>
              <label className="label block truncate normal-case tracking-normal" title={labels[k]}>
                {labels[k]} (%)</label>
              <input type="number" step="0.1" className="field mt-0.5 text-right"
                     value={draft.values[k] ?? ""}
                     onChange={(e) => setDraft({ ...draft, values: { ...draft.values, [k]: e.target.value } })} />
            </div>
          ))}
        </div>
        <button className="btn mt-3" onClick={save}>Store</button>
      </details>
      {msg && <p className="mt-2 text-[11px] text-pass">{msg}</p>}
      {err && <p className="mt-2 text-[11px] text-fail">{err}</p>}
    </Panel>
  );
}
