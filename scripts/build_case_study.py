"""Build the AMZN case study: every figure and every number, from the database.

The case study argues a point — that the orthogonalised panel is worth its cost —
so nothing in it may be typed by hand. Every figure is drawn here and every number
quoted in the prose is emitted as a LaTeX macro, so a rerun after a refreshed panel
updates the charts and the sentences together. A figure that disagrees with the
sentence beside it is the failure mode this exists to prevent.

    python -m scripts.build_case_study          # figures + numbers
    python -m scripts.build_case_study --pdf    # and typeset the document
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from backend.core import covariance as cv
from backend.pipeline.dbsync import connect

# --------------------------------------------------------------------------
# what is being studied
# --------------------------------------------------------------------------

INSTRUMENT = "US:AMZN"
ORTH_SPEC = "88b9b8e7d5af77f185231d7c013f3075"
RAW_SPEC = "a8d1454d32ea74fb103f8a7e8edafb3e"
COV_WINDOW = 504          # matches run_risk
SPECIFIC_FLOOR_ANN = 0.02
SPECIFIC_SHRINK = 0.25

DOCS = Path(__file__).resolve().parents[1] / "Documentation"
FIGS = DOCS / "figures"

# The application's palette, so the paper and the terminal look like one project.
NAVY = "#23456B"
BRICK = "#8C3A2E"
RULE = "#C9C4BA"
PAPER = "#F7F5F0"
MOSS = "#4A6B4A"
INK = "#1A1A1A"
MUTED = "#6E6A63"

# Editorial shading, not a model output. Same list the terminal uses.
EPISODES = [
    ("2008-09-01", "2009-03-31", "GFC"),
    ("2020-02-19", "2020-04-30", "COVID"),
    ("2022-01-03", "2022-10-14", "Inflation"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "Book Antiqua", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": RULE,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.55,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

PCT = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")


def _clean(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _episodes(ax: plt.Axes, label: bool = False) -> None:
    """Crisis bands. Editorial context, never a model result — said so in the caption."""
    lo, hi = ax.get_ylim()
    for start, end, name in EPISODES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=RULE, alpha=0.30, lw=0, zorder=0)
        if label:
            ax.text(pd.Timestamp(start), hi, f" {name}", va="top", ha="left",
                    fontsize=6.5, color=MUTED, zorder=1)
    ax.set_ylim(lo, hi)


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.pdf")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load() -> dict:
    """Everything the document needs, in one pass over the database."""
    q_meta = """
        SELECT window_end, n_obs, alpha, t_alpha, r2, adj_r2, resid_vol_ann,
               durbin_watson, condition_number, max_vif, beta_shift_l1,
               beta_corr_prev, beta_overlap
          FROM fact_regression_meta
         WHERE spec_id = %s AND instrument_id = %s
         ORDER BY window_end
    """
    q_fc = """
        SELECT as_of_date, sigma_pred_ann, sigma_factor_ann, sigma_specific_ann,
               factor_risk_share, sigma_realized_ann, var95_pred, var99_pred,
               is_reliable, condition_number, max_vif
          FROM fact_risk_forecast
         WHERE spec_id = %s AND instrument_id = %s
         ORDER BY as_of_date
    """
    q_bt = """
        SELECT * FROM fact_risk_backtest
         WHERE spec_id = %s AND instrument_id = %s
    """
    q_load = """
        SELECT window_end, factor_id, beta, se, t_stat, p_value, vif
          FROM fact_loading
         WHERE spec_id = %s AND instrument_id = %s
         ORDER BY window_end, factor_id
    """

    out: dict = {}
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT date, ret_log, adj_close FROM fact_input_return "
            "WHERE instrument_id = %s ORDER BY date", (INSTRUMENT,))
        px = pd.DataFrame(cur.fetchall(), columns=["date", "ret", "close"])
        px["date"] = pd.to_datetime(px["date"])
        out["prices"] = px.set_index("date").astype(float)

        for key, spec in (("orth", ORTH_SPEC), ("raw", RAW_SPEC)):
            cur.execute(q_meta, (spec, INSTRUMENT))
            cols = [d[0] for d in cur.description]
            m = pd.DataFrame(cur.fetchall(), columns=cols)
            m["window_end"] = pd.to_datetime(m["window_end"])
            out[f"meta_{key}"] = m.set_index("window_end").astype(float)

            cur.execute(q_fc, (spec, INSTRUMENT))
            cols = [d[0] for d in cur.description]
            f = pd.DataFrame(cur.fetchall(), columns=cols)
            f["as_of_date"] = pd.to_datetime(f["as_of_date"])
            out[f"fc_{key}"] = f.set_index("as_of_date")

            cur.execute(q_bt, (spec, INSTRUMENT))
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            out[f"bt_{key}"] = dict(zip(cols, row)) if row else {}

            cur.execute(q_load, (spec, INSTRUMENT))
            cols = [d[0] for d in cur.description]
            lo = pd.DataFrame(cur.fetchall(), columns=cols)
            lo["window_end"] = pd.to_datetime(lo["window_end"])
            out[f"load_{key}"] = lo

        # The factor panel, both bases, for the covariance the decomposition needs.
        for key, col in (("orth", "ret_orth"), ("raw", "ret_excess")):
            cur.execute(
                f"SELECT date, factor_id, {col} FROM fact_factor_return ORDER BY date")
            p = pd.DataFrame(cur.fetchall(), columns=["date", "factor_id", "ret"])
            p["date"] = pd.to_datetime(p["date"])
            out[f"panel_{key}"] = p.pivot(index="date", columns="factor_id",
                                          values="ret").astype(float)

        cur.execute("SELECT f.factor_id, b.block_id, b.name FROM ref_factor f "
                    "JOIN ref_factor_block b ON b.block_id = f.block_id")
        out["blocks"] = {r[0]: r[2] for r in cur.fetchall()}

        # The peer variance the specific risk is shrunk toward, exactly as
        # run_risk.peer_specific_variance computes it.
        for key, spec in (("orth", ORTH_SPEC), ("raw", RAW_SPEC)):
            cur.execute(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY resid_vol_ann) "
                "FROM fact_regression_meta WHERE spec_id = %s "
                "AND resid_vol_ann IS NOT NULL", (spec,))
            row = cur.fetchone()
            out[f"peer_{key}"] = float(row[0]) ** 2 if row and row[0] else float("nan")
    return out


def block_decomposition(d: dict, key: str) -> pd.DataFrame:
    """Euler risk contributions at the latest window, aggregated to block.

    Rebuilt here rather than read from a table because run_risk stores the total
    and the factor/specific split, not the per-factor contributions. The covariance
    is the same 504-day blend the forecast used, so the figure and the stored
    sigma_pred agree.
    """
    load = d[f"load_{key}"]
    panel = d[f"panel_{key}"]
    as_of = load["window_end"].max()

    betas = (load[load["window_end"] == as_of]
             .set_index("factor_id")["beta"].astype(float))
    window = panel.loc[:as_of].tail(COV_WINDOW)
    usable = [f for f in betas.index
              if f in window.columns and window[f].notna().mean() >= 0.9]
    R = window[usable].dropna().to_numpy()

    res = cv.estimate(R, usable, method="blend", halflife=60.0)
    b = betas[usable].to_numpy(dtype=float)

    spec_vol = float(d[f"meta_{key}"].loc[as_of, "resid_vol_ann"])
    spec_var = spec_vol**2
    peer = d[f"peer_{key}"]
    if np.isfinite(peer):
        spec_var = (1 - SPECIFIC_SHRINK) * spec_var + SPECIFIC_SHRINK * peer
    spec_var = max(spec_var, SPECIFIC_FLOOR_ANN**2)
    rc = cv.risk_contributions(b, res.cov, spec_var)

    df = pd.DataFrame({"factor_id": usable, "rc": rc})
    df["block"] = df["factor_id"].map(d["blocks"])
    out = df.groupby("block")["rc"].sum().sort_values(ascending=False)

    total, systematic, specific = cv.predicted_volatility(b, res.cov, spec_var)
    out.loc["Specific"] = spec_var / total
    return pd.DataFrame({"contribution": out, "share": out / total})


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_security(d: dict) -> None:
    px = d["prices"].loc["2004":]
    cum = px["ret"].cumsum()
    vol = px["ret"].rolling(63).std() * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 3.9), sharex=True,
                             gridspec_kw={"height_ratios": [1.45, 1]})
    ax = axes[0]
    ax.plot(cum.index, np.exp(cum), color=NAVY, lw=1.1)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.set_title("Amazon.com, 2004–2026: the security the model has to price",
                 loc="left", color=INK)
    _episodes(ax, label=True)
    _clean(ax)

    ax = axes[1]
    ax.plot(vol.index, vol, color=BRICK, lw=0.9)
    ax.axhline(float(px["ret"].std() * np.sqrt(252)), color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.set_ylabel("63-day vol, annualised")
    ax.yaxis.set_major_formatter(PCT)
    ax.text(cum.index[-1], float(px["ret"].std() * np.sqrt(252)) * 1.05,
            "full-sample average ", fontsize=6.5, color=MUTED,
            va="bottom", ha="right")
    _episodes(ax)
    _clean(ax)
    save(fig, "amzn-security")


def fig_loadings(d: dict) -> None:
    """Latest window, both panels, ordered by the orthogonalised t-statistic."""
    def latest(key: str) -> pd.DataFrame:
        lo = d[f"load_{key}"]
        return (lo[lo["window_end"] == lo["window_end"].max()]
                .drop(columns=["window_end"])
                .set_index("factor_id").astype(float))

    o, r = latest("orth"), latest("raw")
    order = o["t_stat"].abs().sort_values(ascending=False).head(14).index[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(order))
    for i, f in enumerate(order):
        bo, br = o.loc[f, "beta"], r.loc[f, "beta"] if f in r.index else np.nan
        ax.plot([bo, br], [i, i], color=RULE, lw=1.0, zorder=1)
    # Raw first, orthogonalised on top and smaller: where the two agree the pair
    # would otherwise render as one marker, and "no navy dot" reads as missing data
    # rather than as the two panels saying the same thing.
    ax.scatter(r.reindex(order)["beta"], y, s=44, facecolor="none",
               edgecolor=BRICK, lw=1.2, marker="D", zorder=3, label="Raw")
    ax.scatter(o.loc[order, "beta"], y, s=22, color=NAVY, zorder=4,
               label="Orthogonalised")
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(list(order), fontfamily="monospace", fontsize=7)
    ax.set_xlabel("Loading on the factor")
    ax.set_title("The same security, two panels: loadings in the final window",
                 loc="left", color=INK)
    ax.legend(loc="lower right")
    _clean(ax)
    save(fig, "amzn-loadings")


def fig_beta_paths(d: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True)
    for ax, factor in zip(axes, ("eq_global", "eq_us")):
        for key, colour, label in (("orth", NAVY, "Orthogonalised"),
                                   ("raw", BRICK, "Raw")):
            lo = d[f"load_{key}"]
            s = lo[lo["factor_id"] == factor].set_index("window_end")
            ax.plot(s.index, s["beta"].astype(float), color=colour, lw=1.0,
                    label=label)
            ax.fill_between(s.index,
                            s["beta"].astype(float) - 1.96 * s["se"].astype(float),
                            s["beta"].astype(float) + 1.96 * s["se"].astype(float),
                            color=colour, alpha=0.10, lw=0)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_ylabel(factor, fontfamily="monospace", fontsize=7.5)
        _episodes(ax)
        _clean(ax)
    axes[0].set_title("Beta paths with 95% HAC bands: what each panel calls "
                      "‘equity exposure’", loc="left", color=INK)
    axes[0].legend(loc="upper left", ncols=2)
    save(fig, "amzn-beta-paths")


def fig_fit(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for key, colour, label in (("orth", NAVY, "Orthogonalised"),
                               ("raw", BRICK, "Raw")):
        m = d[f"meta_{key}"]
        ax.plot(m.index, m["adj_r2"], color=colour, lw=1.0, label=label)
        ax.axhline(float(m["adj_r2"].mean()), color=colour, lw=0.7, ls=(0, (4, 3)),
                   alpha=0.8)
    ax.set_ylabel("Adjusted $R^2$")
    ax.set_title("In-sample fit, window by window — the raw panel wins here, "
                 "and only here", loc="left", color=INK)
    ax.legend(loc="upper left", ncols=2)
    _episodes(ax)
    _clean(ax)
    save(fig, "amzn-fit")


def fig_conditioning(d: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.0), sharex=True)
    for ax, col, gate, title in (
        (axes[0], "condition_number", 200, "Condition number of the design"),
        (axes[1], "max_vif", 100, "Largest variance inflation factor"),
    ):
        for key, colour, label in (("orth", NAVY, "Orthogonalised"),
                                   ("raw", BRICK, "Raw")):
            m = d[f"meta_{key}"]
            ax.plot(m.index, m[col], color=colour, lw=0.9, label=label)
        ax.axhline(gate, color=MOSS, lw=0.9, ls=(0, (3, 2)))
        ax.text(d["meta_orth"].index[3], gate * 1.25, f"gate at {gate}",
                fontsize=6.5, color=MOSS)
        ax.set_yscale("log")
        ax.set_ylabel(title, fontsize=7.5)
        _clean(ax)
    axes[0].set_title("Design health on a log scale: the cost the raw panel pays "
                      "for reading directly", loc="left", color=INK)
    axes[0].legend(loc="upper right", ncols=2)
    save(fig, "amzn-conditioning")


def fig_stability(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4),
                             gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for key, colour, label in (("orth", NAVY, "Orthogonalised"),
                               ("raw", BRICK, "Raw")):
        m = d[f"meta_{key}"]
        ax.plot(m.index, m["beta_shift_l1"], color=colour, lw=0.9, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("$L_1$ move of the loading vector")
    ax.set_title("How far the exposure vector travels between windows",
                 loc="left", color=INK)
    ax.legend(loc="upper left", ncols=2)
    _clean(ax)

    ax = axes[1]
    vals = [float(d[f"meta_{k}"]["beta_shift_l1"].median()) for k in ("orth", "raw")]
    ax.bar(["Orth.", "Raw"], vals, color=[NAVY, BRICK], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_title("Median move", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    _clean(ax)
    save(fig, "amzn-stability")


def fig_forecast(d: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True)
    for ax, key, colour, label in ((axes[0], "orth", NAVY, "Orthogonalised"),
                                   (axes[1], "raw", BRICK, "Raw")):
        f = d[f"fc_{key}"]
        ax.plot(f.index, f["sigma_realized_ann"].astype(float), color=MUTED, lw=0.9,
                label="Realised, next 21 days")
        ax.plot(f.index, f["sigma_pred_ann"].astype(float), color=colour, lw=1.1,
                label=f"Predicted, {label.lower()}")
        bad = f[~f["is_reliable"].astype(bool)]
        if len(bad):
            ax.scatter(bad.index, bad["sigma_pred_ann"].astype(float), s=14,
                       facecolor="none", edgecolor=BRICK, lw=0.8, zorder=4,
                       label=f"excluded from scoring ({len(bad)})")
        ax.yaxis.set_major_formatter(PCT)
        ax.legend(loc="upper left", ncols=3)
        _episodes(ax)
        _clean(ax)
    axes[0].set_title("Forecast against outcome: what the model said, and what "
                      "happened next", loc="left", color=INK)
    save(fig, "amzn-forecast")


def fig_mz(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharex=True, sharey=True)
    for ax, key, colour, label in ((axes[0], "orth", NAVY, "Orthogonalised"),
                                   (axes[1], "raw", BRICK, "Raw")):
        f = d[f"fc_{key}"]
        f = f[f["is_reliable"].astype(bool)]
        x = f["sigma_pred_ann"].astype(float) ** 2
        y = f["sigma_realized_ann"].astype(float) ** 2
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=9, color=colour, alpha=0.45, lw=0)

        bt = d[f"bt_{key}"]
        grid = np.linspace(0, float(x[ok].max()), 50)
        ax.plot(grid, float(bt["mz_alpha"]) + float(bt["mz_beta"]) * grid,
                color=colour, lw=1.3)
        ax.plot(grid, grid, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.set_title(f"{label}: slope {float(bt['mz_beta']):.2f}  "
                     f"(joint $p$ = {float(bt['mz_joint_p']):.3f})",
                     loc="left", color=INK, fontsize=8.5)
        ax.set_xlabel("Predicted variance")
        _clean(ax)
    axes[0].set_ylabel("Realised variance")
    fig.suptitle("Mincer-Zarnowitz: a perfect forecast lies on the dashed 45° line",
                 x=0.02, ha="left", fontsize=9, color=INK)
    save(fig, "amzn-mz")


def fig_var(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    f = d["fc_orth"]
    px = d["prices"].loc[f.index[0]:]
    var95 = f["var95_pred"].astype(float).reindex(px.index).ffill()
    r = px["ret"]
    hit = r < -var95.abs()
    ax.plot(r.index, r, color=RULE, lw=0.4)
    ax.plot(var95.index, -var95.abs(), color=NAVY, lw=1.1, zorder=4,
            label="95% VaR in force")
    ax.scatter(r.index[hit], r[hit], s=7, color=BRICK, zorder=3, label="breaches")
    ax.set_ylim(-0.30, 0.22)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Daily return against the 95% VaR in force", loc="left", color=INK)
    ax.legend(loc="lower left", ncols=2)
    _clean(ax)

    ax = axes[1]
    labels, expected, actual, colours = [], [], [], []
    for key, label, colour in (("orth", "Orth.", NAVY), ("raw", "Raw", BRICK)):
        bt = d[f"bt_{key}"]
        for lvl in ("95", "99"):
            labels.append(f"{label}\n{lvl}%")
            expected.append(float(bt[f"expected_{lvl}"]))
            actual.append(float(bt[f"exceptions_{lvl}"]))
            colours.append(colour)
    x = np.arange(len(labels))
    ax.bar(x - 0.19, expected, width=0.36, color=RULE, label="expected")
    ax.bar(x + 0.19, actual, width=0.36, color=colours, label="observed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title("VaR exceptions", loc="left", color=INK)
    ax.legend(loc="upper left", ncols=2, fontsize=7)
    ax.grid(axis="x", visible=False)
    _clean(ax)
    save(fig, "amzn-var")


def fig_risk_share(d: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    f = d["fc_orth"]
    share = f["factor_risk_share"].astype(float)
    ax.fill_between(share.index, 0, share, color=NAVY, alpha=0.75, lw=0,
                    label="explained by the factors")
    ax.fill_between(share.index, share, 1, color=RULE, alpha=0.65, lw=0,
                    label="specific to Amazon")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Share of predicted variance the forty factors account for",
                 loc="left", color=INK)
    ax.legend(loc="lower left", ncols=2)
    _clean(ax)
    save(fig, "amzn-risk-share")


def fig_blocks(d: dict, decomp: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    o, r = decomp["orth"], decomp["raw"]
    order = o["contribution"].sort_values().index
    y = np.arange(len(order))
    ax.barh(y - 0.19, o.loc[order, "contribution"], height=0.36, color=NAVY,
            label="Orthogonalised")
    ax.barh(y + 0.19, r.reindex(order)["contribution"], height=0.36, color=BRICK,
            label="Raw")
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.5)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Contribution to predicted volatility, annualised")
    ax.set_title("Where the risk sits, by block — and how differently the two "
                 "panels answer", loc="left", color=INK)
    ax.legend(loc="lower right")
    _clean(ax)
    save(fig, "amzn-blocks")


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

def macros(d: dict, decomp: dict) -> str:
    """Every figure quoted in the prose, as a LaTeX macro."""
    out: list[str] = [
        "% Generated by scripts/build_case_study.py — do not edit by hand.",
    ]

    def add(name: str, value) -> None:
        out.append(rf"\newcommand{{\{name}}}{{{value}}}")

    px = d["prices"]
    r = px["ret"]
    add("amzFirst", px.index[0].date().isoformat())
    add("amzLast", px.index[-1].date().isoformat())
    add("amzDays", f"{len(px):,}")
    add("amzVol", f"{r.std() * np.sqrt(252) * 100:.1f}\\%")
    add("amzRet", f"{(np.exp(r.sum()) ** (252 / len(r)) - 1) * 100:.1f}\\%")
    add("amzWorst", f"{r.min() * 100:.1f}\\%")
    add("amzBest", f"{r.max() * 100:.1f}\\%")
    add("amzKurt", f"{r.kurtosis():.1f}")

    for key, tag in (("orth", "O"), ("raw", "R")):
        m, f, bt = d[f"meta_{key}"], d[f"fc_{key}"], d[f"bt_{key}"]
        add(f"win{tag}", f"{len(m):,}")
        add(f"adjR{tag}", f"{m['adj_r2'].mean():.3f}")
        add(f"cond{tag}", f"{m['condition_number'].mean():.0f}")
        add(f"condMax{tag}", f"{m['condition_number'].max():,.0f}")
        add(f"vif{tag}", f"{m['max_vif'].mean():,.0f}")
        add(f"vifMax{tag}", f"{m['max_vif'].max():,.0f}")
        add(f"shift{tag}", f"{m['beta_shift_l1'].median():.1f}")
        add(f"corr{tag}", f"{m['beta_corr_prev'].mean():.3f}")
        add(f"resid{tag}", f"{m['resid_vol_ann'].mean() * 100:.1f}\\%")
        add(f"excl{tag}", f"{int((~f['is_reliable'].astype(bool)).sum())}")
        add(f"nfc{tag}", f"{int(bt['n_forecasts']):,}")
        add(f"bias{tag}", f"{float(bt['mean_bias']):.3f}")
        add(f"zstd{tag}", f"{float(bt['z_std']):.3f}")
        add(f"zkurt{tag}", f"{float(bt['z_kurtosis']):.1f}")
        add(f"mzb{tag}", f"{float(bt['mz_beta']):.3f}")
        add(f"mzp{tag}", f"{float(bt['mz_joint_p']):.3f}")
        add(f"mzr{tag}", f"{float(bt['mz_r2']):.3f}")
        add(f"exNF{tag}", f"{int(bt['exceptions_95']):,}")
        add(f"exEF{tag}", f"{float(bt['expected_95']):.0f}")
        add(f"exNN{tag}", f"{int(bt['exceptions_99']):,}")
        add(f"exEN{tag}", f"{float(bt['expected_99']):.0f}")
        add(f"kupF{tag}", _p(float(bt["kupiec_p_95"])))
        add(f"kupN{tag}", _p(float(bt["kupiec_p_99"])))
        add(f"chrF{tag}", _p(float(bt["christoffersen_p_95"])))
        add(f"sample{tag}", str(bt["sample_start"]))
        add(f"predLast{tag}", f"{float(f['sigma_pred_ann'].iloc[-1]) * 100:.1f}\\%")
        add(f"shareLast{tag}", f"{float(f['factor_risk_share'].iloc[-1]) * 100:.0f}\\%")

        lo = d[f"load_{key}"]
        last = lo[lo["window_end"] == lo["window_end"].max()].set_index("factor_id")
        for f_id, short in (("eq_global", "EqG"), ("eq_us", "EqUS"),
                            ("sty_momentum", "Mom")):
            if f_id in last.index:
                add(f"beta{short}{tag}", f"{float(last.loc[f_id, 'beta']):.2f}")
                add(f"vifOf{short}{tag}", f"{float(last.loc[f_id, 'vif']):,.0f}")

        pair = (lo[lo["factor_id"].isin(("eq_global", "eq_us"))]
                .pivot(index="window_end", columns="factor_id", values="beta")
                .astype(float))
        add(f"sdEqG{tag}", f"{pair['eq_global'].std():.2f}")
        add(f"sdEqUS{tag}", f"{pair['eq_us'].std():.2f}")
        add(f"loEqG{tag}", f"{pair['eq_global'].min():.2f}")
        add(f"hiEqG{tag}", f"{pair['eq_global'].max():.2f}")
        add(f"loEqUS{tag}", f"{pair['eq_us'].min():.2f}")
        add(f"hiEqUS{tag}", f"{pair['eq_us'].max():.2f}")
        add(f"corrPair{tag}", f"{pair['eq_global'].corr(pair['eq_us']):.2f}")

        top = decomp[key]["share"].drop("Specific", errors="ignore")
        add(f"topBlock{tag}", top.idxmax())
        add(f"topBlockShare{tag}", f"{float(top.max()) * 100:.0f}\\%")
        add(f"specShare{tag}",
            f"{float(decomp[key]['share'].get('Specific', np.nan)) * 100:.0f}\\%")

    share = d["fc_orth"]["factor_risk_share"].astype(float)
    add("shareMean", f"{share.mean() * 100:.0f}\\%")
    add("shareMajority", f"{(share > 0.5).mean() * 100:.0f}\\%")

    add("amzWindowEnd", d["meta_orth"].index[-1].date().isoformat())
    add("orthSpec", ORTH_SPEC[:12])
    add("rawSpec", RAW_SPEC[:12])
    return "\n".join(out) + "\n"


def _p(v: float) -> str:
    """A p-value for a table cell: the bare number, or a bound below 0.001."""
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


# --------------------------------------------------------------------------

def typeset() -> int:
    exe = shutil.which("pdflatex")
    if not exe:
        print("! pdflatex not found on PATH")
        return 1
    for pass_no in (1, 2):
        proc = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error",
             "--enable-installer", "amzn-case-study.tex"],
            cwd=DOCS, capture_output=True, text=True, errors="replace")
        if proc.returncode != 0:
            bad = [ln for ln in proc.stdout.splitlines() if ln.startswith("!")]
            print(f"! pdflatex failed on pass {pass_no}")
            print("\n".join(bad[-12:]) or proc.stdout[-1500:])
            return proc.returncode
        print(f"  pass {pass_no} ok")
    for junk in ("aux", "log", "out", "toc"):
        (DOCS / f"amzn-case-study.{junk}").unlink(missing_ok=True)
    pdf = DOCS / "amzn-case-study.pdf"
    print(f"  {pdf.name}: {pdf.stat().st_size / 1024:.0f} kB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the AMZN case study")
    ap.add_argument("--pdf", action="store_true", help="also run pdflatex")
    args = ap.parse_args()

    print("loading")
    d = load()
    print(f"  {len(d['prices']):,} price rows, "
          f"{len(d['meta_orth'])} windows per panel")

    decomp = {k: block_decomposition(d, k) for k in ("orth", "raw")}

    print("figures")
    fig_security(d)
    fig_loadings(d)
    fig_beta_paths(d)
    fig_fit(d)
    fig_conditioning(d)
    fig_stability(d)
    fig_forecast(d)
    fig_mz(d)
    fig_var(d)
    fig_risk_share(d)
    fig_blocks(d, decomp)

    print("numbers")
    (DOCS / "case-study-numbers.tex").write_text(macros(d, decomp), encoding="utf-8")
    print(f"  case-study-numbers.tex")

    return typeset() if args.pdf else 0


if __name__ == "__main__":
    sys.exit(main())
