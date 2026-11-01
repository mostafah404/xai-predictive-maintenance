"""
Publication figures (vector PDF) for Section 4 of the RUL paper.
Fonts are sized for the real printed size (text width of the shared template is ~7.2 in), every axis is
labelled, and sensors are named with the symbols of Table 2 of the paper instead of "sensor_19".
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import xai_analysis as XA

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts (selectable text, no Type-3)
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})
TEXTWIDTH = 7.2  # inches
SENSOR_SYMBOLS = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30", "φ", "NRf", "NRc",
                  "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"]
CHANNEL_LABELS = ["OS1", "OS2", "OS3"] + [f"s{i + 1} {SENSOR_SYMBOLS[i]}" for i in range(21)]
SENSOR_NUM = ["OS1", "OS2", "OS3"] + [str(i + 1) for i in range(21)]   # compact axis labels (sensor number)
MODEL_COL = {"cnn": "#1f77b4", "lstm": "#d95f02", "hybrid": "#1b9e77"}
BLUE, RED = "#3b6fb6", "#c0392b"
METRICS = ["spearman", "rank", "feature", "sign", "signedrank5"]
METRIC_LABEL = {"spearman": "Spearman", "rank": "Rank@5", "feature": "Feature@5", "sign": "Sign",
                "signedrank": "Signed-rank@K", "signedrank5": "Signed-rank@5"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", bbox_inches="tight", dpi=130)  # quick-look preview only
    plt.close(fig)


def radar(rows, labels, metrics, title_note, path, ymin=-0.4):
    """rows: list of dict metric->value (one per spoke). One polygon per metric."""
    n = len(rows)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(5.6, 6.6))
    ax = fig.add_axes([0.26, 0.31, 0.48, 0.44], polar=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]
    for m, c, mk in zip(metrics, colors, markers):
        v = [r[m] for r in rows]
        v = np.concatenate([v, v[:1]])
        ax.plot(ang_c, v, color=c, lw=2, marker=mk, ms=6.5, label=METRIC_LABEL[m])
    ax.set_ylim(ymin, 1.0)
    ticks = [t for t in (0, 0.5, 1.0)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=13, bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85))
    ax.set_rlabel_position(360 / n / 2)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis="x", pad=30)
    ax.spines["polar"].set_visible(True)
    ax.grid(lw=0.6, alpha=0.6)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.zeros(200), color="k", lw=1.0, alpha=0.8)  # zero ring
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=13, frameon=False)
    if title_note:
        fig.text(0.5, 0.97, title_note, ha="center", va="top", fontsize=13)
    _save(fig, path)


def _fmt_pair(a, b):
    short = {"occlusion": "Occ.", "ig": "IG", "lrp": "LRP"}
    return f"{short[a]}–{short[b]}"


def radar_plots(res, set_name, outdir):
    # within-model: 9 spokes (model x method pair)
    rows, labels = [], []
    for mk in XA.MODELS:
        for a, b in (("ig", "occlusion"), ("ig", "lrp"), ("occlusion", "lrp")):
            mets = res["within"][(set_name, mk, a, b)]
            rows.append({m: float(np.nanmean(mets[m])) for m in METRICS})
            labels.append(f"{XA.MODEL_LABEL[mk]}\n{_fmt_pair(a, b)}")
    radar(rows, labels, METRICS, None, os.path.join(outdir, "radar_within_model"))
    # cross-model: 9 spokes (method x architecture pair)
    rows, labels = [], []
    for me in XA.METHODS:
        for k1, k2 in (("cnn", "lstm"), ("cnn", "hybrid"), ("lstm", "hybrid")):
            mets = res["cross"][(set_name, me, k1, k2)]
            rows.append({m: float(np.nanmean(mets[m])) for m in METRICS})
            labels.append(f"{XA.METHOD_LABEL[me]}\n{XA.MODEL_LABEL[k1]}–{XA.MODEL_LABEL[k2]}")
    radar(rows, labels, METRICS, None, os.path.join(outdir, "radar_cross_model"))


PANEL = (2.45, 2.1)          # inches; every sub-figure is included at ~1/3 of the text width
FS = dict(label=10, tick=9)


def _panel_style(ax, xlabel, ylabel=None):
    ax.set_xlabel(xlabel, fontsize=FS["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])


def hist_panels(res, set_name, outdir, within=("cnn", "ig", "occlusion"), cross=("ig", "cnn", "lstm")):
    """one vector file per metric and per comparison: hist_within_<metric>.pdf, hist_cross_<metric>.pdf; returns the means"""
    stats = {}
    for kind, mets in (("within", res["within"][(set_name,) + within]), ("cross", res["cross"][(set_name,) + cross])):
        for m in METRICS:
            v = np.asarray(mets[m], float)
            v = v[~np.isnan(v)]
            lo = -1 if m == "spearman" else 0
            fig, ax = plt.subplots(figsize=PANEL)
            ax.hist(v, bins=np.linspace(lo, 1, 21), color=BLUE if kind == "within" else RED, edgecolor="white", lw=0.4)
            ax.set_xlim(lo - 0.03, 1.03)
            ax.axvline(v.mean(), color="k", lw=1.0, ls="--")
            _panel_style(ax, "Agreement value", "Number of windows")
            fig.subplots_adjust(left=0.27, bottom=0.25, right=0.97, top=0.97)
            _save(fig, os.path.join(outdir, f"hist_{kind}_{m}"))
            stats[f"{kind}_{m}"] = float(v.mean())
    return stats


def heatmap_grid(maps, outdir, channels=None):
    ch = channels if channels is not None else [i for i in range(24) if i not in XA.CONSTANT_CHANNELS]
    fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, 7.9))
    for r, mk in enumerate(XA.MODELS):
        for c, me in enumerate(XA.METHODS):
            ax = axes[r, c]
            A = np.abs(maps[(mk, me)]).mean(0)[:, ch].T      # (channels, time)
            im = ax.imshow(A, aspect="auto", cmap="Reds", interpolation="nearest", vmin=0)
            ax.set_title(f"{XA.MODEL_LABEL[mk]} – {XA.METHOD_LABEL[me]}", fontsize=10.5, pad=4)
            ax.set_xticks([0, 10, 20, 30, 40, 49])
            ax.set_xticklabels([1, 11, 21, 31, 41, 50])
            ax.set_yticks(range(len(ch)))
            ax.set_yticklabels([SENSOR_NUM[i] for i in ch] if c == 0 else [], fontsize=9)
            ax.tick_params(axis="x", labelsize=9.5)
            if r == 2:
                ax.set_xlabel("Time step", fontsize=10.5)
            if c == 0:
                ax.set_ylabel("Sensor", fontsize=10.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            cb.ax.tick_params(labelsize=9)
            if c == 2:
                cb.set_label("Mean |attribution|", fontsize=10)
            for s_ in ax.spines.values():
                s_.set_visible(True)
    fig.subplots_adjust(hspace=0.22, wspace=0.3, left=0.07, right=0.94, top=0.96, bottom=0.05)
    _save(fig, os.path.join(outdir, "agg_heatmap_grid"))


def actmax_panels(am, outdir, tag="_con", box=XA.AM_BOX_SD):
    """one vector file per model: actmax_<model>.pdf (box-constrained activation maximization, common x scale)"""
    y = np.arange(24)
    const = set(XA.CONSTANT_CHANNELS)
    for mk in XA.MODELS:
        v = am[mk + tag].mean(0)
        fig, ax = plt.subplots(figsize=(2.45, 4.6))
        cols = ["#bbbbbb" if i in const else (BLUE if v[i] >= 0 else RED) for i in range(24)]
        ax.barh(y, v, color=cols, height=0.75)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(SENSOR_NUM, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(-box * 1.05, box * 1.05)
        _panel_style(ax, "Optimized value", "Sensor")
        fig.subplots_adjust(left=0.24, bottom=0.1, right=0.97, top=0.99)
        _save(fig, os.path.join(outdir, f"actmax_{mk}"))


def eval_hists(ev, outdir):
    for key, name, fname in (("sens", "Sensitivity", "eval_sensitivity_hist"), ("infid", "Infidelity", "eval_infidelity_hist")):
        fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH, 6.6))
        for r, mk in enumerate(XA.MODELS):
            for c, me in enumerate(XA.METHODS):
                ax = axes[r, c]
                v = ev[key][(mk, me)]
                ax.hist(v, bins=14, color=BLUE if key == "sens" else RED, edgecolor="white", lw=0.4)
                ax.axvline(np.mean(v), color="k", lw=1.0, ls="--")
                ax.set_title(f"{XA.MODEL_LABEL[mk]} – {XA.METHOD_LABEL[me]}\nmean = {np.mean(v):.2f}, std = {np.std(v):.2f}", fontsize=9.5)
                ax.set_xlabel(name, fontsize=10)
                ax.tick_params(labelsize=9)
                if c == 0:
                    ax.set_ylabel("Number of windows", fontsize=10)
        fig.subplots_adjust(hspace=0.85, wspace=0.32, top=0.93, bottom=0.08, left=0.09, right=0.98)
        _save(fig, os.path.join(outdir, fname))


def outrob_panels(ev, outdir):
    stats = {}
    for mk in XA.MODELS:
        v = ev["outrob"][mk]
        fig, ax = plt.subplots(figsize=PANEL)
        ax.hist(v, bins=30, color="#e8806c", edgecolor="white", lw=0.4)
        ax.axvline(v.mean(), color="k", lw=1.0, ls="--")
        _panel_style(ax, r"$|\Delta$RUL$|$ (cycles)", "Number of perturbations")
        fig.subplots_adjust(left=0.27, bottom=0.25, right=0.97, top=0.97)
        _save(fig, os.path.join(outdir, f"output_noise_{mk}"))
        stats[mk] = (float(v.mean()), float(np.median(v)))
    return stats
