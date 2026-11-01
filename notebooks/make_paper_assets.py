"""
Builds every figure (vector PDF) and every LaTeX table body of paper Section 4 from the cached results of
run_xai_analysis.py.   python make_paper_assets.py [--set active17|all24]
Output (default folder ../results/paper): figures/*.pdf and *.png, tables.tex, results_summary.md, figure_stats.json
"""
import argparse
import os
import pickle

import numpy as np

import xai_analysis as XA
import xai_plots as P

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="active17", choices=["active17", "all24"])
ap.add_argument("--results", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
RES = args.results or os.path.join(HERE, "..", "results", "fd001")
OUT = args.out or os.path.join(HERE, "..", "results", "paper")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

maps, idx = XA.load_maps(os.path.join(RES, "maps.npz"))
res = pickle.load(open(os.path.join(RES, "pairs.pkl"), "rb"))
ev = pickle.load(open(os.path.join(RES, "eval.pkl"), "rb"))
am = dict(np.load(os.path.join(RES, "actmax.npz")))
S = args.set

# ------------------------------------------------------------------ figures
P.radar_plots(res, S, FIG)
import json
fig_stats = {"hist": P.hist_panels(res, S, FIG)}
P.heatmap_grid(maps, FIG)
P.actmax_panels(am, FIG)
P.eval_hists(ev, FIG)
fig_stats["outrob"] = P.outrob_panels(ev, FIG)
json.dump(fig_stats, open(os.path.join(OUT, "figure_stats.json"), "w"), indent=1)

# ------------------------------------------------------------------ tables
def mean(v):
    return float(np.nanmean(v))


def fmt_table(rows, cols, label_cols, bold_max=True):
    """rows: list of (label tuple, {col: value}); bold the column maxima (as in the draft)"""
    best = {c: max(r[1][c] for r in rows) for c in cols}
    lines = []
    for lab, vals in rows:
        cells = []
        for c in cols:
            s = f"{vals[c]:.3f}"
            cells.append(f"\\textbf{{{s}}}" if bold_max and abs(vals[c] - best[c]) < 5e-4 else s)
        lines.append(" & ".join(list(lab) + cells) + r" \\")
    return "\n".join(lines)


pair_name = {"occlusion": "Occlusion", "ig": "IG", "lrp": "LRP"}
within_rows, cross_rows = [], []
for mk in XA.MODELS:
    for a, b in (("ig", "occlusion"), ("ig", "lrp"), ("occlusion", "lrp")):
        m = res["within"][(S, mk, a, b)]
        within_rows.append(((XA.MODEL_LABEL[mk], f"{pair_name[a]} vs.\\ {pair_name[b]}"), {k: mean(m[k]) for k in P.METRICS}))
for me in XA.METHODS:
    for k1, k2 in (("cnn", "lstm"), ("cnn", "hybrid"), ("lstm", "hybrid")):
        m = res["cross"][(S, me, k1, k2)]
        cross_rows.append(((pair_name[me], f"{XA.MODEL_LABEL[k1]} vs.\\ {XA.MODEL_LABEL[k2]}"), {k: mean(m[k]) for k in P.METRICS}))

ev_rows = []
for mk in XA.MODELS:
    for me in XA.METHODS:
        s, f = ev["sens"][(mk, me)], ev["infid"][(mk, me)]
        ev_rows.append(((XA.MODEL_LABEL[mk], pair_name[me]),
                        {"sm": s.mean(), "ss": s.std(), "im": f.mean(), "is": f.std()}))
best_low = {c: min(r[1][c] for r in ev_rows) for c in ("sm", "ss", "im", "is")}
ev_lines = []
for lab, v in ev_rows:
    cells = [(f"\\textbf{{{v[c]:.3f}}}" if abs(v[c] - best_low[c]) < 5e-4 else f"{v[c]:.3f}") for c in ("sm", "ss", "im", "is")]
    ev_lines.append(" & ".join(list(lab) + cells) + r" \\")

with open(os.path.join(OUT, "tables.tex"), "w") as fh:
    fh.write(f"% channel set: {S}\n% ---- within-model (Spearman, Rank@5, Feature@5, Sign, Signed-rank)\n")
    fh.write(fmt_table(within_rows, P.METRICS, 2) + "\n\n% ---- cross-model\n")
    fh.write(fmt_table(cross_rows, P.METRICS, 2) + "\n\n% ---- sensitivity / infidelity\n")
    fh.write("\n".join(ev_lines) + "\n")

# ------------------------------------------------------------------ plain-text summary for checking
with open(os.path.join(OUT, "results_summary.md"), "w") as fh:
    for set_name in ("all24", "active17"):
        fh.write(f"\n## Agreement metrics, channel set = {set_name}\n\n| comparison | " + " | ".join(P.METRIC_LABEL[m] for m in P.METRICS) + " | Signed-rank@5 | Rank@K |\n|---|" + "---|" * 7 + "\n")
        for kind in ("within", "cross"):
            for key, mets in res[kind].items():
                if key[0] != set_name:
                    continue
                vals = [mean(mets[m]) for m in P.METRICS] + [mean(mets["signedrank5"]), mean(mets["rankK"])]
                fh.write(f"| {kind}: {' / '.join(key[1:])} | " + " | ".join(f"{v:.3f}" for v in vals) + " |\n")
    fh.write("\n## Sensitivity / Infidelity (mean, median, std)\n\n")
    for k in ev["sens"]:
        s, f = XA.summarize(ev["sens"][k]), XA.summarize(ev["infid"][k])
        fh.write(f"- {k[0]} {k[1]}: sens {s['mean']:.3f}/{s['median']:.3f}/{s['std']:.3f}; infid {f['mean']:.3f}/{f['median']:.3f}/{f['std']:.3f}\n")
    from scipy.stats import spearmanr
    act = [i for i in range(24) if i not in XA.CONSTANT_CHANNELS]
    fh.write("\n## Activation maximization, Spearman of per-sensor means: unconstrained 17 ch / unconstrained 24 ch / box +-1, +-2, +-3 SD (17 ch)\n")
    for a, b in (("cnn", "lstm"), ("cnn", "hybrid"), ("lstm", "hybrid")):
        box = " / ".join(f"{spearmanr(am[a + t].mean(0)[act], am[b + t].mean(0)[act])[0]:.3f}" for t in ("_con1", "_con", "_con3"))
        fh.write(f"- {a} vs {b}: {spearmanr(am[a].mean(0)[act], am[b].mean(0)[act])[0]:.3f} / {spearmanr(am[a].mean(0), am[b].mean(0))[0]:.3f} / {box}\n")
    for mk in XA.MODELS:
        s = XA.summarize(ev["outrob"][mk])
        fh.write(f"- output robustness {mk}: mean {s['mean']:.3f}, median {s['median']:.3f}, std {s['std']:.3f}\n")
print("assets written to", OUT)
