"""
Compares the numbers in the tables of the paper (main.tex) with the numbers computed by this repository.
    python verify_paper_numbers.py [path/to/main.tex]        (default: ../paper/main.tex)
Needs the cached results in ../results/fd001 (run reproduce_paper_results.ipynb or run_xai_analysis.py first).
"""
import collections
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "paper", "main.tex")
RES = os.path.join(HERE, "..", "results", "fd001")
if not os.path.exists(TEX):
    print(f"{TEX} not found. The paper text is not part of this repository, so there is nothing to compare.")
    sys.exit(0)
T = open(TEX).read()
tot = {}


def table(label):
    i = T.index("\\label{" + label + "}")
    j = T.index("\\end{tabular}", i)
    body = re.sub(r"\\(toprule|midrule|bottomrule)", "", T[i:j])
    return [[re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).replace("\\%", "").replace("\\", "").strip() for c in r.split("&")]
            for r in body.split("\\\\") if "&" in r]


def tally(name, n, bad):
    a, b = tot.get(name, (0, 0))
    tot[name] = (a + n, b + bad)


res = pickle.load(open(f"{RES}/pairs.pkl", "rb"))
ev = pickle.load(open(f"{RES}/eval.pkl", "rb"))
val = json.load(open(f"{RES}/lrp_validation.json"))
perf = json.load(open(f"{RES}/performance_checkpoints.json"))
names = {"IG": "ig", "Occlusion": "occlusion", "LRP": "lrp"}
mean = lambda a: float(np.nanmean(a))

for c in table("tab:within-model"):
    if c[0] in ("CNN", "LSTM", "Hybrid"):
        a, b = [names[x.strip()] for x in c[1].replace("vs.", "vs").split("vs")]
        r = res["within"][("active17", c[0].lower(), a, b)]
        e = [mean(r[k]) for k in ("spearman", "rank", "feature", "sign", "signedrank5")]
        tally("Table 6", 5, sum(abs(x - float(y)) > 6e-4 for x, y in zip(e, c[2:7])))
for c in table("tab:cross-model"):
    if c[0] in names:
        k1, k2 = [x.strip().lower() for x in c[1].replace("vs.", "vs").split("vs")]
        r = res["cross"][("active17", names[c[0]], k1, k2)]
        e = [mean(r[k]) for k in ("spearman", "rank", "feature", "sign", "signedrank5")]
        tally("Table 7", 5, sum(abs(x - float(y)) > 6e-4 for x, y in zip(e, c[2:7])))
for c in table("tab:eval-metrics"):
    if c[0] in ("CNN", "LSTM", "Hybrid"):
        k = (c[0].lower(), names[c[1]])
        s, f = ev["sens"][k], ev["infid"][k]
        e = [s.mean(), s.std(), f.mean(), f.std()]
        tally("Table 8", 4, sum(abs(x - float(y)) > 6e-4 for x, y in zip(e, c[2:6])))
for c in table("tab:lrp-checks"):
    if c[0] in ("CNN", "LSTM", "Hybrid"):
        mk, meth = c[0].lower(), c[1].strip()
        key = {"IG": "IG", "Occlusion": "Occlusion", "Random ranking": "random"}.get(meth)
        if key is None:
            eps = re.search(r"varepsilon\s*=\s*(.*)", meth.replace("$", "")).group(1).strip()
            key = "LRP eps=" + {"10^{-3}": "0.001", "10{-3}": "0.001"}.get(eps, eps)
        r = val[mk][key]
        e = [r["del5"], r["del10"]] + ([r["spearman_ig"], r["spearman_occ"], r["conserv_corr"], r["early3"]] if key.startswith("LRP") else [])
        g = [float(x) for x in c[2:2 + len(e)] if x not in ("--", "", "–")]
        ok = len(g) == len(e) and all(abs(x - y) < (0.051 if i < 2 else 0.0051) for i, (x, y) in enumerate(zip(e, g)))
        tally("Table 9", len(e), 0 if ok else len(e))

rows = {c[0].strip(): c[1:] for c in table("tab:performance")}
key4 = {"RMSE (all)": "rmse", "MAE (all)": "mae", "RMSE (HIGH)": "rmse_high", "MAE (HIGH)": "mae_high", "Accuracy": "acc"}
for name, k in key4.items():
    e = [perf[m]["val"][k] for m in ("lstm", "cnn", "hybrid")] + [perf[m]["test"][k] for m in ("lstm", "cnn", "hybrid")]
    tally("Table 4", 6, sum(abs(float(g) - x) > 0.051 for g, x in zip(rows[name], e)))
tally("Table 4", 3, sum(abs(float(g) - perf[m]["test"]["s_mean"]) > 0.006 for g, m in zip(rows["S-score (mean)"][3:], ("lstm", "cnn", "hybrid"))))

# FD002-FD004 of Table 5 come from the multi-dataset run, whose output is saved in data_preprocessing.ipynb
nb = json.load(open(os.path.join(HERE, "data_preprocessing.ipynb")))
text = ""
for cell in nb["cells"]:
    outs = ["".join(o.get("text", [])) + "".join(o.get("data", {}).get("text/plain", [])) for o in cell.get("outputs", [])]
    if any("Full Cross-Dataset Results" in t for t in outs):
        text = "\n".join(outs)
multi = {}
for line in text.split("\n"):
    m = re.match(r"\d+\s+(FD00\d)\s+Test\s+(LSTM|CNN|HYBRID)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", line)
    if m:
        multi[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
for c in table("tab:cross-dataset"):
    if c[0].strip().startswith("FD"):
        fd = c[0].strip()
        if fd == "FD001":
            e = [perf[m]["test"]["rmse"] for m in ("lstm", "cnn", "hybrid")] + [perf[m]["test"]["acc"] for m in ("lstm", "cnn", "hybrid")]
        else:
            e = [multi[(fd, m)][0] for m in ("LSTM", "CNN", "HYBRID")] + [multi[(fd, m)][1] for m in ("LSTM", "CNN", "HYBRID")]
        tally("Table 5", 6, sum(abs(float(g) - x) > 0.051 for g, x in zip(c[1:7], e)))

df = pd.read_csv(os.path.join(HERE, "..", "data", "AI Predictive Maintenance.csv"))
cols = df.columns.tolist()
D = df[[cols[1], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9]]].values.astype(float)
t10 = [float(r[2]) for r in table("tab:trust-progression") if r[0].strip().isdigit()]
tally("Table 10", 7, sum(abs(a - round(b, 2)) > 1e-9 for a, b in zip(t10, D.mean(0))))
borda = collections.Counter()
for _, r in df.iterrows():
    for pos, item in enumerate([str(r[x]).strip() for x in cols[10:16]]):
        borda[item] += 6 - pos
t11 = [int(r[1]) for r in table("tab:borda") if r[1].strip().isdigit()]
tally("Table 11", 6, sum(a != b for a, b in zip(t11, [v for _, v in borda.most_common()])))

# Table 2: trend arrow of every sensor
raw = pd.read_csv(os.path.join(HERE, "..", "data", "train_FD001.txt"), sep=r"\s+", header=None,
                  names=["machine_id", "cycle"] + ["OS1", "OS2", "OS3"] + [f"s{k}" for k in range(1, 22)])
raw["frac"] = raw.groupby("machine_id")["cycle"].transform(lambda c: c / c.max())
paper = {}
for r in table("tab:sensors"):
    if r[0].strip().isdigit():
        paper[int(r[0])] = "up" if "uparrow" in r[4] else "down" if "downarrow" in r[4] else "~"
bad = 0
for s in range(1, 22):
    x = raw[f"s{s}"]
    if x.std() < 1e-9:
        d = "~"
    else:
        ch = (raw[raw.frac >= 0.8][f"s{s}"].mean() - raw[raw.frac <= 0.2][f"s{s}"].mean()) / x.std()
        d = "up" if ch > 0.5 else "down" if ch < -0.5 else "~"
    bad += d != paper[s]
tally("Table 2 (trend arrows)", 21, bad)

print(f"paper: {TEX}\n")
total_bad = 0
for k, (n, b) in tot.items():
    print(f"{k:24s} {n:3d} numbers checked, {b} mismatches")
    total_bad += b
print("\nALL NUMBERS MATCH" if total_bad == 0 else f"\n{total_bad} MISMATCHES")
