"""
Stand-alone runner for the whole XAI consistency / evaluation analysis (paper Section 4).
Loads the saved FD001 weights (no training) and writes everything to ../results/fd001/.

    python run_xai_analysis.py            # full run (N=200 consistency windows, N=50 evaluation windows)
    python run_xai_analysis.py --quick    # tiny smoke test
Stages are cached; delete the corresponding file in the results folder to recompute a stage.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

import xai_analysis as XA
from rul_common import load_fd001, load_models

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = args.out or os.path.join(HERE, "..", "results", "fd001" + ("_quick" if args.quick else ""))
os.makedirs(OUT, exist_ok=True)
log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)

torch.manual_seed(42)
np.random.seed(42)
data = load_fd001()
lstm, cnn, hyb = load_models()
models, fns = XA.raw_models(lstm, cnn, hyb), XA.forward_fns(lstm, cnn, hyb)
X = data["X_val"]
n_cons = 8 if args.quick else XA.N_CONSISTENCY
n_eval = 3 if args.quick else XA.N_EVAL
idx = XA.consistency_indices(len(X), n_cons)
log(f"X_val {tuple(X.shape)}; consistency sample {len(idx)} windows; evaluation windows {n_eval}")

# --- 1. attribution maps ------------------------------------------------------------------
p_maps = os.path.join(OUT, "maps.npz")
if os.path.exists(p_maps):
    maps, idx_saved = XA.load_maps(p_maps)
    assert (idx_saved == idx).all()
    log("loaded cached maps")
else:
    log("computing attribution maps (IG, Occlusion(1,1), LRP) for CNN / LSTM / Hybrid")
    maps = XA.compute_maps(models, fns, X, idx, log=log)
    XA.save_maps(p_maps, maps, idx)

# --- 2. agreement metrics ----------------------------------------------------------------
active = [i for i in range(24) if i not in XA.CONSTANT_CHANNELS]
channel_sets = {"all24": list(range(24)), "active17": active}
p_pairs = os.path.join(OUT, "pairs.pkl")
if os.path.exists(p_pairs):
    res = pickle.load(open(p_pairs, "rb"))
else:
    log("computing agreement metrics")
    res = XA.all_pairs(maps, channel_sets)
    pickle.dump(res, open(p_pairs, "wb"))
summary = {kind: {"|".join(k): {m: XA.summarize(a) for m, a in v.items()} for k, v in res[kind].items()}
           for kind in ("within", "cross")}
json.dump(summary, open(os.path.join(OUT, "pair_summary.json"), "w"), indent=1)

# --- 3. evaluation metrics ---------------------------------------------------------------
p_eval = os.path.join(OUT, "eval.pkl")
if os.path.exists(p_eval):
    ev = pickle.load(open(p_eval, "rb"))
else:
    log("computing sensitivity / infidelity / output robustness")
    ev = XA.eval_metrics(models, fns, X, idx, maps, n_eval=n_eval, log=log)
    pickle.dump(ev, open(p_eval, "wb"))

# --- 4. activation maximization ----------------------------------------------------------
p_am = os.path.join(OUT, "actmax.npz")
if os.path.exists(p_am):
    am = dict(np.load(p_am))
else:
    log("activation maximization")
    am = {mk: XA.activation_maximization(fns[mk]) for mk in XA.MODELS}
    for c in (1.0, 2.0, 3.0):
        tag = "_con" if c == XA.AM_BOX_SD else f"_con{c:g}"
        am.update({mk + tag: XA.activation_maximization(fns[mk], bounds=XA.am_bounds(c)) for mk in XA.MODELS})
    np.savez(p_am, **am)

# --- 5. print a compact report -----------------------------------------------------------
def line(k, v):
    return f"{k:38s} " + "  ".join(f"{m}={XA.summarize(v[m])['mean']:.3f}" for m in
                                   ("spearman", "rank", "feature", "sign", "signedrank", "signedrank5"))
for set_name in channel_sets:
    print(f"\n==== WITHIN-model [{set_name}] ====")
    for k, v in res["within"].items():
        if k[0] == set_name:
            print(line("|".join(k[1:]), v))
    print(f"==== CROSS-model [{set_name}] ====")
    for k, v in res["cross"].items():
        if k[0] == set_name:
            print(line("|".join(k[1:]), v))
print("\n==== sensitivity / infidelity (mean, median, std) ====")
for k in ev["sens"]:
    s, f = XA.summarize(ev["sens"][k]), XA.summarize(ev["infid"][k])
    print(f"{k[0]:6s} {k[1]:9s} sens {s['mean']:.3f}/{s['median']:.3f}/{s['std']:.3f}   infid {f['mean']:.3f}/{f['median']:.3f}/{f['std']:.3f}")
for mk in XA.MODELS:
    s = XA.summarize(ev["outrob"][mk])
    print(f"{mk:6s} |dRUL| mean={s['mean']:.3f} median={s['median']:.3f} std={s['std']:.3f}")
from scipy.stats import spearmanr
print("\nActivation maximization cross-model Spearman (per-sensor mean over time):")
_act = [i for i in range(24) if i not in XA.CONSTANT_CHANNELS]
for a, b in (("cnn", "lstm"), ("cnn", "hybrid"), ("lstm", "hybrid")):
    print(f"  {a} vs {b}: unconstrained all24 {spearmanr(am[a].mean(0), am[b].mean(0))[0]:.3f}, "
          f"unconstrained 17 {spearmanr(am[a].mean(0)[_act], am[b].mean(0)[_act])[0]:.3f}, "
          f"box+-2SD 17 {spearmanr(am[a + '_con'].mean(0)[_act], am[b + '_con'].mean(0)[_act])[0]:.3f}")
for tag in ("_con1", "_con", "_con3"):
    print(f"  box {tag}: " + ", ".join(f"{a}-{b} {spearmanr(am[a + tag].mean(0)[_act], am[b + tag].mean(0)[_act])[0]:.3f}"
                                      for a, b in (("cnn", "lstm"), ("cnn", "hybrid"), ("lstm", "hybrid"))))
for mk in XA.MODELS:
    with torch.no_grad():
        print(f"  {mk}: predicted RUL of synthetic input, unconstrained {fns[mk](torch.tensor(am[mk])[None]).item():.1f}, "
              f"box +-2 SD {fns[mk](torch.tensor(am[mk + '_con'])[None]).item():.1f}")
log("done -> " + OUT)
