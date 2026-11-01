"""
Extra checks for the LRP implementation (paper Section 3.4 / Appendix):
  (1) deletion test: zero the top-p % attributed inputs of every window and measure the mean |change in predicted RUL|
      (IG, Occlusion, LRP at several epsilons, random ranking as the reference)
  (2) epsilon sensitivity of LRP: Spearman / Feature@5 agreement with IG and Occlusion, relevance conservation
      (correlation of the summed relevance with the prediction) and the share of |relevance| in the first 3 time steps.
Uses the cached IG / Occlusion maps of run_xai_analysis.py.   python lrp_validation.py
Writes ../results/fd001/lrp_validation.json and ../results/paper/lrp_validation.tex
"""
import json
import os

import numpy as np
import torch

import lrp_recurrent as L
import xai_analysis as XA
from rul_common import load_fd001, load_models

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "fd001")
OUT = os.path.join(HERE, "..", "results", "paper")
os.makedirs(OUT, exist_ok=True)
EPSS = [1e-3, 1e-2, 1e-1, 1.0]
FRACS = [0.05, 0.10]
ACTIVE = [i for i in range(24) if i not in XA.CONSTANT_CHANNELS]

data = load_fd001()
lstm, cnn, hyb = load_models()
models, fns = XA.raw_models(lstm, cnn, hyb), XA.forward_fns(lstm, cnn, hyb)
maps, idx = XA.load_maps(os.path.join(RES, "maps.npz"))
X = data["X_val"][idx]
n, T, C = X.shape
g = torch.Generator()
g.manual_seed(7)
rand = torch.randn(X.shape, generator=g).numpy()


def deletion(f, A, p):
    flat = torch.from_numpy(np.abs(A)).reshape(n, -1)
    top = flat.topk(int(p * T * C), dim=1).indices
    Xd = X.clone().reshape(n, -1)
    Xd.scatter_(1, top, 0.0)
    with torch.no_grad():
        y0, y1 = f(X).squeeze(), f(Xd.reshape_as(X)).squeeze()
    return float((y1 - y0).abs().mean())


def prof_spearman(a, b):
    pa, pb = a.mean(1), b.mean(1)
    return float(np.nanmean([XA.agreement(pa[i, ACTIVE], pb[i, ACTIVE])["spearman"] for i in range(n)])), \
        float(np.mean([XA.agreement(pa[i, ACTIVE], pb[i, ACTIVE])["feature"] for i in range(n)]))


out = {}
for mk in XA.MODELS:
    f = fns[mk]
    with torch.no_grad():
        y = f(X).squeeze().numpy()
    cand = {"IG": maps[(mk, "ig")], "Occlusion": maps[(mk, "occlusion")]}
    for e in EPSS:
        cand[f"LRP eps={e:g}"] = L.lrp(models[mk], X, eps=e).detach().float().numpy()
    cand["random"] = rand
    rows = {}
    for k, A in cand.items():
        r = {f"del{int(p * 100)}": deletion(f, A, p) for p in FRACS}
        tp = np.abs(A).sum((0, 2))
        r["early3"] = float(tp[:3].sum() / tp.sum())
        r["conserv_corr"] = float(np.corrcoef(A.sum((1, 2)), y)[0, 1]) if k != "random" else None
        if k.startswith("LRP"):
            r["spearman_ig"], r["feature_ig"] = prof_spearman(A, maps[(mk, "ig")])
            r["spearman_occ"], r["feature_occ"] = prof_spearman(A, maps[(mk, "occlusion")])
        rows[k] = r
        print(mk, k, {a: (round(b, 3) if b is not None else None) for a, b in r.items()}, flush=True)
    out[mk] = rows
json.dump(out, open(os.path.join(RES, "lrp_validation.json"), "w"), indent=1)

with open(os.path.join(OUT, "lrp_validation.tex"), "w") as fh:
    fh.write("% deletion test (mean |change in RUL| in cycles after zeroing top-5% / top-10% inputs) and LRP epsilon sensitivity\n")
    for mk in XA.MODELS:
        for k, r in out[mk].items():
            cells = [f"{r['del5']:.1f}", f"{r['del10']:.1f}"]
            if k.startswith("LRP"):
                cells += [f"{r['spearman_ig']:.2f}", f"{r['spearman_occ']:.2f}", f"{r['conserv_corr']:.2f}", f"{r['early3']:.2f}"]
            else:
                cells += ["--"] * 4
            fh.write(" & ".join([XA.MODEL_LABEL[mk], k.replace("eps=", "$\\varepsilon$=")] + cells) + " \\\\\n")
print("written")
