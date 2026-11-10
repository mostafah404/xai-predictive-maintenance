"""
Cross-method / cross-model XAI consistency analysis for the RUL paper (CMAPSS FD001).

The numbers in Section 4 of the paper are computed here.  Used both by the notebook
(`data_preprocessing.ipynb`, last section) and by the stand-alone runner `run_xai_analysis.py`.

Methods : IG (zero baseline, 50 steps), Occlusion (1,1) with zero baseline, LRP (own implementation,
          lrp_recurrent.py, eps-rule + Arras et al. rules for LSTM) -- LRP is applied to all three models.
Models  : CNN, LSTM + attention, hybrid CNN-LSTM-BiLSTM.
"""
import json
import os
import time
from itertools import combinations

import numpy as np
import torch
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients, Occlusion

import lrp_recurrent as L
from rul_common import lstm_rul

MODELS = ["cnn", "lstm", "hybrid"]
METHODS = ["ig", "occlusion", "lrp"]
METHOD_LABEL = {"ig": "IG", "occlusion": "Occlusion", "lrp": "LRP"}
MODEL_LABEL = {"cnn": "CNN", "lstm": "LSTM", "hybrid": "Hybrid"}
LRP_EPS = 1e-3          # epsilon of the LRP epsilon-rule (Arras et al. 2017 use 1e-3)
N_CONSISTENCY = 200     # windows in the consistency sample
N_EVAL = 50             # windows used for sensitivity / infidelity / output robustness
AM_BOX_SD = 2.0         # constrained activation maximization: every channel stays within +-AM_BOX_SD standard deviations
SIGMA = 0.05            # std of the Gaussian input noise
N_PERTURB_SENS, N_PERTURB_INFID, N_PERTURB_OUT = 3, 5, 10
CONSTANT_CHANNELS = [2, 3, 7, 12, 18, 20, 21]   # operating_setting_3, sensors 1, 5, 10, 16, 18, 19 (constant in FD001)


def forward_fns(lstm, cnn, hybrid):
    return {"cnn": cnn, "lstm": lstm_rul(lstm), "hybrid": hybrid}


def raw_models(lstm, cnn, hybrid):
    return {"cnn": cnn, "lstm": lstm, "hybrid": hybrid}


def consistency_indices(n_val, n=N_CONSISTENCY, seed=42):
    np.random.seed(seed)
    return np.random.choice(n_val, size=min(n, n_val), replace=False)


# attribution maps
def attribute(model_kind, method, models, fns, x, occ_batch=100, lrp_eps=LRP_EPS):
    """x: (B,50,24) float32 -> attribution (B,50,24) float32 numpy"""
    f = fns[model_kind]
    if method == "ig":
        out = IntegratedGradients(f).attribute(x, baselines=torch.zeros_like(x), n_steps=50,
                                               internal_batch_size=1000)
    elif method == "occlusion":
        out = Occlusion(f).attribute(x, baselines=0, sliding_window_shapes=(1, 1), strides=(1, 1),
                                     perturbations_per_eval=occ_batch)
    elif method == "lrp":
        out = L.lrp(models[model_kind], x, eps=lrp_eps)
    else:
        raise ValueError(method)
    return out.detach().float().numpy()


def compute_maps(models, fns, X, idx, chunk=10, log=print, **kw):
    maps = {}
    for mk in MODELS:
        for meth in METHODS:
            t0 = time.time()
            parts = [attribute(mk, meth, models, fns, X[idx[s:s + chunk]], **kw)
                     for s in range(0, len(idx), chunk)]
            maps[(mk, meth)] = np.concatenate(parts)
            log(f"  maps {mk:6s} {meth:9s} done in {time.time() - t0:5.0f}s")
    return maps


# agreement metrics  (definitions follow Eqs. 3-6 of the paper / Krishna et al. 2022)
def _order(v):
    """sensors sorted by decreasing |attribution|; stable so that exact ties are broken by channel index"""
    return np.argsort(-np.abs(v), kind="stable")


def agreement(a, b, k=5):
    """
    a, b : per-sensor profiles (1-D, same length K).  Returns dict of metrics for one pair.
    rank       : fraction of the k top positions occupied by the same sensor            (Eq. 3)
    feature    : |Top_k(a) & Top_k(b)| / k                                              (Eq. 4)
    sign       : sensors identifying the same sign among the K (all) sensors, / K       (Eq. 5, k=K)
    signedrank : positions j<=kk with the same sensor AND the same sign / kk            (Eq. 6) for kk=K and kk=5
    """
    K = len(a)
    oa, ob = _order(a), _order(b)
    out = {}
    out["rank"] = float(np.mean(oa[:k] == ob[:k]))
    out["feature"] = len(set(oa[:k]) & set(ob[:k])) / k
    out["sign"] = float(np.mean(np.sign(a) == np.sign(b)))
    for name, kk in (("signedrank", K), ("signedrank5", 5)):
        same = oa[:kk] == ob[:kk]
        sgn = np.sign(a[oa[:kk]]) == np.sign(b[ob[:kk]])
        out[name] = float(np.mean(same & sgn))
    out["rankK"] = float(np.mean(oa == ob))
    if np.std(a) == 0 or np.std(b) == 0:
        out["spearman"] = np.nan
    else:
        out["spearman"] = float(spearmanr(a, b)[0])
    return out


def profile(map_):
    """(N,50,24) -> (N,24) time-averaged signed attribution per sensor"""
    return map_.mean(axis=1)


def pair_metrics(pa, pb, channels):
    rows = [agreement(pa[i, channels], pb[i, channels]) for i in range(len(pa))]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def summarize(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(arr.std())}


def all_pairs(maps, channel_sets):
    """within-model (method pairs per model) and cross-model (model pairs per method) metrics"""
    prof = {k: profile(v) for k, v in maps.items()}
    res = {"within": {}, "cross": {}}
    for set_name, ch in channel_sets.items():
        for mk in MODELS:
            for m1, m2 in combinations(METHODS, 2):
                res["within"][(set_name, mk, m1, m2)] = pair_metrics(prof[(mk, m1)], prof[(mk, m2)], ch)
        for meth in METHODS:
            for k1, k2 in combinations(MODELS, 2):
                res["cross"][(set_name, meth, k1, k2)] = pair_metrics(prof[(k1, meth)], prof[(k2, meth)], ch)
    return res


# evaluation metrics (Yeh et al. 2019 style) with fixed random seeds
def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def eval_metrics(models, fns, X, idx, base_maps, n_eval=N_EVAL, log=print, occ_batch=100):
    """Sensitivity, infidelity and output robustness on the first n_eval windows of the sample."""
    pos = range(min(n_eval, len(idx)))
    sens = {(mk, me): [] for mk in MODELS for me in METHODS}
    sens_abs = {k: [] for k in sens}
    sens_norm = {k: [] for k in sens}
    infid = {k: [] for k in sens}
    outrob = {mk: [] for mk in MODELS}
    for mk in MODELS:
        f = fns[mk]
        t0 = time.time()
        for p in pos:
            x = X[idx[p]:idx[p] + 1]
            with torch.no_grad():
                y0 = f(x).item()
            # ---- sensitivity: attributions of x + delta_j
            for me in METHODS:
                base = base_maps[(mk, me)][p]
                nb = np.linalg.norm(base) + 1e-12
                worst, worst_abs = 0.0, 0.0
                for j in range(N_PERTURB_SENS):
                    d = torch.randn(x.shape, generator=_gen(10_000 + 97 * p + 13 * j)) * SIGMA
                    pert = attribute(mk, me, models, fns, x + d, occ_batch=occ_batch)[0]
                    diff = np.linalg.norm(pert - base)
                    worst = max(worst, diff / nb)
                    worst_abs = max(worst_abs, diff / (np.linalg.norm(d.numpy()) + 1e-12))
                sens[(mk, me)].append(worst)
                sens_abs[(mk, me)].append(worst_abs)
                sens_norm[(mk, me)].append(nb)
            # ---- infidelity: E[(Attr^T delta - (f(x) - f(x - delta)))^2]
            ds = [torch.randn(x.shape, generator=_gen(20_000 + 97 * p + 13 * j)) * SIGMA for j in range(N_PERTURB_INFID)]
            with torch.no_grad():
                ys = [f(x - d).item() for d in ds]
            for me in METHODS:
                base = base_maps[(mk, me)][p]
                errs = [(float((base * d.numpy()[0]).sum()) - (y0 - yj)) ** 2 for d, yj in zip(ds, ys)]
                infid[(mk, me)].append(float(np.mean(errs)))
            # ---- output robustness: |f(x + delta) - f(x)|
            dd = [torch.randn(x.shape, generator=_gen(30_000 + 97 * p + 13 * j)) * SIGMA for j in range(N_PERTURB_OUT)]
            with torch.no_grad():
                outrob[mk] += [abs(f(x + d).item() - y0) for d in dd]
        log(f"  eval metrics {mk:6s} done in {time.time() - t0:5.0f}s")
    to_np = lambda d: {k: np.array(v) for k, v in d.items()}
    return {"sens": to_np(sens), "sens_abs": to_np(sens_abs), "attr_norm": to_np(sens_norm),
            "infid": to_np(infid), "outrob": to_np(outrob)}


# activation maximization (zero init, Adam, 500 steps, lr 0.05) -- identical to the notebook
def am_bounds(c, n_feat=24):
    """symmetric box of +-c standard deviations (the inputs are z-scored); constant channels are fixed at 0"""
    hi = np.full(n_feat, c, dtype=np.float32)
    hi[CONSTANT_CHANNELS] = 0
    return -hi, hi


def activation_maximization(fn, n_steps=500, lr=0.05, n_feat=24, window=50, bounds=None):
    """
    Gradient ascent on the input, zero start, Adam.  bounds=(lo, hi) (each shape (n_feat,)) restricts the synthetic
    input to the range seen in the training data (projected gradient ascent: after every step the values are clipped
    to [lo, hi] per channel).  bounds=None is the unconstrained version.
    """
    syn = torch.zeros(1, window, n_feat, requires_grad=True)
    opt = torch.optim.Adam([syn], lr=lr)
    if bounds is not None:
        lo, hi = (torch.as_tensor(b, dtype=torch.float32).view(1, 1, -1) for b in bounds)
    for _ in range(n_steps):
        opt.zero_grad()
        (-fn(syn)).sum().backward()
        opt.step()
        if bounds is not None:
            with torch.no_grad():
                syn.copy_(torch.max(torch.min(syn, hi), lo))
    return syn.detach().numpy().squeeze()


# persistence helpers
def save_maps(path, maps, idx):
    np.savez_compressed(path, idx=idx, **{f"{k[0]}__{k[1]}": v for k, v in maps.items()})


def load_maps(path):
    z = np.load(path)
    maps = {tuple(k.split("__")): z[k] for k in z.files if k != "idx"}
    return maps, z["idx"]


def table_rows(res, set_name, kind="within"):
    """flatten summary statistics for tables: list of dict rows"""
    rows = []
    for key, mets in res[kind].items():
        if key[0] != set_name:
            continue
        row = {"model_or_method": key[1], "a": key[2], "b": key[3]}
        for m, arr in mets.items():
            s = summarize(arr)
            row[m] = s["mean"]
            row[m + "_median"] = s["median"]
            row[m + "_std"] = s["std"]
        rows.append(row)
    return rows
