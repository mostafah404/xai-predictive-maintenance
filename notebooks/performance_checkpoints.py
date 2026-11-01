"""
Performance of the saved FD001 checkpoints (data/model.pth, cnn_model.pth, hybrid_model.pth) = the models explained in Section 4.
Validation: the 20 held-out engines (3070 windows).  Test: last 50 cycles of each of the 100 test engines (RUL file, capped at 125).
HIGH risk = true RUL < 40; risk classes: HIGH < 40 <= MEDIUM < 80 <= LOW.   S = mean of the CMAPSS score.
Writes ../results/fd001/performance_checkpoints.json and prints the LaTeX rows of Tables 4 and 5.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

import xai_analysis as XA
from rul_common import load_fd001, load_models

HERE = os.path.dirname(os.path.abspath(__file__))
data = load_fd001()
lstm, cnn, hyb = load_models()
fns = XA.forward_fns(lstm, cnn, hyb)
scaler = data["scaler"]


def predict(f, X):
    with torch.no_grad():
        return torch.cat([f(X[i:i + 512]) for i in range(0, len(X), 512)]).squeeze().numpy()


def risk(r):
    return np.where(r < 40, 0, np.where(r < 80, 1, 2))     # 0 = HIGH, 1 = MEDIUM, 2 = LOW


def score(e):
    return np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1)


def metrics(p, t, with_s=False):
    e, hi = p - t, t < 40
    out = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
           "rmse_high": float(np.sqrt((e[hi] ** 2).mean())), "mae_high": float(np.abs(e[hi]).mean()),
           "acc": float(100 * (risk(p) == risk(t)).mean()), "n": int(len(t)), "n_high": int(hi.sum())}
    if with_s:
        out["s_mean"] = float(score(e).mean())
    return out


cols = ["machine_id", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
te = pd.read_csv(os.path.join(HERE, "..", "data", "test_FD001.txt"), sep=r"\s+", header=None, names=cols)
rul = np.minimum(np.loadtxt(os.path.join(HERE, "..", "data", "RUL_FD001.txt")), 125)
Xt = []
for _, g in te.groupby("machine_id"):
    v = g[cols[2:]].values[-50:]
    if len(v) < 50:
        v = np.vstack([np.repeat(v[:1], 50 - len(v), 0), v])
    Xt.append(v)
Xt = torch.tensor(scaler.transform(np.array(Xt).reshape(-1, 24)).reshape(-1, 50, 24), dtype=torch.float32)

res = {}
for mk in ("lstm", "cnn", "hybrid"):
    res[mk] = {"val": metrics(predict(fns[mk], data["X_val"]), data["y_val"]),
               "test": metrics(predict(fns[mk], Xt), rul, with_s=True)}
json.dump(res, open(os.path.join(HERE, "..", "results", "fd001", "performance_checkpoints.json"), "w"), indent=1)

for mk, r in res.items():
    v, t = r["val"], r["test"]
    print(f"{mk:7s} VAL  rmse {v['rmse']:.2f} mae {v['mae']:.2f} rmseH {v['rmse_high']:.2f} maeH {v['mae_high']:.2f} acc {v['acc']:.1f}  (n={v['n']}, high={v['n_high']})")
    print(f"{'':7s} TEST rmse {t['rmse']:.2f} mae {t['mae']:.2f} rmseH {t['rmse_high']:.2f} maeH {t['mae_high']:.2f} acc {t['acc']:.1f} S {t['s_mean']:.2f}  (n={t['n']}, high={t['n_high']})")
