"""
Correctness tests for lrp_recurrent.py.  Run:  python test_lrp_recurrent.py
Uses the saved FD001 weights (data/*.pth) and real validation windows.
"""
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import lrp_recurrent as L
from rul_common import load_fd001, load_models, lstm_rul

torch.manual_seed(0)
data = load_fd001()
lstm, cnn, hyb = load_models()
X = data["X_val"][torch.randperm(len(data["X_val"]))[:12]]
models = {"cnn": cnn, "lstm": lstm, "hybrid": hyb}
ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


def predict(m, x):
    with torch.no_grad():
        out = m(x)
    return (out[0] if isinstance(out, tuple) else out).double()


# 1. forward value of the modified graph equals the real model output
for k, m in models.items():
    _, y = L.lrp(m, X, eps=1e-6, return_output=True)
    err = (y - predict(m, X)).abs().max().item()
    report(f"forward equality ({k})", err < 1e-4, f"max|dy|={err:.2e}")

# 2. explicit propagation == modified-gradient formulation
for eps in (1e-9, 1e-6, 1e-3, 1e-2):
    for k, m in models.items():
        Ra = L.lrp(m, X, eps=eps)
        Rb = L.lrp_explicit(m, X, eps=eps)
        rel = ((Ra - Rb).abs().max() / (Ra.abs().max() + 1e-30)).item()
        report(f"explicit == modified-gradient ({k}, eps={eps:g})", rel < 1e-6, f"rel.err={rel:.2e}")


# 3. relevance conservation when all biases are zero (sum of input relevance == output)
def zero_bias(m):
    m = copy.deepcopy(m)
    for n, p in m.named_parameters():
        if "bias" in n:
            p.data.zero_()
    return m


for k, m in models.items():
    mz = zero_bias(m)
    R, y = L.lrp(mz, X, eps=1e-12, return_output=True)
    s = R.sum(dim=(1, 2))
    rel = ((s - y).abs() / (y.abs() + 1e-9)).max().item()
    report(f"conservation, zero biases ({k})", rel < 1e-6, f"max rel.dev={rel:.2e}")

# 4. CNN: agreement with Captum LRP (epsilon rule everywhere, as in the notebook)
from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule
for mod in cnn.modules():
    if isinstance(mod, (nn.Conv1d, nn.Linear, nn.ReLU, nn.Dropout)):
        mod.rule = EpsilonRule()
R_cap = LRP(cnn).attribute(X.clone()).detach().double()
R_own = L.lrp(cnn, X, eps=1e-9)
rel = ((R_cap - R_own).abs().max() / R_cap.abs().max()).item()
rho = np.mean([spearmanr(R_cap[i].numpy().ravel(), R_own[i].numpy().ravel())[0] for i in range(len(X))])
report("CNN: own LRP == Captum LRP", rel < 1e-3, f"rel.err={rel:.2e}, spearman={rho:.6f}")

# 5. stability w.r.t. epsilon (sensor-profile rank correlation to the eps=1e-6 reference)
for k, m in models.items():
    ref = L.lrp(m, X, eps=1e-6).mean(1)
    for eps in (1e-9, 1e-3, 1e-2):
        alt = L.lrp(m, X, eps=eps).mean(1)
        r = np.mean([spearmanr(ref[i].numpy(), alt[i].numpy())[0] for i in range(len(X))])
        print(f"      eps-stability ({k}): spearman(eps=1e-6, eps={eps:g}) = {r:.4f}")

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
