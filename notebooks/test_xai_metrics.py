"""
Tests for the agreement metrics and the attribution methods, on small examples whose answer is known.
Run:  python test_xai_metrics.py
"""
import numpy as np
import torch
from scipy.stats import spearmanr

import xai_analysis as XA

ok = True


def check(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


rng = np.random.default_rng(0)
a = rng.normal(size=17)

# 1. identical profiles agree completely
r = XA.agreement(a, a.copy())
check("identical profiles: every metric is 1", all(abs(r[k] - 1) < 1e-12 for k in ("rank", "feature", "sign", "signedrank", "signedrank5", "spearman")))

# 2. a profile and its negative: same ranking by |value|, opposite signs
r = XA.agreement(a, -a)
check("negated profile: rank = feature = 1, sign = 0, signed-rank = 0, spearman = -1",
      r["rank"] == 1 and r["feature"] == 1 and r["sign"] == 0 and r["signedrank5"] == 0 and abs(r["spearman"] + 1) < 1e-12)

# 3. top-5 overlap of exactly 3 sensors gives feature agreement 0.6
x = np.zeros(10); y = np.zeros(10)
x[[0, 1, 2, 3, 4]] = [5, 4, 3, 2, 1]
y[[0, 1, 2, 7, 8]] = [5, 4, 3, 2, 1]
r = XA.agreement(x, y, k=5)
check("overlap of 3 of 5: feature@5 = 0.6", abs(r["feature"] - 0.6) < 1e-12, f"(got {r['feature']})")
check("same sensor at the same rank for the first 3 positions: rank@5 = 0.6", abs(r["rank"] - 0.6) < 1e-12, f"(got {r['rank']})")

# 4. same sensor and rank but different sign counts for rank, not for signed-rank
x = np.array([5., 4, 3, 2, 1, 0.1, 0.1, 0.1]); y = x.copy(); y[0] = -5
r = XA.agreement(x, y, k=5)
check("one sign flipped in the top 5: rank@5 = 1, signed-rank@5 = 0.8", r["rank"] == 1 and abs(r["signedrank5"] - 0.8) < 1e-12, f"(got {r['rank']}, {r['signedrank5']})")

# 5. spearman equals scipy
b = rng.normal(size=17)
check("spearman equals scipy.stats.spearmanr", abs(XA.agreement(a, b)["spearman"] - spearmanr(a, b)[0]) < 1e-12)

# 6. IG (zero baseline) and Occlusion on a linear model both give w * x
w = torch.tensor(rng.normal(size=(5, 3)), dtype=torch.float32)


class Linear(torch.nn.Module):
    def forward(self, x):
        return (x * w).sum(dim=(1, 2))


lin = Linear()
fns = {"cnn": lin}
X = torch.tensor(rng.normal(size=(4, 5, 3)), dtype=torch.float32)
exact = (X * w).numpy()
ig = XA.attribute("cnn", "ig", {}, fns, X)
oc = XA.attribute("cnn", "occlusion", {}, fns, X)
check("linear model: integrated gradients = w * x", np.allclose(ig, exact, atol=1e-5), f"(max error {np.abs(ig - exact).max():.1e})")
check("linear model: occlusion = w * x", np.allclose(oc, exact, atol=1e-5), f"(max error {np.abs(oc - exact).max():.1e})")
check("linear model: attributions sum to f(x) - f(0) (completeness)", np.allclose(ig.sum((1, 2)), lin(X).detach().numpy(), atol=1e-4))

# 7. the constant channels of FD001 are exactly zero after scaling and get zero attribution
try:
    from rul_common import load_fd001
    d = load_fd001()
    Xv = d["X_val"]
    check("constant channels are 0 in the scaled input (up to rounding, < 1e-12)", float(Xv[:, :, XA.CONSTANT_CHANNELS].abs().max()) < 1e-12)
    maps, _ = XA.load_maps("../results/fd001/maps.npz")
    zero = all(float(np.abs(m[:, :, XA.CONSTANT_CHANNELS]).max()) < 1e-4 for m in maps.values())
    check("constant channels get attribution 0 (< 1e-4, the largest attribution is 10 to 200) for all 9 maps", zero)
except FileNotFoundError:
    print("[SKIP] cached maps not found, run run_xai_analysis.py first")

print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
raise SystemExit(0 if ok else 1)
