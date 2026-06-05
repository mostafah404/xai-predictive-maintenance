import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata

# Load survey responses
df = pd.read_csv('../data/AI Predictive Maintenance.csv')
cols = df.columns.tolist()

# Trust rating columns (Stage 1, 3, 4, 5, 6, 7, 8)
stage_cols = [cols[1], cols[4], cols[5], cols[6], cols[7], cols[8], cols[9]]
data = df[stage_cols].values.astype(float)

stage_labels = ['Stage 1', 'Stage 3', 'Stage 4', 'Stage 5', 'Stage 6', 'Stage 7', 'Stage 8']

print("=" * 60)
print("TRUST EVALUATION — STATISTICAL ANALYSIS")
print(f"N = {data.shape[0]} participants, {data.shape[1]} stages")
print("=" * 60)

# Means per stage
print("\n--- Mean Trust Scores per Stage ---")
for label, mean in zip(stage_labels, data.mean(axis=0)):
    print(f"  {label}: {mean:.3f}")

# ── 1. Wilcoxon Signed-Rank Tests ─────────────────────────────────────────────
print("\n--- Wilcoxon Signed-Rank Tests (one-tailed, alternative='greater') ---")
pairs = [
    ("Stage 1 -> 3", 0, 1),
    ("Stage 3 -> 4", 1, 2),
    ("Stage 4 -> 5", 2, 3),
    ("Stage 5 -> 6", 3, 4),
    ("Stage 6 -> 7", 4, 5),
    ("Stage 7 -> 8", 5, 6),
    ("Stage 1 -> 8", 0, 6),
]
for label, i, j in pairs:
    diff = data[:, j] - data[:, i]
    diff_nz = diff[diff != 0]
    if len(diff_nz) == 0:
        print(f"  {label}: no differences, p = 1.0")
    else:
        stat, p = stats.wilcoxon(diff_nz, alternative='greater')
        sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "ns"))
        print(f"  {label}: W = {stat:.1f}, p = {p:.6f}  {sig}")

# ── 2. Friedman Test ──────────────────────────────────────────────────────────
print("\n--- Friedman Test (all 7 stages) ---")
stat, p = stats.friedmanchisquare(*[data[:, i] for i in range(7)])
print(f"  chi2(6) = {stat:.3f}, p = {p:.8f}")

# ── 3. Kendall's W ────────────────────────────────────────────────────────────
print("\n--- Kendall's Coefficient of Concordance (W) ---")
ranks = np.apply_along_axis(rankdata, 1, data)
R_j   = ranks.sum(axis=0)
N, k  = data.shape
R_bar = N * (k + 1) / 2
S     = np.sum((R_j - R_bar) ** 2)
W     = 12 * S / (N ** 2 * (k ** 3 - k))
chi2  = N * (k - 1) * W
p_w   = 1 - stats.chi2.cdf(chi2, k - 1)
print(f"  Kendall's W = {W:.4f}")
print(f"  chi2({k-1}) = {chi2:.3f}, p = {p_w:.8f}")

print("\n" + "=" * 60)
print("Significance: * p<.05  ** p<.01  *** p<.001  ns = not significant")
print("=" * 60)
