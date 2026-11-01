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
    print(f"  {label}: {mean:.3f}   (2 decimals: {round(mean, 2):.2f})")

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
W_uncorrected = 12 * S / (N ** 2 * (k ** 3 - k))
# Ratings on a 5-point scale produce many ties within a participant, so W is computed with the tie correction
# T_i = sum over groups of tied ratings (t^3 - t) of participant i   (this is the value reported in the paper)
T_ties = sum(sum(t ** 3 - t for t in np.unique(r, return_counts=True)[1]) for r in ranks)
W     = 12 * S / (N ** 2 * (k ** 3 - k) - N * T_ties)
chi2  = N * (k - 1) * W
p_w   = 1 - stats.chi2.cdf(chi2, k - 1)
print(f"  Kendall's W (tie-corrected, reported) = {W:.4f}")
print(f"  chi2({k-1}) = {chi2:.3f}, p = {p_w:.8f}   (equals the Friedman statistic)")
print(f"  [without tie correction W would be {W_uncorrected:.4f}; not used]")

# ── 3b. Effect size (Stage 1 -> Stage 8) ─────────────────────────────────────
delta = data[:, 6] - data[:, 0]
d_z   = delta.mean() / delta.std(ddof=1)
d_av  = delta.mean() / np.sqrt((data[:, 0].var(ddof=1) + data[:, 6].var(ddof=1)) / 2)
print("\n--- Cohen's d, Stage 1 -> Stage 8 ---")
print(f"  d_z  (mean difference / SD of the differences)      = {d_z:.3f}   (reported in the paper)")
print(f"  d_av (mean difference / average of the two SDs)     = {d_av:.3f}")

# ── 4. Cronbach's Alpha (internal consistency / reliability) ──────────────────
print("\n--- Cronbach's Alpha (reliability across the 7 trust-stage items) ---")
item_variances = data.var(axis=0, ddof=1)
total_variance = data.sum(axis=1).var(ddof=1)
alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
print(f"  Cronbach's alpha = {alpha:.4f}  (k = {k} items, N = {N})")
if alpha >= 0.9:
    reliability = "excellent"
elif alpha >= 0.8:
    reliability = "good"
elif alpha >= 0.7:
    reliability = "acceptable"
elif alpha >= 0.6:
    reliability = "questionable"
else:
    reliability = "poor"
print(f"  Reliability: {reliability} (conventional Cronbach's-alpha thresholds)")

print("\n" + "=" * 60)
print("Significance: * p<.05  ** p<.01  *** p<.001  ns = not significant")
print("=" * 60)
