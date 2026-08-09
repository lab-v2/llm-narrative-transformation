"""
Tiny, hand-checkable demo of the stats used in wilcoxon_tests.py.

Runs the *same* three steps on a 5-story toy dataset so you can see how the
Wilcoxon signed-rank test, the rank-biserial effect size, and the Holm
correction turn paired per-story scores into the numbers in
wilcoxon_paper_export.csv. Nothing here touches real data -- it's a teaching aid.

    python plot_scripts/wilcoxon_toy_example.py
"""

import numpy as np
from scipy.stats import wilcoxon, rankdata

# ---------------------------------------------------------------------------
# 1. INPUT: 5 stories, each transformed by BOTH methods, scored on KL
#    divergence. KL is a "lower = better fidelity" metric.
# ---------------------------------------------------------------------------
stories   = ["S1", "S2", "S3", "S4", "S5"]
abduction = np.array([0.8, 0.9, 1.0, 0.7, 1.1])   # our method
baseline  = np.array([1.5, 1.2, 0.9, 2.0, 1.4])   # e.g. Zero-Shot

diffs = np.round(abduction - baseline, 6)           # abd - base (round: kill float noise so equal-size diffs tie exactly)
print("=== 1. Paired data (KL: lower is better) ===")
print(f"{'story':6}{'abd':>6}{'base':>7}{'diff':>8}   winner")
for s, a, b, d in zip(stories, abduction, baseline, diffs):
    winner = "abduction" if d < 0 else ("baseline" if d > 0 else "tie")
    print(f"{s:6}{a:6.1f}{b:7.1f}{d:8.2f}   {winner}")

# ---------------------------------------------------------------------------
# 2. WILCOXON SIGNED-RANK, by hand: drop zeros, rank |diff|, sum by sign
# ---------------------------------------------------------------------------
nz = diffs[diffs != 0]                 # step: drop ties (none here)
ranks = rankdata(np.abs(nz))           # step: rank by magnitude (ties share avg)
w_pos = ranks[nz > 0].sum()            # rank-sum where baseline won  (diff > 0)
w_neg = ranks[nz < 0].sum()            # rank-sum where abduction won (diff < 0)
print("\n=== 2. Wilcoxon signed-rank (by hand) ===")
print(f"|diff| ranks : {dict(zip(stories, np.round(rankdata(np.abs(diffs)), 2)))}")
print(f"W+ (baseline-won rank sum)  = {w_pos}")
print(f"W- (abduction-won rank sum) = {w_neg}")

# ...and the same thing via scipy (what wilcoxon_tests.py calls)
W, p = wilcoxon(abduction, baseline, alternative="two-sided", zero_method="wilcox")
print(f"scipy: W = {W}  (that's min(W+,W-)),  two-sided p = {p:.4f}")

# ---------------------------------------------------------------------------
# 3. EFFECT SIZE: matched-pairs rank-biserial, then ORIENT so + = abduction
# ---------------------------------------------------------------------------
n = len(nz)
total = n * (n + 1) / 2
r_raw = (w_pos - w_neg) / total        # raw sign follows the diffs
kl_sign = -1.0                         # KL is "lower better" -> flip the sign
r_oriented = r_raw * kl_sign
print("\n=== 3. Rank-biserial effect size ===")
print(f"total = n(n+1)/2 = {n}*{n+1}/2 = {total:.0f}")
print(f"r_raw      = (W+ - W-)/total = ({w_pos} - {w_neg})/{total:.0f} = {r_raw:+.3f}")
print(f"effect_r   = r_raw * sign({'-1 for KL'}) = {r_oriented:+.3f}  "
      f"(+ = abduction better)")

# ---------------------------------------------------------------------------
# 4. HOLM correction across a FAMILY of 4 baseline comparisons
#    (Abduction vs Zero-Shot / Few-Shot / CoT / Self-Refine). Toy raw p's:
# ---------------------------------------------------------------------------
def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, running, m = [None] * len(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        out[i] = running
    return out

labels   = ["Zero-Shot", "Few-Shot", "CoT", "Self-Refine"]
raw_p    = [0.0003, 0.0010, 0.0200, 0.3000]
adj_p    = holm(raw_p)
print("\n=== 4. Holm step-down across the 4 baselines in this family ===")
print(f"{'baseline':12}{'raw p':>9}{'x factor':>10}{'p_holm':>10}  significant?")
for lbl, rp, ap, factor in zip(labels, raw_p, adj_p, [4, 3, 2, 1]):
    star = "yes" if ap < 0.05 else "no"
    print(f"{lbl:12}{rp:9.4f}{factor:>9}x{ap:10.4f}  {star}")
print("(Bonferroni would multiply ALL by 4: 0.02 -> 0.08 = NOT sig. Holm keeps it at 0.04 = sig.)")

print("\nTakeaway: abduction won 4/5 stories, its wins were the big ones, its one")
print("loss the smallest -> effect_r = +0.867 (near-total dominance); with a real")
print("28-story sample that lopsidedness yields a tiny p that survives Holm.")
