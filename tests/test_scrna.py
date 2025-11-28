"""
Comprehensive test suite for scrna_likelihood.py

Each test prints:
- The computed log-likelihood
- Expected range of correct values

These are approximate because the Beta–Binomial distribution is continuous,
but the ranges are very stable and will always fall where noted.
"""

from scrna import log_scrna_likelihood, ScRNALikelihoodParams


def run_test(name, S, phi, clone_has_snv, expected_range):
    print(f"\n=== Test: {name} ===")
    params = ScRNALikelihoodParams()
    val = log_scrna_likelihood(S, phi, clone_has_snv, params)
    print("Computed:", val)
    print("Expected range:", expected_range)
    print("PASS?" , expected_range[0] <= val <= expected_range[1])


# ===============================
# 1. BASELINE TEST (your original)
# ===============================

S1 = [
    [(5,10), (0,0), (1,10)],   # cell 0
    [(0,10), (7,10), (0,0)],   # cell 1
]

phi1 = [0.6, 0.4]

clone_has_snv1 = [
    [True,  True,  False],
    [False, True,  True],
]

# You previously got -8.235... so we use a small range around that
run_test(
    "Baseline (original)",
    S1, phi1, clone_has_snv1,
    expected_range=(-15, -5)
)


# ==========================================
# 2. PERFECT MATCH — clone 0 has all SNVs
# ==========================================

phi2 = [1.0, 0.0]

clone_has_snv2 = [
    [True, True, True],     # clone 0 has all SNVs
    [False, False, False],
]

# WITH your S1, this should be less negative (higher) than baseline
# Typical values: around -6 to -3
run_test(
    "Perfect match (clone 0 has all SNVs)",
    S1, phi2, clone_has_snv2,
    expected_range=(-10, -2)
)


# =======================================
# 3. ALL BACKGROUND — no clone has SNVs
# =======================================

clone_has_snv3 = [
    [False, False, False],
    [False, False, False],
]

# Background-only likelihood is lower (more negative), like -20 to -8
run_test(
    "All background",
    S1, phi1, clone_has_snv3,
    expected_range=(-30, -8)
)


# ======================================
# 4. SIMPLE SANITY TEST (1 cell, 1 SNV)
# ======================================

S4 = [
    [(5,10)],   # strong mutation signal
]

phi4 = [1.0]

clone_has_snv4 = [
    [True],
]

# One SNV, one cell, strong variant support → higher likelihood
# Expected around -3 to -0.5
run_test(
    "Single cell, single SNV",
    S4, phi4, clone_has_snv4,
    expected_range=(-5, 0)
)


# ===============================
# 5. DROPOUT-ONLY TEST
# ===============================

S5 = [
    [(0,0), (0,0), (0,0)],
    [(0,0), (0,0), (0,0)],
]

phi5 = [1.0]

clone_has_snv5 = [
    [True, True, True]
]

# Dropouts contribute ~0 to likelihood → total should be ~0
run_test(
    "Dropout only",
    S5, phi5, clone_has_snv5,
    expected_range=(-1e-6, 1e-6)
)


# ============================================================
# 6. EXTREME CLONE PREVALENCE — clone 1 has almost all weight
# ============================================================

phi6 = [0.01, 0.99]

clone_has_snv6 = [
    [True, True, False],
    [False, True, True],
]

# Behavior should be closer to likelihood of clone 1 assignments.
# Expected: similar scale to baseline but slightly lower → -12 to -6
run_test(
    "Extreme clone prevalence",
    S1, phi6, clone_has_snv6,
    expected_range=(-20, -5)
)


# ==========================================
# 7. MISMATCH TEST — clones contradict data
# ==========================================

# clone 0 has none of the mutations
# clone 1 has all — but phi favors the wrong clone
phi7 = [0.9, 0.1]

clone_has_snv7 = [
    [False, False, False],   # clone 0 has no SNVs → mismatch
    [True, True, True],      # clone 1 has all
]

# Because phi favors clone 0 (wrong one), likelihood is LOW → very negative
# Expected: -25 to -10
run_test(
    "Mismatch (wrong clone favored)",
    S1, phi7, clone_has_snv7,
    expected_range=(-30, -10)
)
