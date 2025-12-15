"""
scRNA-seq Likelihood Module

Methods in this file will compute the scRNA-seq likelihood term log p(S | T, z, phi)

It models:
- allelic imbalance using a Beta-Binomial mixture
- dropout (zero reads)
- background noise for SNVs not present in a clone
- cell-to-clone marginalization using clone prevalences phi
"""

import math
from typing import List, Tuple


#computes the sum of log probabilites in a list (function is same as one in bulk_dna_likelihood)
def logsumexp(values: List[float]) -> float:
    if not values:
        return -math.inf

    m = max(values)

    if m == -math.inf:
        return -math.inf

    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s)


def log_beta_binomial(b: int, d: int, alpha: float, beta: float) -> float:
    """
    Compute the log of the Beta-Binomial PMF:
    C(d, b) * Beta(b + alpha, d - b + beta) / Beta(alpha, beta)
    """
    if b < 0 or b > d:
        return float('-inf')

    # log of the binomial coefficient C(d, b) - equivalent to d choose b in log space
    log_comb = (
        math.lgamma(d + 1)
        - math.lgamma(b + 1)
        - math.lgamma(d - b + 1)
    )

    # log Beta(b+alpha, d-b+beta) - log Beta(alpha, beta)
    log_beta_num = (
        math.lgamma(b + alpha)
        + math.lgamma(d - b + beta)
        - math.lgamma(d + alpha + beta)
    )
    log_beta_den = (
        math.lgamma(alpha)
        + math.lgamma(beta)
        - math.lgamma(alpha + beta)
    )

    return log_comb + (log_beta_num - log_beta_den)


# Beta–Binomial Mixture Parameters
class ScRNALikelihoodParams:
    """
    Holds the hyperparameters for the Beta-Binomial mixture model.
    these are the hyperparameters that were provided in the supplementary file

    We use these to initialize default parameters 
    These define:
    - mono-allelic expression
    - bi-allelic expression
    - background/error distribution
    """

    def __init__(self, w_mono: float = 0.5, mut_mono_alpha: float = 1.0, mut_mono_beta: float = 5.0, mut_bi_alpha: float = 5.0,
        mut_bi_beta: float = 5.0, bg_alpha: float = 0.5, bg_beta: float = 20.0):
        
        self.w_mono = w_mono
        self.mut_mono_alpha = mut_mono_alpha
        self.mut_mono_beta = mut_mono_beta
        self.mut_bi_alpha = mut_bi_alpha
        self.mut_bi_beta = mut_bi_beta
        self.bg_alpha = bg_alpha
        self.bg_beta = bg_beta


# Likelihood for mutated and non-mutated SNVs
def log_mutated_mixture(b: int, d: int, params: ScRNALikelihoodParams) -> float:
    """
    Compute the log likelihood for a mutated SNV in a clone.
    Uses a mixture of two Beta-Binomial distributions:
    - mono-allelic expression
    - bi-allelic expression
    """
    if d == 0:
        return 0.0

    w = params.w_mono

    log_mono = log_beta_binomial(b, d, params.mut_mono_alpha, params.mut_mono_beta)
    log_bi   = log_beta_binomial(b, d, params.mut_bi_alpha, params.mut_bi_beta)

    return logsumexp([
        math.log(w) + log_mono,
        math.log(1 - w) + log_bi
    ])


def log_background(b: int, d: int, params: ScRNALikelihoodParams) -> float:
    """
    Compute the log likelihood for an SNV that is *not* mutated in this clone.
    This acts as a noise/error model.
    """
    if d == 0:
        return 0.0
    return log_beta_binomial(b, d, params.bg_alpha, params.bg_beta)


# Likelihood of a single cell assuming it belongs to 1 clone
def log_likelihood_cell_given_clone(
    cell_index: int,
    S: List[List[Tuple[int, int]]],
    clone_k: int,
    clone_has_snv: List[List[bool]],
    params: ScRNALikelihoodParams,
) -> float:
    """
    Compute log P(S_cell | clone = k).

    We loop over all SNVs and:
    - use the mixture model if mutation is present in clone k
    - use background noise if mutation is absent
    """
    total = 0.0
    cell_snvs = S[cell_index]

    for n, (b, d) in enumerate(cell_snvs):
        if d == 0:   
            continue

        if clone_has_snv[clone_k][n]:
            total += log_mutated_mixture(b, d, params)
        else:
            total += log_background(b, d, params)

    return total


# Full scRNA-seq likelihood with clone marginalization
def log_scrna_likelihood(
    S: List[List[Tuple[int, int]]],   
    phi: List[float],                
    clone_has_snv: List[List[bool]], 
    params: ScRNALikelihoodParams,
) -> float:
    """
    Compute log p(S | T, z, phi) by marginalizing over which clone
    each cell may have come from.

    For each cell c:
        P(S_c | T, z, phi) = sum_k phi[k] * P(S_c | clone k)

    We return the sum of log P(S_c | ...) over all cells.
    """
    num_cells = len(S)
    num_clones = len(phi)

    total_loglik = 0.0

    for c in range(num_cells):
        # collect log(phi[k] * likelihood of cell c in clone k)
        log_terms = []

        for k in range(num_clones):
            if phi[k] <= 0:
                continue

            log_prior = math.log(phi[k])
            log_lik   = log_likelihood_cell_given_clone(c, S, k, clone_has_snv, params)

            log_terms.append(log_prior + log_lik)

        # marginalize over all clone options via log-sum-exp
        cell_loglik = logsumexp(log_terms)
        total_loglik += cell_loglik

    return total_loglik
