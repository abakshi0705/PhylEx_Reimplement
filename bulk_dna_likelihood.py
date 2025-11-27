"""Bulk DNA likelihood for PhylEx (SNV + clone prevalences)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math


@dataclass
class SNV:
    """Single SNV in bulk data."""
    variant_reads: int
    total_reads: int
    major_cn: int
    minor_cn: int
    clone_index: int


@dataclass
class Genotype:
    """Genotype with total copies c and variant copies v."""
    total_copies: int
    variant_copies: int


ClonePrevalences = List[float]


def enumerate_genotypes(major_cn: int, minor_cn: int) -> List[Genotype]:
    """Return all genotypes for given major/minor copy numbers."""
    total_copies = major_cn + minor_cn
    if total_copies <= 0:
        return [Genotype(total_copies=0, variant_copies=0)]
    return [
        Genotype(total_copies=total_copies, variant_copies=v)
        for v in range(total_copies + 1)
    ]


def theta(genotype: Genotype, phi_clone: float, epsilon: float) -> float:
    """Per-read variant probability θ(g, φ, ε) for one genotype."""
    c = genotype.total_copies
    v = genotype.variant_copies

    if c <= 0:
        return epsilon

    if v == 0:
        return epsilon

    if v == c:
        return phi_clone * (1.0 - epsilon) + (1.0 - phi_clone) * epsilon

    return phi_clone * (v / c) + (1.0 - phi_clone) * epsilon


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    """log Binomial(n, p) at k, computed stably."""
    if n == 0:
        return 0.0 if k == 0 else -math.inf

    if k < 0 or k > n:
        return -math.inf

    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)

    log_comb = (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma((n - k) + 1)
    )
    return log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)


def _logsumexp(values: List[float]) -> float:
    """Stable logsumexp."""
    if not values:
        return -math.inf
    m = max(values)
    if m == -math.inf:
        return -math.inf
    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s)


def snv_log_likelihood(snv: SNV,
                       phi: ClonePrevalences,
                       epsilon: float) -> float:
    """log P(b_n | d_n, M_n, m_n, φ_{z_n}, ε) for one SNV."""
    b_n = snv.variant_reads
    d_n = snv.total_reads
    M_n = snv.major_cn
    m_n = snv.minor_cn
    z_n = snv.clone_index

    if d_n == 0:
        return 0.0 if b_n == 0 else -math.inf

    try:
        phi_clone = phi[z_n]
    except IndexError:
        raise IndexError(
            f"Clone index {z_n} out of range for phi of length {len(phi)}"
        )

    genotypes = enumerate_genotypes(M_n, m_n)

    log_terms: List[float] = []
    for g in genotypes:
        p = theta(g, phi_clone, epsilon)
        log_p = _log_binomial_pmf(b_n, d_n, p)
        log_terms.append(log_p)

    if not log_terms:
        return -math.inf

    log_sum = _logsumexp(log_terms)
    return log_sum - math.log(len(genotypes))


def bulk_log_likelihood(snvs: List[SNV],
                        phi: ClonePrevalences,
                        epsilon: float) -> float:
    """Total bulk log-likelihood log p(B | T, z, φ)."""
    total = 0.0
    for snv in snvs:
        total += snv_log_likelihood(snv, phi, epsilon)
    return total
