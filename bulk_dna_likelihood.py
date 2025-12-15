from typing import List
import math


class SNV:
    # Single SNV in bulk data.
    """
        Each SNV has certain properties, including:
            variant_reads: number of reads supporing the variant allele
            total_reads: total read depth
            major_cn: major copy number (baseline/most common copy number for a genomic region)
            minor_cn: minor copy number (indicate a gain or loss from major state)
            clone_index: which clone this SNV is assigned to in z_n
            id: id of this SNV from the sample data; defaults to none
    """
    def __init__(self, variant_reads, total_reads, major_cn, minor_cn, clone_index):
        self.variant_reads = variant_reads 
        self.total_reads = total_reads 
        self.major_cn = major_cn 
        self.minor_cn = minor_cn 
        self.clone_index = clone_index 
        self.id = ""



class Genotype:
    """
        Genotype with total copies c and variant copies v. So for instance if c = 2, v = 1 then we have one reference and one variant
    """
    def __init__(self, total_copies, variant_copies):
        self.total_copies = total_copies
        self.variant_copies = variant_copies

#phi[k] is cellular prev of clone k
ClonePrevalences = List[float]

"""
    Return all genotypes for given major/minor copy numbers so assume that total copy c is the sum of major and minor
    Then for this c, consider all possible variant copy counts v which puts a uniform prior over v given c 
"""
def enumerate_genotypes(major_cn, minor_cn):
    total_copies = major_cn + minor_cn

    #no DNA copies
    if total_copies <= 0:
        return [Genotype(total_copies=0, variant_copies=0)]
    
    return [
        Genotype(total_copies=total_copies, variant_copies=v)
        for v in range(total_copies + 1)
    ]


""" 
    from the paper: 
    theta(g, phi, epsilon) = 
        epsilon                                     if nu(gn) = 0
        phi(1-epsilon) + (1-phi)epsilon             if nu(gn) = c(gn)
        phi(nu(gn)/c(gn)) + (1-phi)epsilon          otherwise
"""

def theta(genotype: Genotype, phi_clone: float, epsilon: float):
    """Per-read variant probability theta(g, phi, epsilon) for one genotype."""
    c = genotype.total_copies
    v = genotype.variant_copies

    if c <= 0: 
        return epsilon

    if v == 0: 
        return epsilon

    if v == c: #SNV is present in all copies and mixture of tumor clone (1 - epsilon) and epsilon weighted by phi_clone
        return phi_clone * (1.0 - epsilon) + (1.0 - phi_clone) * epsilon

    #intermediate case 0 < v < c so within the clone the expected variant fraction is v/c
    return phi_clone * (v / c) + (1.0 - phi_clone) * epsilon

"""
    log Binomial(n, p) at k computed stably so we just implement the og C(n, k) + k log p + (n - k) log(1 - p)
    using lgamma for the log factorial terms.
"""
def log_binomial_pmf(k: int, n: int, p: float):
    if n == 0: #if no trials 0
        return 0.0 if k == 0 else -math.inf

    if k < 0 or k > n: #out of range k gives zero prob. in Bin
        return -math.inf

    #make sure not log(0) bu clipping p into it
    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)

     # log C(n, k) = log(n!) - log(k!) - log((n - k)!)
    log_comb = ( math.lgamma(n + 1)  - math.lgamma(k + 1)  - math.lgamma((n - k) + 1))
    return log_comb + k * math.log(p) + (n - k) * math.log(1.0 - p)


# Stable logsumexp over a list of log vals
def logsumexp(values: List[float]):
    if not values:
        return -math.inf

    m = max(values)

    if m == -math.inf:
        return -math.inf

    s = sum(math.exp(v - m) for v in values)
    return m + math.log(s)

"""
    log P(b_n | d_n, M_n, m_n, phi_{z_n}, epsilon) for one SNV
"""
def snv_log_likelihood(snv: SNV, phi: ClonePrevalences, epsilon: float):
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
        raise IndexError( #if z_n out of range err to catch
            f"Clone index {z_n} out of range for phi of length {len(phi)}"
        )

    genotypes = enumerate_genotypes(M_n, m_n) 

    #compute log P(b_n | d_n, M_n, m_n, phi_{z_n}, epsilon) = log Bin(d_n, b_n; theta)
    log_terms: List[float] = []
    for g in genotypes:
        p = theta(g, phi_clone, epsilon)
        log_p = log_binomial_pmf(b_n, d_n, p)
        log_terms.append(log_p)

    if not log_terms:
        return -math.inf

    #assume a uniform prior over genotypes
    log_sum = logsumexp(log_terms)
    return log_sum - math.log(len(genotypes))


"""
    Total bulk log-likelihood log p(B | T, z, phi) = sum over n P(b_n | d_n, M_n, m_n, phi_{z_n}, epsilon)
"""
def bulk_log_likelihood(snvs: List[SNV], phi: ClonePrevalences, epsilon: float):
    total = 0.0
    #sum per SNV log likelihoods in the bulk dataset
    for snv in snvs: 
        total += snv_log_likelihood(snv, phi, epsilon)
    return total
