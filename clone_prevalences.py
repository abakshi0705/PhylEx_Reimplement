print("[DEBUG] clone_prevalences.py imported")
import numpy as np
from math import lgamma
from scrna import log_scrna_likelihood, ScRNALikelihoodParams
from bulk_dna_likelihood import bulk_log_likelihood


def dirichlet_log(phi, alpha):
    alpha = np.asarray(alpha)
    alpha0 = np.sum(alpha)
    norm = lgamma(alpha0) - sum(lgamma(a) for a in alpha)
    terms = np.sum((alpha - 1) * np.log(phi))
    return norm + terms


class PhiSample:
    def __init__(self, tree, alpha=1.0):
        self.tree = tree
        self.K = len(tree.nodes)
        self.alpha = np.full(self.K, alpha)
        self.phi = self.previous_sample()

    def previous_sample(self):
        return np.random.dirichlet(self.alpha)

    def previous_log(self, phi):
        return dirichlet_log(phi, self.alpha)

    def sample_prior(self):
        return np.random.dirichlet(self.alpha)

    #propose a new phi value for Metropolis-Hastings
    def propose(self, phi, step=50):
        alpha_prop = phi * step
        return np.random.dirichlet(alpha_prop)

    def update(self, phi, snvs, epsilon, S, clone_has_snv):
        phi_propose = self.propose(phi)
        log_bulk_old = bulk_log_likelihood(snvs, phi, epsilon)
        log_bulk_new = bulk_log_likelihood(snvs, phi_propose, epsilon)
        log_scrna_old = log_scrna_likelihood(S, phi, clone_has_snv, self.scrna_params)
        log_scrna_new = log_scrna_likelihood(
            S, phi_propose, clone_has_snv, self.scrna_params
        )
        log_prior_old = self.prior_log(phi)
        log_prior_new = self.prior_log(phi_propose)
        log_accept = (log_bulk_new + log_scrna_new + log_prior_new) - (
            log_bulk_old + log_scrna_old + log_prior_old
        )
        if np.log(np.random.rand()) < log_accept:
            return phi_propose

        return phi

    def maptophi(self, z):
        return np.array([self.phi[k] for k in z])
