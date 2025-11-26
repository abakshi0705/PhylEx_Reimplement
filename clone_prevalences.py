print("[DEBUG] clone_prevalences.py imported")
import numpy as np
from math import lgamma


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

    def propose(self, phi, step=50):
        alpha_prop = phi * step
        return np.random.dirichlet(alpha_prop)

    def update(self, phi, T, z, bulk_likelihood, scrna_likelihood):
        phi_propose = self.propose(phi)
        loglike_old = bulk_likelihood(T, z, phi) + scrna_likelihood(T, z, phi)
        loglike_new = bulk_likelihood(T, z, phi_propose) + scrna_likelihood(
            T, z, phi_propose
        )
        log_previous = self.previous_log(phi)
        log_newprevious = self.previous_log(phi_propose)
        log_accept = (loglike_new + log_newprevious) - (loglike_old + log_previous)
        if np.log(np.random.rand()) < log_accept:
            return phi_propose
        return phi

    def maptophi(self, z):
        return np.array([self.phi[k] for k in z])
