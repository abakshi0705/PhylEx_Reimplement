import numpy as np
from math import lgamma
from scrna import log_scrna_likelihood, ScRNALikelihoodParams
from bulk_dna_likelihood import bulk_log_likelihood

""" Computing the log density of a dirichlet distriubtion, using the formula: 
log p(phi | alpha) = lgamma(sum(alpha)) - sum(lgamma(alpha_i)) + sum((alpha_i -1) * log(phi_1))
This is the prior over clone prevalences.  """


def dirichlet_log(phi, alpha):
    alpha = np.asarray(alpha)
    alpha0 = np.sum(alpha)
    norm = lgamma(alpha0) - sum(lgamma(a) for a in alpha)
    terms = np.sum((alpha - 1) * np.log(phi))
    return norm + terms

""" 
This class handles sampling clone prevalences in the PhylEx model.
tree contains the clone structure and number of clones
alpha is the Dirichlet prior concentration parameter. Dirichlet(1) is the standard non-informative prior
so we set alpha=1.0 for each clone.

"""

class PhiSample:
    def __init__(self, tree, alpha=1.0):
        self.tree = tree
        self.K = len(tree.nodes)
        self.alpha = np.full(self.K, alpha)
        self.phi = self.previous_sample()

    """Initial sample from dirichlet prior"""
    def previous_sample(self):
        return np.random.dirichlet(self.alpha)

    """Computing the Dirichlet log prior for a given phi """
    def previous_log(self, phi):
        return dirichlet_log(phi, self.alpha)

    """Drawing fresh phi from Dirichlet prior"""
    def sample_prior(self):
        return np.random.dirichlet(self.alpha)

    """Propose a new clone prevalence vector by sampling from a Dirichlet distrbution centered around the current phi.
      We scale phi by a step facotr so that most proposals stay close to the current value,
     which helps the MCMC explore the space smoothly without making huge jumps."""
    def propose(self, phi, step=50):
        alpha_prop = phi * step
        return np.random.dirichlet(alpha_prop)

    """
    Run one update step of the Metropolis Hastings sampler for phi.
    Generate a new phi and compute its posterior probability using the bulk DNA likelihood, scRNA likelihood,
    and the DiRichlet prior over phi.
    The proposal is accepted with the standard MH acceptance probability. 
    The parameters are defined as follows: 
    phi: current prevalence
    snvs: list of snv objects providing bulk read information
    epsilon: bulk sequencing noise level
    S: per-cell scRNA seq read counts
    clone_has_snv: boolean matrix including mutation inheritance 
    """
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

    """ 
    Given an assignment vector z[n] = clone index of SNV n, return an array snv_phi[n]= phi[z[n]]
    """
    def maptophi(self, z):
        return np.array([self.phi[k] for k in z])
