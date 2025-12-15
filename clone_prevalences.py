import numpy as np
from math import lgamma
from scrna import log_scrna_likelihood, ScRNALikelihoodParams
from bulk_dna_likelihood import bulk_log_likelihood


# Small floor to keep Dirichlet concentration parameters strictly positive
_ALPHA_EPS = 1e-8


""" Computing the log density of a dirichlet distriubtion, using the formula: 
log p(phi | alpha) = lgamma(sum(alpha)) - sum(lgamma(alpha_i)) + sum((alpha_i -1) * log(phi_1))
This is the prior over clone prevalences.  """

def dirichlet_log(phi, alpha):
    phi = np.asarray(phi)
    phi = np.clip(phi, 1e-12, 1.0)  # prevents log(0)
    alpha = np.asarray(alpha)
    if np.any(alpha <= 0):
        raise ValueError("dirichlet_log: all alpha concentrations must be > 0")
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
    def __init__(self, tree, alpha=1.0, scrna_params=None):
        self.tree = tree
        self.K = len(tree.nodes_except_root)
        # keep the original alpha parameter and construct alpha vectors on demand
        # validate alpha (must be > 0)
        if np.isscalar(alpha):
            if alpha <= 0:
                raise ValueError("PhiSample: alpha must be > 0")
        else:
            a_arr = np.asarray(alpha)
            if np.any(a_arr <= 0):
                raise ValueError("PhiSample: all entries of alpha must be > 0")

        self.alpha_param = alpha

        self.alpha = np.full(self.K, alpha)
        self.phi = self.prior_sample()
        self.scrna_params = scrna_params

    """Initial sample from dirichlet prior"""
    def prior_sample(self):
        # sample using the current tree size; ensure concentrations are strictly positive
        alpha_vec = np.full(self.K, self.alpha_param)
        alpha_vec = np.clip(alpha_vec, _ALPHA_EPS, None)
        return np.random.dirichlet(alpha_vec)

    """Computing the Dirichlet log prior for a given phi """
    def prior_log(self, phi):
        # ensure alpha vector matches phi length and is strictly positive
        phi = np.asarray(phi)
        alpha_vec = np.full(len(phi), self.alpha_param)
        alpha_vec = np.clip(alpha_vec, _ALPHA_EPS, None)
        return dirichlet_log(phi, alpha_vec)

    """Drawing fresh phi from Dirichlet prior"""
    def sample_prior(self):
        alpha_vec = np.full(self.K, self.alpha_param)
        alpha_vec = np.clip(alpha_vec, _ALPHA_EPS, None)
        return np.random.dirichlet(alpha_vec)

    """Propose a new clone prevalence vector by sampling from a Dirichlet distrbution centered around the current phi.
      We scale phi by a step facotr so that most proposals stay close to the current value,
     which helps the MCMC explore the space smoothly without making huge jumps."""

    def propose(self, phi, step=50):
        # form proposal concentration parameters centered on current phi
        phi = np.asarray(phi)
        alpha_prop = phi * step
        # ensure all entries > 0 for Dirichlet sampling
        alpha_prop = np.clip(alpha_prop, _ALPHA_EPS, None)
        return np.random.dirichlet(alpha_prop)

    """
    Run one update step of the Metropolis Hastings sampler for phi.
    Generate a new phi and compute its posterior probability using the bulk DNA likelihood, scRNA likelihood,
    and the Dirichlet prior over phi.
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
        log_scrna_old = 0.0
        log_scrna_new = 0.0
        if S is not None:
            log_scrna_old = log_scrna_likelihood(S, phi, clone_has_snv, self.scrna_params)
            log_scrna_new = log_scrna_likelihood(S, phi_propose, clone_has_snv, self.scrna_params)
        log_prior_old = self.prior_log(phi)
        log_prior_new = self.prior_log(phi_propose)
        
        log_post_old = log_bulk_old + log_scrna_old + log_prior_old
        log_post_new = log_bulk_new + log_scrna_new + log_prior_new

        #add in Hastings Correction
        alpha_forward = np.clip(phi * 50, _ALPHA_EPS, None)
        alpha_reverse = np.clip(phi_propose * 50, _ALPHA_EPS, None)

        log_forward = dirichlet_log(phi_propose, alpha_forward)
        log_reverse = dirichlet_log(phi, alpha_reverse)

        log_accept = (log_post_new - log_post_old + log_reverse - log_forward)
        
        accept = False
        if np.log(np.random.rand()) < log_accept:
            accept = True 
            return phi_propose, accept

        return phi, accept

    """ 
    Given an assignment vector z[n] = clone index of SNV n, return an array snv_phi[n]= phi[z[n]]
    """
    def maptophi(self, z):
        return np.array([self.phi[k] for k in z])
