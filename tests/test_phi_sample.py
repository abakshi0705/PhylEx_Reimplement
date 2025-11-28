import numpy as np
from clone_prevalences import PhiSample, dirichlet_log

class DummyTree:
    def __init__(self, K):
        self.nodes = list(range(K))

    
def dummy_bulk(T, z, phi):
    # simple dummy likelihood: log(φ_k for each SNV)
    return np.sum(np.log(phi[z]))


def dummy_scrna(T, z, phi):
    return np.sum(np.log(phi[z] + 1e-6))


def test_prior_sampling():
    tree = DummyTree(K=3)
    phi_s = PhiSample(tree, alpha=1.0)
    phi = phi_s.sample_prior()

    assert len(phi) == 3
    assert np.all(phi > 0)
    assert np.isclose(np.sum(phi), 1.0)


def test_prior_log_matches_manual():
    phi = np.array([0.3, 0.5, 0.2])
    alpha = np.array([1.0, 1.0, 1.0])

    computed = dirichlet_log(phi, alpha)
    assert np.isfinite(computed)


def test_proposal_stays_in_simplex():
    tree = DummyTree(K=4)
    phi_s = PhiSample(tree)
    phi = np.array([0.1, 0.3, 0.2, 0.4])
    proposed = phi_s.propose(phi)

    assert len(proposed) == 4
    assert np.all(proposed > 0)
    assert np.isclose(np.sum(proposed), 1.0)


def test_mh_update_returns_valid_phi():
    tree = DummyTree(K=3)
    phi_s = PhiSample(tree)

    phi = phi_s.phi
    z = np.array([0, 1, 2, 1, 0])

    updated = phi_s.update(phi, tree, z, dummy_bulk, dummy_scrna)

    assert len(updated) == 3
    assert np.all(updated > 0)
    assert np.isclose(np.sum(updated), 1.0)


def test_maptophi_correct():
    tree = DummyTree(K=3)
    phi_s = PhiSample(tree)
    phi_s.phi = np.array([0.1, 0.3, 0.6])

    z = np.array([2, 0, 1, 2])
    mapped = phi_s.maptophi(z)

    expected = np.array([0.6, 0.1, 0.3, 0.6])
    assert np.allclose(mapped, expected)
