
import numpy as np
import math
from clone_prevalences import PhiSample, dirichlet_log
from TSSB import Node, get_node_list, assign_snvs
from bulk_dna_likelihood import SNV, bulk_log_likelihood
from scrna import log_scrna_likelihood, ScRNALikelihoodParams, log_likelihood_cell_given_clone


class Tree:
    def __init__(self, node_list):

        #the full list of all nodes in the tree, including the non-cancerous root node
        self.node_list = node_list

        #all the nodes in the tree excluding the non-cancerous root node
        self.nodes_except_root = [node for node in node_list if not node.is_root]

        self.k = len(self.nodes_except_root)
    
    def node_index_full_tree(self, clone_index):
        return self.node_list.index(self.nodes[clone_index])
    
#adjust z so that it represents clone indices in the tree (bascially the indices in tree.nodes_except_root)
def adjust_z(z, tree):
    z_updated = []
    for node_index in z:
        node = tree.node_list[node_index]
        clone_index = tree.nodes_except_root.index(node)
        z_updated.append(clone_index)
    return z_updated

def clone_has_snv_matrix(z, tree, num_snvs):
    k = tree.k
    clone_has_snv = [[False for i in range(num_snvs)] for i in range(k)] 

    for i in range(len(z)):
        node = tree.node_list(z[i])

        for k in range(len(tree.nodes)):
            clone_node = tree.nodes[k]
            if check_if_descendant(clone_node, node):
                clone_has_snv[k][i] = True
    
    return clone_has_snv

def check_if_descendant(current_node, ancestor_node):
    while current_node is not None:
        if current_node == ancestor_node:
            return True
        current_node = current_node.parent
    
    return False

def update_bulk_snvs_indices(bulk_snvs, z_updated):
    updated_snvs = []
    for i, snv in enumerate(bulk_snvs):
        updated_snv = SNV(
            variant_reads=snv.variant_reads,
            total_reads=snv.total_reads,
            major_cn=snv.major_cn,
            minor_cn=snv.minor_cn,
            clone_index=z_updated[i]
        )
        updated_snvs.append(updated_snv)
    return updated_snvs


def initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, alpha=1.0):
    node_list = get_node_list(lamb_0, lamb, gamma)
    tree = Tree(node_list)

    z = assign_snvs(node_list, num_snvs)

    phi_sampler = PhiSample(tree, alpha)

    phi = phi_sampler.sample_prior()

    return tree, z, phi_sampler, phi 

def compute_log_posterior(phi, bulk_snvs, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler):
    log_bulk = bulk_log_likelihood(bulk_snvs, phi.toList(), epsilon)

    log_scrna = log_scrna_likelihood(scrna_data, phi.toList(), clone_has_snv, scrna_config)

    log_prior = phi_sampler.prior_log(phi)

    log_posterior = log_bulk + log_scrna + log_prior

    return log_posterior


def mcmc(bulk_snvs, scrna_data, lamb_0, lamb, gamma, epsilon, num_iterations, burnin, scrna_config):
    num_snvs = len(bulk_snvs)

    tree, z, phi_sampler = initialize_prior_tree(lamb_0, lamb, gamma, num_snvs)

    z_clone = adjust_z(z, tree)

    bulk_snvs_updated = update_bulk_snvs_indices(bulk_snvs, z_clone)

    clone_has_snv = clone_has_snv_matrix(z, tree, num_snvs)

    maximum_log_post = -math.inf

    for i in range(num_iterations):
        phi = phi_sampler.update(phi, bulk_snvs_updated, epsilon, scrna_data, clone_has_snv)

        log_post = compute_log_posterior(phi, bulk_snvs_updated, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler)
    
    return log_post



