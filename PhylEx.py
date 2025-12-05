
import numpy as np
import math
from clone_prevalences import PhiSample, dirichlet_log
from TSSB import Node, get_node_list, assign_snvs
from bulk_dna_likelihood import SNV, bulk_log_likelihood, snv_log_likelihood
from scrna import log_scrna_likelihood, ScRNALikelihoodParams, log_likelihood_cell_given_clone
import time


class Tree:
    def __init__(self, node_list, snvs):
        #the full list of all nodes in the tree, including the non-cancerous root node
        self.node_list = node_list
        self.snvs = snvs

        #all the nodes in the tree excluding the non-cancerous root node
        self.nodes_except_root = [node for node in node_list if not node.is_root]
        self.k = len(self.nodes_except_root)
    
    #returnes the index of a node in the full tree, given a clone index
    #a clone index would be an index from self.nodes_except_root
    def node_index_full_tree(self, clone_index):
        return self.node_list.index(self.nodes_except_root[clone_index])
    
    def return_newick_form(self):
        node_to_index = {node: i for i, node in enumerate(self.node_list)}
               # Build children adjacency
        children = {i: [] for i in range(len(self.node_list))}
        root_index = None

        for index, node in enumerate(self.node_list):
            if node.is_root:
                root_index = index
            if node.parent is not None:
                parent_index = node_to_index[node.parent]
                children[parent_index].append(index)
        
        def build_subtree(index):
           node = self.node_list[index]
           label = f"N{index}"
           if not children[index]:
               # Leaf
               return label
           else:
               child_strs = [build_subtree(c) for c in children[index]]
               return "(" + ",".join(child_strs) + ")" + label

        return build_subtree(root_index) + ";"
    

#class for slice sampling a tree; we use this to explore the tree space
class TreeSampler:
    def __init__(self, tree, scrna_params=None):
        self.tree = tree
        self.scrna_params = scrna_params if scrna_params else ScRNALikelihoodParams()

    def slice_sample_tree(self, phi, bulk_snvs, scrna_data, epsilon, min_depth):
        #this list will hold all th new snv assignments that we make
        new_snvs = self.tree.snvs.copy()

        for n in range(len(self.tree.snvs)):
            new_snvs[n] = self.sample_snv_assignment(n, new_snvs, phi, bulk_snvs, scrna_data, epsilon)
        
        #create a new tree with the new_snvs
        new_tree = self.prune_empty_nodes(new_snvs)

        tree_depth = max(node.height for node in new_tree.node_list)
    
        if tree_depth < min_depth:
            # Tree too shallow, reject and keep old tree
            print(f"  Warning: Rejected tree with depth {tree_depth} < {min_depth}")
            return self.tree, self.tree.snvs
    
        return new_tree, new_tree.snvs


    def sample_snv_assignment(self, n, current_snvs, phi, bulk_snvs, scrna_data, epsilon):
        #find the node that this SNV is currently assigned to
        currently_assigned_node = current_snvs[n]

        #calculate the current log probability of this snv
        current_log_prob = self.log_prob_snv(n, currently_assigned_node, current_snvs, phi, bulk_snvs, scrna_data, epsilon)

        #we slice sample u ~ uniform(0, current_prob)
        #mathematically, this is some value x * current_prob where x is between 0 and 1
        #taking the log of this, we get log(x) + current_log_prob
        log_u = current_log_prob + np.log(np.random.uniform(0,1))

        #will hold all nodes where new_log_prob > log_u
        valid_nodes = []
        valid_log_probs = []

        for index, node in enumerate(self.tree.node_list):
            if not node.is_root:
                new_log_prob = self.log_prob_snv(n, index, current_snvs, phi, bulk_snvs, scrna_data, epsilon)
                if new_log_prob > log_u:
                    valid_nodes.append(index)
                    valid_log_probs.append(new_log_prob)
        
        #if there are no valid nodes, ie the currently assigned node is the only possible node for this SNV
        if not valid_nodes:
            return currently_assigned_node
        
        #normalization
        valid_log_probs = np.array(valid_log_probs)
        valid_log_probs = valid_log_probs - np.max(valid_log_probs)
        probs = np.exp(valid_log_probs)
        probs = probs / np.sum(probs)
       
        #sample
        new_assignment = np.random.choice(valid_nodes, p=probs)
    
        return new_assignment


    def log_prob_snv(self, snv_index, node_index, current_snvs, phi, bulk_snvs, scrna_data, epsilon):
        #create an adjusted snv assignment list where the snv of interest is assigned to the node of interest
        #the node of interest is the node which is currently being looped over in sample_snv_assignment
        snvs_temp = current_snvs.copy()
        snvs_temp[snv_index] = node_index

        #get the probability for the snv being assigned to this node
        node = self.tree.node_list[node_index]
        old_log = np.log(node.pi_u + 1e-10)

        #adjust the snv assignment list so that it represents clone indices (excludes root indices)
        snv_clone_temp = adjust_z(snvs_temp, self.tree)

        bulk_snvs_temp = update_bulk_snvs_indices(bulk_snvs, snv_clone_temp)

        log_bulk = snv_log_likelihood(bulk_snvs_temp[snv_index], phi.tolist(), epsilon)

        log_scrna = 0.0
        if scrna_data is not None:
            temp_clone_has_snv = make_clone_has_snv_matrix(snvs_temp, self.tree, len(self.tree.snvs))

            log_scrna = log_scrna_likelihood(scrna_data, phi.tolist(), temp_clone_has_snv, self.scrna_params)
        
        return old_log + log_bulk + log_scrna
       
    """
    removes any nodes with no SNVs assigned to it and whos descendants have no snvs assigned to them
    """
    def prune_empty_nodes(self, new_snvs):
        snv_counts = {i: 0 for i in range(len(self.tree.node_list))}
        
        for index in new_snvs:
            snv_counts[index] += 1
        
        nodes_to_keep = set()

        for node_index, count in snv_counts.items():
            if count > 0:
                current = self.tree.node_list[node_index]
                while current is not None:
                    current_index = self.tree.node_list.index(current)
                    nodes_to_keep.add(current_index)
                    current = current.parent

        new_node_list = [self.tree.node_list[i] for i in sorted(nodes_to_keep)]
    
        # Create mapping from old indices to new indices
        old_to_new = {old_index: new_index for new_index, old_index in enumerate(sorted(nodes_to_keep))}
    
        # Update z with new indices
        new_snv_list = [old_to_new[node_index] for node_index in new_snvs]
    
        return Tree(new_node_list, new_snv_list)
    

#adjust z so that it represents clone indices in the tree (bascially the indices in tree.nodes_except_root)
#for example, if a node has an index of x in the real tree, the index of the clone should x-1
def adjust_z(z, tree):
    z_updated = []
    for node_index in z:
        node = tree.node_list[node_index]
        clone_index = tree.nodes_except_root.index(node)
        z_updated.append(clone_index)
    
    return z_updated

"""
    returns a matrix containing information on if a clone has an SNV or not
    clone_has_snv is an k x n matrix where n = # of snvs and k is the number of clones
    if a clone has an snv, or any of its ancestor nodes have an snv, then clone_has_snv[k][n] = True
    otherwise clone_has_snv[k][n] = False
"""
def make_clone_has_snv_matrix(z, tree, num_snvs):
    k = tree.k
    clone_has_snv = [[False for _ in range(num_snvs)] for _ in range(k)] 

    for snv_index, node_index in enumerate(z):
        origin_node = tree.node_list[node_index]
        for clone_index, clone_node in enumerate(tree.nodes_except_root):
            if check_if_descendant(clone_node, origin_node):
                clone_has_snv[clone_index][snv_index] = True

    return clone_has_snv

"""
    helper method that checks whether a specific node is the descendant of another node;
    descendants may not be the direct child of the node, but can be the grandchild or great grandchild etc. 
"""
def check_if_descendant(current_node, ancestor_node):
    while current_node is not None:
        if current_node == ancestor_node:
            return True
        current_node = current_node.parent
    
    return False

"""
updates SNV assignment for bulk DNA data 
"""
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


def initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, alpha=1.0, scrna_params = None):
    node_list = get_node_list(lamb_0, lamb, gamma)

    z = assign_snvs(num_snvs, node_list)

    tree = Tree(node_list, z)

    if scrna_params is None:
        scrna_params = ScRNALikelihoodParams()

    phi_sampler = PhiSample(tree, alpha = alpha, scrna_params=scrna_params)
    phi = phi_sampler.sample_prior()

    return tree, z, phi_sampler, phi 

def compute_log_posterior(phi, bulk_snvs, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler):
    
    log_bulk = bulk_log_likelihood(bulk_snvs, phi.tolist(), epsilon)

    log_scrna = 0.0
    if scrna_data is not None:
        log_scrna = log_scrna_likelihood(scrna_data, phi.tolist(), clone_has_snv, scrna_config)

    log_prior = phi_sampler.previous_log(phi)

    log_posterior = log_bulk + log_scrna + log_prior

    return log_posterior, log_bulk, log_scrna, log_prior


def mcmc(bulk_snvs, scrna_data, lamb_0, lamb, gamma, epsilon, num_iterations, burnin, scrna_config = None):
    t0 = time.time()
    num_snvs = len(bulk_snvs)

    if scrna_config is None:
        scrna_config = ScRNALikelihoodParams()

    tree, z, phi_sampler, phi = initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, 1.0, scrna_config)
    tree_sampler = TreeSampler(tree, scrna_config)
    tree = tree_sampler.prune_empty_nodes(tree.snvs)
    

    map_tree = {"tree": tree, "phi": phi, "z": z, "log_posterior": -math.inf}

    for i in range(num_iterations):
         print ("MCMC loop ", i)
         z_clone = adjust_z(tree.snvs, tree)
         t1 = time.time(); print("adjust_z time:", t1 - t0)

         bulk_snvs_updated = update_bulk_snvs_indices(bulk_snvs, z_clone)
         t2 = time.time(); print("update_bulk_snvs_indices time:", t2 - t1)

         clone_has_snv = make_clone_has_snv_matrix(tree.snvs, tree, num_snvs)
         t3 = time.time(); print("make_clone_has_snv_matrix time:", t3 - t2)

         print("total pre-iteration time:", t3 - t0)

         phi = phi_sampler.update(phi, bulk_snvs_updated, epsilon, scrna_data, clone_has_snv)

         tree, new_snvs = tree_sampler.slice_sample_tree(phi, bulk_snvs, scrna_data, epsilon, 3)
         
         tree_sampler.tree = tree
        
         phi_sampler.tree = tree
         phi_sampler.K = tree.k

         if len(phi) != tree.k:
            # Adjust phi to match new tree size
            if tree.k > len(phi):
                # Tree grew, add new clone prevalences
                new_phi = np.zeros(tree.k)
                new_phi[:len(phi)] = phi * (1 - 0.1)  # Shrink existing
                new_phi[len(phi):] = 0.1 / (tree.k - len(phi))  # New clones
                phi = new_phi
            else:
                # Tree shrunk, remove clones
                phi = phi[:tree.k]
                phi = phi / np.sum(phi)  # Renormalize
        
         z_clone = adjust_z(tree.snvs, tree)
         bulk_snvs_updated = update_bulk_snvs_indices(bulk_snvs, z_clone)
         clone_has_snv = make_clone_has_snv_matrix(tree.snvs, tree, num_snvs)
        
         log_post, log_bulk, log_scrna, log_prior = compute_log_posterior(phi, bulk_snvs_updated, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler)

         if log_post > map_tree["log_posterior"]:
             map_tree["tree"] = tree
             map_tree["phi"] = phi.copy()
             map_tree["z"] = tree.snvs.copy()
             map_tree["log_posterior"] = log_post
    
    print(f"MAP log posterior: {map_tree['log_posterior']:.2f}")

    return map_tree


""" #==========================================================================================
#TEST ON SMALL, SYNTHETIC DATA

if __name__ == "__main__":
    # Example: Create synthetic data and run MCMC
    from bulk_dna_likelihood import SNV
    
    # Create example bulk SNVs
    bulk_snvs = [
        SNV(15, 100, 1, 1, 0),
        SNV(30, 100, 2, 1, 0),
        SNV(25, 100, 1, 1, 0),
        SNV(10, 80, 1, 0, 0),
        SNV(20, 90, 1, 1, 0),
    ]
    
    # Run MCMC
    map_result = mcmc(
        bulk_snvs=bulk_snvs,
        scrna_data=None,  # No scRNA data for this example
        lamb_0=1.0,
        lamb=0.5,
        gamma=1.0,
        epsilon=0.001,
        num_iterations=1000,
        burnin=500
    )
    
    print(f"\nMAP Tree has {map_result['tree'].k} clones")
    print(f"MAP phi: {map_result['phi']}")
    print(f"MAP log posterior: {map_result['log_posterior']:.2f}")

def main():
    pass """