
import numpy as np
import math
from clone_prevalences import PhiSample, dirichlet_log
from TSSB import Node, Simplified_Node, get_node_list, assign_snvs, initialize_fixed_tree_snv_assignment
from bulk_dna_likelihood import SNV, bulk_log_likelihood, snv_log_likelihood
from scrna import log_scrna_likelihood, ScRNALikelihoodParams, log_likelihood_cell_given_clone
import time

"""
Tree class will hold:
    node_list: a list of all the nodes in the tree, including the non-cancerous root node
    snvs: is a list with size equal to the number of snvs; snvs[i] will hold the clone that the ith snv is assigned to

    nodes_except_root: a list of all nodes in the tree excluding the non-cancerous root node; made because it integrates better
        with how clone_prevalences was setup
    k: length of nodes_except_root
"""
class Tree:
    def __init__(self, node_list, snvs):
        #the full list of all nodes in the tree, including the non-cancerous root node
        self.node_list = node_list
        self.snvs = snvs

        #all the nodes in the tree excluding the non-cancerous root node
        self.nodes_except_root = [node for node in node_list if not node.is_root]
        self.k = len(self.nodes_except_root)

    def __eq__(self, other):
        if not isinstance(other, Tree):
            return False
        if self.node_list == other.node_list and self.snvs == other.snvs:
            if self.nodes_except_root == other.nodes_except_root:
                if self.k == other.k:
                    return True
        
        return False
             
    
    #returnes the index of a node in the full tree, given a clone index
    #a clone index would be an index from self.nodes_except_root
    def node_index_full_tree(self, clone_index):
        return self.node_list.index(self.nodes_except_root[clone_index])
    
    def to_newick(self):
        node_to_index = {node: i for i, node in enumerate(self.node_list)}
        #build child adjacency matrix
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
    
"""
TreeSampler is a class for slice sampling a tree; we use this to explore SNV assignment in the tree
the TreeSampler holds:
    tree: the tree for which SNV assignment is being explored
    scrna_params: any scrna parameters that were initially loaded into the progroam

"""
class TreeSampler:
    def __init__(self, tree, scrna_params=None):
        self.tree = tree
        self.scrna_params = scrna_params if scrna_params else ScRNALikelihoodParams()
        print("initial tree ", self.tree.to_newick())

    def slice_sample_tree(self, phi, bulk_snvs, scrna_data, epsilon, min_depth):
        #this list will hold all th new snv assignments that we make
        new_snvs = self.tree.snvs.copy()

        for n in range(len(self.tree.snvs)):
            new_snvs[n] = self.sample_snv_assignment(n, new_snvs, phi, bulk_snvs, scrna_data, epsilon)
        
        self.tree.snvs = new_snvs

        #We created this code with an older implementation of our TSSB algorithm which generated not only
        #a prior distribution for SNV assignment, but also for the structure of our tree
        #in this case, we wanted to prune our tree structures if they ever had nodes with no snv assignment,
        #as our original TSSB implementation could sometimes generate a prior tree that was very large
        """ #create a new tree with the new_snvs
        new_tree = self.prune_empty_nodes(new_snvs)

        tree_depth = max(node.height for node in new_tree.node_list)
        print("tree depth ", tree_depth)
    
        if tree_depth < min_depth:
            # Tree too shallow, reject and keep old tree
            #print(f"  Warning: Rejected tree with depth {tree_depth} < {min_depth}")
            return self.tree, self.tree.snvs """
    
        return self.tree, self.tree.snvs


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
        old_log = np.log(node.pi_u + 1e-10) #add 1e-10 in case pi_u is 0

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
    #note, we don't end up using this method in our current implementation; however we left it here in case we wanted
    #switch back to using the older implementation. 
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


"""
This function will initialize a prior tree with the original implementation of our TSSB algorithm, which
generates a prior tree of arbitrary size
Also initializes a prior distribution for cellular prevalences

returns:
    tree - tree
    z - SNV assignments to clones in the tree
    phi_sampler - an object used to sample phi (cellular prevalence) values using Metropolis Hastings
    phi - an array containing cellular prevalence values for each clone
"""
def initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, alpha=1.0, scrna_params = None):
    node_list = get_node_list(lamb_0, lamb, gamma)

    z = assign_snvs(num_snvs, node_list)

    tree = Tree(node_list, z)

    if scrna_params is None:
        scrna_params = ScRNALikelihoodParams()

    phi_sampler = PhiSample(tree, alpha = alpha, scrna_params=scrna_params)
    phi = phi_sampler.sample_prior()

    return tree, z, phi_sampler, phi 

"""
This function will initialize a tree that already has a known prior structure (in this case a cherry shaped tree)
important note: trees with known prior structure will use a Simplified_Node, which contains a field for the nodes children since
    these are known before hand; this is necessary to work with our refined implementation of TSSB

returns:
    tree - tree
    z - SNV assignments to clones in the tree
    phi_sampler - an object used to sample phi (cellular prevalence) values using Metropolis Hastings
    phi - an array containing cellular prevalence values for each clone
"""

def initialize_prior_fixed_tree(lamb_0, lamb, gamma, num_snvs, alpha = 1.0, scrna_params = None):
    node_list = []
    root_node = Simplified_Node(None, None, 0, 0, True, 0)
    progenitor_node = Simplified_Node(root_node, None, 0, 0, False, 1)
    left_branch_node = Simplified_Node(progenitor_node, None, 0, 0, False, 2)
    right_branch_node = Simplified_Node(progenitor_node, None, 0, 0, False, 2)
    
    root_node.children = [progenitor_node]
    progenitor_node.children = [left_branch_node, right_branch_node]

    node_list.extend([root_node, progenitor_node, left_branch_node, right_branch_node])

    #note use of different initializer
    initialize_fixed_tree_snv_assignment(node_list, lamb_0, lamb, gamma)

    z = assign_snvs(num_snvs, node_list)

    tree = Tree(node_list, z)

    if scrna_params is None:
        scrna_params = ScRNALikelihoodParams()

    phi_sampler = PhiSample(tree, alpha = alpha, scrna_params=scrna_params)
    phi = phi_sampler.sample_prior()

    return tree, z, phi_sampler, phi 

"""Computes the posterior probability for a tree given our updated phi samples"""
def compute_log_posterior(phi, bulk_snvs, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler):
    
    log_bulk = bulk_log_likelihood(bulk_snvs, phi.tolist(), epsilon)

    log_scrna = 0.0
    if scrna_data is not None:
        log_scrna = log_scrna_likelihood(scrna_data, phi.tolist(), clone_has_snv, scrna_config)

    log_prior = phi_sampler.prior_log(phi)

    log_posterior = log_bulk + log_scrna + log_prior

    return log_posterior, log_bulk, log_scrna, log_prior

"""This is the main part of the algorithm; we use MCMC to explore SNV and cellular prevalence assignments"""
def mcmc(bulk_snvs, scrna_data, lamb_0, lamb, gamma, epsilon, num_iterations, burnin, use_fixed_tree, scrna_config = None, phi_init = None):
    #t0 = time.time()
    num_snvs = len(bulk_snvs)

    if scrna_config is None:
        scrna_config = ScRNALikelihoodParams()

    if use_fixed_tree:
        tree, z, phi_sampler, phi = initialize_prior_fixed_tree(lamb_0, lamb, gamma, num_snvs, 1.0, scrna_config)
        if phi_init is not None:
            phi = phi_init.copy()
    else:
        #if using an unfixed tree, we want to enforce a minimum tree depth so that our prior tree doesn't initialize with 1 node
        #we do this because our tree structure can only shrink (through the prune function), not grow
        tree, z, phi_sampler, phi = initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, 1.0, scrna_config)
        tree_depth = max(node.height for node in tree.node_list)
        while tree_depth < 2:
            tree, z, phi_sampler, phi = initialize_prior_tree(lamb_0, lamb, gamma, num_snvs, 1.0, scrna_config)
            tree_depth = max(node.height for node in tree.node_list)

    tree_sampler = TreeSampler(tree, scrna_config)

    chain_of_trees = []
    
    #map_tree holds the maximum a posteriori tree, along with its associated snv and cell prev. assignments, and posterior prob.
    map_tree = {"tree": tree, "phi": phi, "z": z, "log_posterior": -math.inf}

    for i in range(num_iterations):
         if i % 5 == 0:
             print ("MCMC loop ", i)
         #old_tree_k = tree.k
         #old_tree_id = id(tree)
         
         new_tree, new_snvs = tree_sampler.slice_sample_tree(phi, bulk_snvs, scrna_data, epsilon, 2)
         #tree = tree_sampler.slice_sample_tree(phi, bulk_snvs, scrna_data, epsilon, 2)
         
         #tree_changed = (id(new_tree) != old_tree_id) or (new_tree.k != old_tree_k)
         tree = new_tree

         phi_sampler.tree = tree
         phi_sampler.K = tree.k

         z_clone = adjust_z(tree.snvs, tree)
         bulk_snvs_updated = update_bulk_snvs_indices(bulk_snvs, z_clone)
         clone_has_snv = make_clone_has_snv_matrix(tree.snvs, tree, num_snvs)

         phi = phi_sampler.update(phi, bulk_snvs_updated, epsilon, scrna_data, clone_has_snv)



         """  #first adjust our SNV assignment index so that it works with the phi_sampler
         old_tree_k = tree.k
         old_tree_id = id(tree)

         z_clone = adjust_z(tree.snvs, tree)
         bulk_snvs_updated = update_bulk_snvs_indices(bulk_snvs, z_clone)
         

         clone_has_snv = make_clone_has_snv_matrix(tree.snvs, tree, num_snvs)
     
         phi_temp = phi_sampler.update(phi, bulk_snvs_updated, epsilon, scrna_data, clone_has_snv)

         old_tree_k = tree.k
         old_tree_id = id(tree)  

         new_tree, new_snvs = tree_sampler.slice_sample_tree(phi_temp, bulk_snvs, scrna_data, epsilon, 2)
         
         tree_changed = (id(new_tree) != old_tree_id) or (new_tree.k != old_tree_k)

         if not tree_changed:
             #print("reached a")
             phi = phi_temp
             tree = tree
         else:
             #print("reached b")
             phi = phi_temp
             tree = new_tree """

         #tree_sampler.tree = tree
        
         #phi_sampler.tree = tree
         #phi_sampler.K = tree.k


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
        

         """ z_clone = adjust_z(tree.snvs, tree)
         bulk_snvs_updated = update_bulk_snvs_indices(bulk_snvs, z_clone)
         clone_has_snv = make_clone_has_snv_matrix(tree.snvs, tree, num_snvs) """
        
         

         log_post, log_bulk, log_scrna, log_prior = compute_log_posterior(phi, bulk_snvs_updated, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler)

         sampled_tree = {"tree": tree, "phi": phi.copy(), "z": tree.snvs.copy(), "log_posterior": log_post,
                         "log_bulk": log_bulk, "log_scrna": log_scrna, "log_prior": log_prior}
         
         chain_of_trees.append(sampled_tree)

         if log_post > map_tree["log_posterior"]:
             map_tree["tree"] = tree
             map_tree["phi"] = phi.copy()
             map_tree["z"] = tree.snvs.copy()
             map_tree["log_posterior"] = log_post
    
    print(f"MAP log posterior: {map_tree['log_posterior']:.2f}")

    return map_tree, chain_of_trees


def mcmc_fixed_snv(bulk_snvs, scrna_data, epsilon, num_iterations, z, scrna_config = None, phi_init = None):
    num_snvs = len(bulk_snvs)
    if scrna_config is None:
        scrna_config = ScRNALikelihoodParams()
    
    #initialize a prior fixed tree
    node_list = []
    root_node = Simplified_Node(None, None, 0, 0, True, 0)
    progenitor_node = Simplified_Node(root_node, None, 0, 0, False, 1)
    left_branch_node = Simplified_Node(progenitor_node, None, 0, 0, False, 2)
    right_branch_node = Simplified_Node(progenitor_node, None, 0, 0, False, 2)
    
    root_node.children = [progenitor_node]
    progenitor_node.children = [left_branch_node, right_branch_node]

    node_list.extend([root_node, progenitor_node, left_branch_node, right_branch_node])

    fixed_tree = Tree(node_list=node_list, snvs=z)

    #initialize phi
    if phi_init is not None:
        phi = phi_init.copy()

    phi_sampler = PhiSample(fixed_tree, alpha = 1.0, scrna_params=scrna_config)

    chain_of_trees = []
    map_tree = {"tree": fixed_tree, "phi": phi, "z": z, "log_posterior": -math.inf}

    clone_has_snv = make_clone_has_snv_matrix(fixed_tree.snvs, fixed_tree, num_snvs)

    for i in range(num_iterations):
         if i % 5 == 0:
             print ("MCMC loop ", i)

         phi = phi_sampler.update(phi, bulk_snvs, epsilon, scrna_data, clone_has_snv)

         if len(phi) != fixed_tree.k:
                # Adjust phi to match new tree size
                if fixed_tree.k > len(phi):
                    # Tree grew, add new clone prevalences
                    new_phi = np.zeros(fixed_tree.k)
                    new_phi[:len(phi)] = phi * (1 - 0.1)  # Shrink existing
                    new_phi[len(phi):] = 0.1 / (fixed_tree.k - len(phi))  # New clones
                    phi = new_phi
                else:
                    # Tree shrunk, remove clones
                    phi = phi[:fixed_tree.k]
                    phi = phi / np.sum(phi)  # Renormalize
     

         log_post, log_bulk, log_scrna, log_prior = compute_log_posterior(phi, bulk_snvs, scrna_data, clone_has_snv, epsilon, scrna_config, phi_sampler)

         sampled_tree = {"tree": fixed_tree, "phi": phi.copy(), "z": fixed_tree.snvs.copy(), "log_posterior": log_post,
                            "log_bulk": log_bulk, "log_scrna": log_scrna, "log_prior": log_prior}
            
         chain_of_trees.append(sampled_tree)

         if log_post > map_tree["log_posterior"]:
            map_tree["tree"] = fixed_tree
            map_tree["phi"] = phi.copy()
            map_tree["z"] = fixed_tree.snvs.copy()
            map_tree["log_posterior"] = log_post
        
    print(f"MAP log posterior: {map_tree['log_posterior']:.2f}")

    return map_tree, chain_of_trees
    

    
