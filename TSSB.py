""" Reimplementation of the TSSB algorithm to generate a prior for the distribution of clonal trees and assign SNVs
to specific clones in the tree """

import numpy as np
import math

class Node():
    """
        parent: the parent of this node
        upsilon_u: the portion of the stick available to be broken up by the node
        remaining_stick: the portion of the stick that remains after this node breaks off its piece
        is_root: if the node is the root or not
        height: the height of this node in the tree
        snvs: snvs assigned to this node; snvs assigned after node has been created
    """
    def __init__(self, parent, upsilon_u, remaining_stick, is_root, height):
        self.parent = parent
        self.upsilon_u = upsilon_u
        self.remaining_stick = remaining_stick
        self.is_root = is_root
        self.height = height
        self.pi_u = upsilon_u - remaining_stick
        self.snvs = []

    def get_parent(self):
        return self.parent


"""
gets list of nodes for a specific tree, based on parameters lamb_0, lamb, gamma

requires 0 < lamb_0
requires 0 < lamb <=1
requires gamma > 0  
"""
def get_node_list(lamb_0, lamb, gamma):
    node_list = []  

    # initialize the root node as our base node
    # the root node is the healthy non-cancerous cell and has exactly one child, which is the first cancer (progenitor) cell
    root_node = Node(None, 1.0, 1.0, True, 0)
    node_list.append(root_node)

    # initialize the progenitor cell
    # upsilon represents the proportion of the stick able to be broken by a node; this should equal 1 for the progenitor
    upsilon = 1.0     
    v_progenitor = np.random.beta(1, lamb_0 * math.pow(lamb, 1))
    pi_u = upsilon * v_progenitor
    remaining_stick = (1-pi_u) * upsilon
    # progenitor is a cancer clone (not the healthy root), so is_root should be False
    progenitor_node = Node(root_node, upsilon, remaining_stick, False, 1)
    node_list.append(progenitor_node)

    child_list = tssb(progenitor_node, lamb_0, lamb, gamma)
    node_list.extend(child_list)

    return node_list


""" 
Recursive helper function that generates all the children of a given node, and those childrens' nodes until
remaining stick is smaller than a threshold value.

requires 0 < lamb_0
requires 0 < lamb <=1
requires gamma > 0  
"""
def tssb(node, lamb_0, lamb, gamma):
    k = 0
    height = node.height + 1
    remaining_stick = node.remaining_stick

    node_list =[]

    while (remaining_stick > 0.005):
        psi = np.random.beta(1, gamma)
        upsilon_k = remaining_stick * psi
    
        v_k = np.random.beta(1, lamb_0 * math.pow(lamb, height))
        pi_k = upsilon_k * v_k
        
        child_remaining_stick = (1-pi_k)*upsilon_k
        child_node = Node(node, upsilon_k, child_remaining_stick, False, height)
        
        node_list.append(child_node)

        node_list.extend(tssb(child_node, lamb_0, lamb, gamma))

        k+=1
        remaining_stick *= (1 - psi)

    return node_list

"""
Assign SNVs to a clone
n: number of snvs

the probability of an snv being assigned to a clone is equal to that clones pi value, which was found in the tssb algorithm
"""
def assign_snvs(n, node_list):
    non_root_nodes = []
    non_root_indices = []

    i = 0
    for node in node_list:
        if not node.is_root:
            non_root_nodes.append(node)
            non_root_indices.append(i)
        i +=1 

    pi_values = []
    for node in non_root_nodes:
        pi_values.append(node.pi_u)

    pi_values = np.array(pi_values)
    total = np.sum(pi_values)
    if total == 0 or len(pi_values) == 0:
        raise ValueError(
            "No non-root nodes with positive pi values found when assigning SNVs. Check tree generation"
        )
    pi_values = pi_values / total

    # randomly a node based on the pi_value for that node to assign an SNV to
    # index of z represents SNV number
    z = np.random.choice(non_root_indices, size=n, p=pi_values)

    return z.tolist()


"""
gets the snvs for a specific node index 
"""
def get_snvs_for_node(z, node_index):
    return [n for n, assigned_node in enumerate(z) if assigned_node == node_index]


# gets a list of all SNV indices present at a specific node, including the ancestral SNVs
def get_node_genotypes(node_list, z, node_index):
    genotype = []
    current_node = node_list(node_index)

    while(not current_node.is_root):
        genotype.extend(get_snvs_for_node(z, node_list.index(current_node)))
        current_node = current_node.parent

    return sorted(genotype)


class Simplified_Node():
    def __init__(self, parent, children, remaining_stick, pi, is_root, height):
        self.parent = parent
        self.children = children
        self.remaining_stick = remaining_stick
        self.pi_u = pi
        self.is_root = is_root
        self.height = height


def initialize_fixed_tree_snv_assignment(node_list, lamb_0, lamb, gamma):
    root_node = node_list[0]
    node = root_node.children[0]
    
    upsilon_progenitor = 1
    v_progenitor = np.random.beta(1, lamb_0 * math.pow(lamb, 1))
    pi_progenitor = upsilon_progenitor * v_progenitor
    remaining_stick = (1-pi_progenitor) * upsilon_progenitor
    
    node.pi_u = pi_progenitor
    node.remaining_stick = remaining_stick


    def tssb_recurse(node, lamb_0, lamb, gamma):
        children = node.children
        if not children:
            return
        
        rs = node.remaining_stick

        for child in children:
            psi = np.random.beta(1, gamma)
            upsilon_child = rs * psi
            v_child = np.random.beta(1, lamb_0 * math.pow(lamb, child.height))

            pi_child = upsilon_child * v_child
            child.pi_u = pi_child
            child.remaining_stick = upsilon_child * (1 - v_child)

            rs = rs * (1 - psi)

            tssb_recurse(child, lamb_0, lamb, gamma)


    tssb_recurse(node, lamb_0, lamb, gamma)



