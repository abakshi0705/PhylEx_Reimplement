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
    """
    def __init__(self, parent, upsilon_u, remaining_stick, is_root, height):
        self.parent = parent
        self.upsilon_u = upsilon_u
        self.remaining_stick = remaining_stick
        self.is_root = is_root
        self.height = height
        self.pi_u = upsilon_u - remaining_stick

    def get_parent(self):
        return self.parent


def get_node_list(lamb_0, lamb, gamma):
    #node list will hold all the nodes in our tree
    node_list = []  

    # initialize the root node as our base node
    # the root node is the healthy non-cancerous cell and has exactly one child, which is the first cancer (progenitor) cell
    root_node = Node(None, 1.0, 1.0, True, 0)
    node_list.append(root_node)


    # upsilon represents the proportion of the stick able to be broken by a node; this should equal 1 for the progenitor
    upsilon = 1.0     
    v_progenitor = np.random.beta(1, lamb_0 * math.pow(lamb, 1))
    pi_u = upsilon * v_progenitor
    remaining_stick = (1-pi_u) * upsilon
    progenitor_node = Node(root_node, upsilon, remaining_stick, True, 1)
    node_list.append(progenitor_node)

    child_list = tssb(progenitor_node, lamb_0, lamb, gamma)
    node_list.extend(child_list)

    return node_list



""" requires 0 < lamb_0
requires 0 < lamb <=1
requires gamma > 0  """
def tssb(node, lamb_0, lamb, gamma):
    k = 0
    psi_array = []
    height = node.height + 1
    remaining_stick = node.remaining_stick

    node_list =[]

    while (remaining_stick > 0.005):
        psi = np.random.beta(1, gamma)
        psi_array.append(psi)
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


# n: number of snvs
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
    pi_values = pi_values / np.sum(pi_values)

    #randomly a node based on the pi_value for that node to assign an SNV to
    #index of z represents SNV number
    z = np.random.choice(non_root_indices, size=n, p=pi_values)

    return z.to_list()


def get_snvs_for_node(z, node_index):
    return [n for n, assigned_node in enumerate(z) if assigned_node == node_index]


#gets a list of all SNV indices present at a specific node, including the ancestral SNVs
def get_node_genotypes(node_list, z, node_index):
    genotype = []
    current_node = node_list(node_index)

    while(not current_node.is_root):
        genotype.extend(get_snvs_for_node(z, node_list.index(current_node)))
        current_node = current_node.parent

    return sorted(genotype)
