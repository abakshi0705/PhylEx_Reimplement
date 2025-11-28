""" Reimplementation of the TSSB algorithm to generate a prior for the distribution of clonal trees """
import numpy as np
import math


class Node():
    def __init__(self, parent, upsilon_u, remaining_stick, is_root, height):
        self.parent = parent
        self.upsilon_u = upsilon_u
        self.remaining_stick = remaining_stick
        self.is_root = is_root
        self.height = height

    def get_parent(self):
        return self.parent


def get_node_list(lamb_0, lamb, gamma):

    node_list = []  #node list will hold all the nodes in our tree

 
    # initialize the root node as our base node
    # root node is able to break up the entire stick so upsilon equals 1

    upsilon = 1     
    v = np.random.beta(1, lamb_0 * math.pow(lamb, 0))
    pi_u = upsilon * v
    remaining_stick = (1-pi_u) * upsilon
    root_node = Node(None, upsilon, remaining_stick, True, 0)
    node_list.append(root_node)

    child_list = tssb(root_node, lamb_0, lamb, gamma)

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
