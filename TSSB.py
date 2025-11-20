""" Reimplementation of the TSSB algorithm to generate a prior for the distribution of clonal trees """



import numpy as np
import math


class Node():
    def __init__(self, name, parent, branch_length, is_root):
        self.name = name
        self.parent = parent
        self.branch_length = branch_length
        self.is_root = is_root

    def get_parent(self):
        return self.parent

    def get_height(self):
        parent = self.parent
        counter = 1
        while parent != None:
            parent = parent.get_parent()
            counter += 1
        return counter


def get_node_list(lamb_0, lamb, gamma):
    upsilon = 1
    v = np.random.beta(1, lamb_0 * math.pow(lamb, 1))
    pi_u = upsilon * v
    remaining_stick = (1-pi_u) * upsilon
    
    k = 0
    while(remaining_stick > 0):
        psi = np.random.beta(1, gamma)
        if k >= 1:
            
        upsilon_k = 


    node_list = tssb(remaining_stick, )



""" requires 0 < lamb_0
requires 0 < lamb <=1
requires gamma > 0  """
def tssb(upsilon, node, lamb_0, lamb, gamma):
   if upsilon == 0:
       return []
   else:
       u = node.get_height()
       v = np.random.beta(1, lamb_0 * math.pow(lamb, u))
       pi_u = v * 
        


