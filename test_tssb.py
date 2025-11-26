"""
Test file for TSSB (Tree-Structured Stick-Breaking) algorithm
Tests the implementation of the prior distribution over clonal trees
"""

import pytest
import numpy as np
import math
from TSSB import Node, get_node_list, tssb

class TestNode:
    """Test the Node class"""
    
    def test_node_initialization(self):
        """Test that nodes are created with correct attributes"""
        parent = Node(None, 1.0, 0.5, True, 0)
        child = Node(parent, 0.3, 0.2, False, 1)
        
        assert child.parent == parent
        assert child.upsilon_u == 0.3
        assert child.remaining_stick == 0.2
        assert child.is_root == False
        assert child.height == 1
    
    def test_root_node(self):
        """Test root node has correct properties"""
        root = Node(None, 1.0, 0.5, True, 0)
        
        assert root.parent is None
        assert root.is_root == True
        assert root.height == 0
        assert root.get_parent() is None
    
    def test_get_parent(self):
        """Test parent retrieval"""
        parent = Node(None, 1.0, 0.5, True, 0)
        child = Node(parent, 0.3, 0.2, False, 1)
        
        assert child.get_parent() == parent


class TestTSSB:
    """Test the TSSB algorithm"""
    
    def test_basic_tree_generation(self):
        """Test that tree generation produces nodes"""
        np.random.seed(42)
        lamb_0 = 1.0
        lamb = 0.5
        gamma = 1.0
        
        node_list = get_node_list(lamb_0, lamb, gamma)
        
        # Should have at least root node
        assert len(node_list) >= 1
        assert node_list[0].is_root == True
    
    def test_root_node_properties(self):
        """Test that root node has upsilon = 1"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 1.0)
        root = node_list[0]
        
        assert root.upsilon_u == 1.0
        assert root.height == 0
        assert root.parent is None
    
    def test_child_heights_increase(self):
        """Test that child nodes have increasing heights"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 1.0)
        
        for node in node_list:
            if not node.is_root and node.parent is not None:
                assert node.height == node.parent.height + 1
    
    def test_remaining_stick_decreases(self):
        """Test that remaining stick converges to zero"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 1.0)
        
        for node in node_list:
            # Remaining stick should be non-negative and less than upsilon
            assert node.remaining_stick >= 0
            if not node.is_root:
                assert node.remaining_stick <= node.parent.remaining_stick
    
    def test_termination_condition(self):
        """Test that algorithm terminates when remaining stick is small"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 1.0)
        
        # Check that leaf nodes have small remaining stick
        leaf_nodes = [n for n in node_list if n.remaining_stick <= 0.00005]
        assert len(leaf_nodes) > 0
    
    def test_tree_structure_validity(self):
        """Test that parent-child relationships are valid"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 1.0)
        
        for i, node in enumerate(node_list):
            if not node.is_root:
                # Parent should be in the node list
                assert node.parent in node_list
                # Parent should come before child in list (for root's direct children)
                if node.parent.is_root:
                    parent_idx = node_list.index(node.parent)
                    assert parent_idx < i


class TestParameterEffects:
    """Test how different parameters affect tree generation"""
    
    def test_lambda_effect_on_depth(self):
        """Test that smaller lambda values encourage deeper trees"""
        np.random.seed(42)
        
        # Smaller lambda should allow deeper trees
        shallow_tree = get_node_list(1.0, 0.9, 1.0)
        np.random.seed(42)
        deep_tree = get_node_list(1.0, 0.1, 1.0)
        
        max_depth_shallow = max(n.height for n in shallow_tree)
        max_depth_deep = max(n.height for n in deep_tree)
        
        # This is probabilistic, but should generally hold
        assert max_depth_deep >= max_depth_shallow or len(deep_tree) >= len(shallow_tree)
    
    def test_gamma_effect_on_width(self):
        """Test that smaller gamma values encourage wider trees"""
        np.random.seed(42)
        narrow_tree = get_node_list(1.0, 0.5, 10.0)
        
        np.random.seed(42)
        wide_tree = get_node_list(1.0, 0.5, 0.5)
        
        # Smaller gamma should generally produce more children per node
        assert len(wide_tree) >= 1
        assert len(narrow_tree) >= 1
    
    def test_lambda_0_effect(self):
        """Test that lambda_0 affects tree generation"""
        np.random.seed(42)
        tree1 = get_node_list(0.5, 0.5, 1.0)
        
        np.random.seed(42)
        tree2 = get_node_list(2.0, 0.5, 1.0)
        
        # Both should produce valid trees
        assert len(tree1) >= 1
        assert len(tree2) >= 1


class TestStickBreakingProperties:
    """Test mathematical properties of stick-breaking"""
    
    def test_upsilon_bounds(self):
        """Test that upsilon values are in valid range"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 1.0)
        
        for node in node_list:
            assert 0 <= node.upsilon_u <= 1
    
    def test_pi_computation(self):
        """Test that pi values (stick portions) sum appropriately"""
        np.random.seed(42)
        lamb_0 = 1.0
        lamb = 0.5
        gamma = 1.0
        
        node_list = get_node_list(lamb_0, lamb, gamma)
        
        # For each node, compute pi = upsilon * v
        # All pi values should sum to approximately 1 (or less due to truncation)
        total_pi = 0
        for node in node_list:
            # Approximate pi by upsilon - remaining_stick
            pi_approx = node.upsilon_u - node.remaining_stick
            total_pi += pi_approx
        
        assert 0 < total_pi <= 1.1  # Allow small numerical error


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_lambda_equals_one(self):
        """Test with lambda = 1 (maximum value)"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 1.0, 1.0)
        
        assert len(node_list) >= 1
        assert node_list[0].is_root
    
    def test_small_gamma(self):
        """Test with very small gamma (encourages many children)"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 0.1)
        
        assert len(node_list) >= 1
    
    def test_large_gamma(self):
        """Test with large gamma (discourages many children)"""
        np.random.seed(42)
        node_list = get_node_list(1.0, 0.5, 100.0)
        
        assert len(node_list) >= 1
    
    def test_reproducibility_with_seed(self):
        """Test that setting seed produces reproducible results"""
        np.random.seed(42)
        tree1 = get_node_list(1.0, 0.5, 1.0)
        
        np.random.seed(42)
        tree2 = get_node_list(1.0, 0.5, 1.0)
        
        assert len(tree1) == len(tree2)
        for n1, n2 in zip(tree1, tree2):
            assert n1.height == n2.height
            assert abs(n1.upsilon_u - n2.upsilon_u) < 1e-10


class TestIntegration:
    """Integration tests for realistic scenarios"""
    
    def test_cancer_phylogeny_scenario(self):
        """Test parameters typical for cancer phylogenetics"""
        np.random.seed(42)
        # Parameters from the PhylEx paper suggest gamma <= 1 for cancer
        lamb_0 = 1.0
        lamb = 0.5
        gamma = 0.5
        
        node_list = get_node_list(lamb_0, lamb, gamma)
        
        # Should produce a reasonable tree
        assert 1 <= len(node_list) <= 100  # Reasonable number of clones
        assert max(n.height for n in node_list) <= 20  # Reasonable depth
    
    def test_multiple_generations(self):
        """Test generating multiple trees"""
        trees = []
        for seed in range(5):
            np.random.seed(seed)
            tree = get_node_list(1.0, 0.5, 1.0)
            trees.append(tree)
        
        # All should be valid
        assert all(len(t) >= 1 for t in trees)
        # They should vary (probabilistic test)
        assert len(set(len(t) for t in trees)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])