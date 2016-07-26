# Author: Almero Gouws <14366037@sun.ac.za>
"""
This module contains unit tests for all methods and functions implemented
the python file graph.py.

This test module has been implemented for NOSE test framework. There is only
1 kind of test in this module:
1 - Tests that test functions.
    - These methods are named as such: test_[function name]

HOW TO USE THIS MODULE
======================

1 - Run testBNT batch file located in the PyBNT\Tests directory.

OR
==
For windows:
With NOSE installed: Use the command console to change directories
to the directory this module is in and run the nosetests script.

EXAMPLE: If package installed to C:\PyBNT
1 - Open command prompt
2 - Type: 'cd C:\PyBNT\Tests'
3 - Type: 'nosetests'
"""

import numpy as np
import graph
from numpy.testing import assert_array_equal

__docformat__ = 'restructuredtext'

class Test_dag_operations:
    """
    This class tests all functions in graph.py that require a matrix
    representing a directed acyclic graph as input.
    """
    def setUp(self):
        """
        Create an adjacency matrix representing the following directed
        four node graph, where all arrows point down, to use as input:
                            Node1
                            /   \
                           /     \
                        Node2   Node3
                           \     /
                            \   /
                            Node 4
        """
        self.n = 4
        dag = np.zeros((self.n, self.n))
        dag[0, [1, 2]] = 1
        dag[1, 3] = 1
        dag[2, 3] = 1

        self.dag = dag

    def test_parents(self):
        """
        FUNCTION: parents, in graph.py.
        """
        """Use the function to determine the parents of each node"""
        parents = []
        for i in range(0, self.n):
            parents.append(graph.parents(self.dag, i))

        """
        Assert that the function returns the expected parents for each node.
        """
        assert parents[0] == []
        assert parents[1] == [0]
        assert parents[2] == [0]
        assert parents[3] == [1, 2]

    def test_children(self):
        """
        FUNCTION: children, in graph.py.
        """
        """Use the function to determine the children of each node"""
        children = []
        for i in range(0, self.n):
            children.append(graph.children(self.dag, i))

        """
        Assert that the function returns the expected children for each node.
        """
        assert children[0] == [1, 2]
        assert children[1] == [3]
        assert children[2] == [3]
        assert children[3] == []

    def test_neighbours(self):
        """
        FUNCTION: neighbours, in graph.py.
        """
        """Use the function to determine the neighbours of each node"""
        neighbours = []
        for i in range(0, self.n):
            neighbours.append(graph.neighbours(self.dag, i))

        """
        Assert that the function returns the expected neighbours for each node.
        """
        assert neighbours[0] == [1, 2]
        assert neighbours[1] == [3, 0]
        assert neighbours[2] == [3, 0]
        assert neighbours[3] == [1, 2]

    def test_family(self):
        """
        FUNCTION: family, in graph.py.
        """
        """Use the function to determine the family of each node"""
        family = []
        for i in range(0, self.n):
            family.append(graph.family(self.dag, i))

        """
        Assert that the function returns the expected family for each node.
        """
        assert family[0] == [0]
        assert family[1] == [0, 1]
        assert family[2] == [0, 2]
        assert family[3] == [1, 2, 3]

    def test_topological_sort(self):
        """
        FUNCTION: topological_sort, in graph.py.
        """
        """Execute the function"""
        order = graph.topological_sort(self.dag)

        """Assert that the functions output matches the expected output"""
        assert order == [0, 1, 2, 3]

    def test_moralize(self):
        """
        FUNCTION: moralize, in graph.py.
        """
        """Execute function"""
        [M, moral_edges] = graph.moralize(self.dag)

        """Create data representing the expected output of the function."""
        t_M = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
        t_moral_edges = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        """Assert that the functions output matches the expected output"""
        assert_array_equal(M, t_M)
        assert_array_equal(moral_edges, t_moral_edges)

def test_setdiag():
    """
    FUNCTION: setdiag, in graph.py.
    """
    """Create data used as input for function"""
    G = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]])

    """Execute function"""
    G = graph.setdiag(G, 0)

    """Create data representing the expected output of the function."""
    t_G = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    """Assert that the functions output matches the expected output"""
    assert_array_equal(G, t_G)

def test_best_first_elim_order():
    """
    FUNCTION: best_first_elim_order, in graph.py.
    """
    """Create data used as input for function"""
    G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
    ns = np.array([[2, 2, 2, 2]])
    stages = [[0, 1, 2, 3]]

    """Execute function"""
    order = graph.best_first_elim_order(G, ns, stages)

    """Create data representing the expected output of the function."""
    t_order = np.array([[0, 1, 2, 3]])

    """Assert that the functions output matches the expected output"""
    assert_array_equal(order, t_order)

def test_triangulate():
    """
    FUNCTION: triangulate, in graph.py.
    """
    """Create data used as input for function"""
    G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
    order = np.array([[0, 1, 2, 3]])

    """Execute function"""
    [G, cliques] = graph.triangulate(G, order)

    """Create data representing the expected output of the function."""
    t_G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
    t_cliques = [np.array([0, 1, 2]), np.array([1, 2, 3])]

    """Assert that the functions output matches the expected output"""
    assert_array_equal(G, t_G)
    assert_array_equal(cliques, t_cliques)

def test_minimum_spanning_tree():
    """
    FUNCTION: mimimum_spanning_tree, in graph.py.
    """
    """Create data used as input for function"""
    C1 = np.mat([[0, -2], [-2, 0]])
    C2 = np.mat([[0, 16], [16, 0]])

    """Execute function"""
    A = graph.minimum_spanning_tree(C1, C2)

    """Create data representing the expected output of the function."""
    t_A = np.array([[0, 1], [1, 0]])

    """Assert that the functions output matches the expected output"""
    assert_array_equal(A, t_A)

def test_cliques_to_jtree():
    """
    FUNCTION: cliques_to_jtree, in graph.py.
    """
    """Create data used as input for function"""
    cliques = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    ns = np.array([[2, 2, 2, 2]])

    """Execute function"""
    [jtree, num_cliques, B, w] = graph.cliques_to_jtree(cliques, ns)

    """Create data representing the expected output of the function."""
    t_jtree = np.array([[0, 1], [1, 0]])
    t_num_cliques = 2
    t_B = np.array([[1, 1, 1, 0], [0, 1, 1, 1]])
    t_w = np.array([[8], [8]])

    """Assert that the functions output matches the expected output"""
    assert_array_equal(jtree, t_jtree)
    assert_array_equal(num_cliques, t_num_cliques)
    assert_array_equal(B, t_B)
    assert_array_equal(w, t_w)

def test_graph_to_jtree():
    """
    FUNCTION: graph_to_jtree, in graph.py.
    """
    """Create data used as input for function"""
    MG = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
    ns = np.array([[2, 2, 2, 2]])
    partial_order = []
    stages = [[0, 1, 2, 3]]
    clusters = []

    """Execute function"""
    [jtree, root, cliques, B, w] = graph.graph_to_jtree(MG, ns)

    """Create data representing the expected output of the function."""
    t_jtree = np.array([[0, 1], [1, 0]])
    t_root = 2
    t_cliques = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    t_B = np.array([[1, 1, 1, 0], [0, 1, 1, 1]])
    t_w = np.array([[8], [8]])

    """Assert that the functions output matches the expected output"""
    assert_array_equal(jtree, t_jtree)
    assert root == t_root
    assert_array_equal(cliques, t_cliques)
    assert_array_equal(B, t_B)
    assert_array_equal(w, t_w)

def test_mk_rooted_tree():
    """
    FUNCTION: mk_rooted_tree, in graph.py.
    """
    """Create data used as input for function"""
    G = np.array([[0, 1], [1, 0]])
    root = 1

    """Execute function"""
    [T, pre, post, cycle] = graph.mk_rooted_tree(G, root)

    """Create data representing the expected output of the function."""
    t_T = np.array([[0, 0], [1, 0]])
    t_pre = [1, 0]
    t_post = [0, 1]
    t_cycle = 0

    """Assert that the functions output matches the expected output"""
    assert_array_equal(T, t_T)
    assert pre == t_pre
    assert post == t_post
    assert cycle == t_cycle
