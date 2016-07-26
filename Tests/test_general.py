# Author: Almero Gouws <14366037@sun.ac.za>
"""
This module contains unit tests for all methods and functions implemented
the python file general.py.

This test module has been implemented for NOSE test framework. There is only
1 kind of test in this module:
1 - Test functions
    - These methods are named as such: test_[function name]
"""

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import models
import general
import cpds
from utilities import read_samples

__docformat__ = 'restructuredtext'

class Test_bnet_operations:
    """
    This class tests all functions contained in the general.py that require
    a bnet object as input.
    """
    def setUp(self):
        """Create all data required to instantiate the bnet object"""
        nodes = 4
        dag = np.zeros((nodes, nodes))
        C = 0
        S = 1
        R = 2
        W = 3
        dag[C, [R, S]] = 1
        dag[R, W] = 1
        dag[S, W] = 1
        ns = 2 * np.ones((1, nodes))

        """Instantiate the CPD for each node in the network"""
        node_cpds = [[], [], [], []]
        CPT = np.array([0.5, 0.5])
        node_cpds[C] = cpds.tabular_CPD(C, ns, dag, CPT)
        CPT = np.array([[0.8, 0.2], [0.2, 0.8]])
        node_cpds[R] = cpds.tabular_CPD(R, ns, dag, CPT)
        CPT = np.array([[0.5, 0.5], [0.9, 0.1]])
        node_cpds[S] = cpds.tabular_CPD(S, ns, dag, CPT)
        CPT = np.array([[[1, 0], [0.1, 0.9]], [[0.1, 0.9], [0.01, 0.99]]])
        node_cpds[W] = cpds.tabular_CPD(W, ns, dag, CPT)

        """Instantiate the object"""
        self.net = models.bnet(dag, ns, node_cpds)

def test_determine_observed():
    """
    FUNCTION: determine_observed, in general.py.
    """
    """Create data required to test function"""
    evidence = [[], [], 1, 2 ,[], 3]

    """Execute function"""
    [hidd, obs] = general.determine_observed(evidence)

    """Assert the function executed correctly"""
    assert hidd == [0, 1, 4]
    assert obs == [2, 3, 5]

def test_mysetdiff():
    """
    FUNCTION: mysetdiff, in general.py.
    """
    A = np.array([0, 1, 2, 3])
    B = np.array([])

    """When the first set is empty, the result should be equal to []"""
    res = general.mysetdiff(B, A)
    assert_array_equal(res, np.array([]))

    """When the second set is empty, the result should be equal to the first set"""
    res = general.mysetdiff(A, B)
    assert_array_equal(res, A)

    """When neither set is empty, the result should be the set difference"""
    B = np.array(([0, 2]))
    res = general.mysetdiff(A, B)
    assert_array_equal(res, np.array([1, 3]))

    B = A.copy()
    res = general.mysetdiff(A, B)
    assert_array_equal(res, np.array([]))

def test_issubset():
    """
    FUNCTION: mysubset, in general.py.
    """
    """Create arrays small and large, where small IS a subset of large."""
    small = np.array(([1, 3]))
    large = np.array(([0, 1, 2, 3]))

    """Assert that small is a subset of large."""
    assert general.issubset(small, large)

    """Assert that large is not a subset of small."""
    assert not (general.issubset(large, small))

    """
    Create array small so that it contains less elements than large,
    but is not a subset of large.
    """
    small = np.array(([1, 4]))

    """Assert that small is not a subset of large."""
    assert not (general.issubset(small, large))

def test_find():
    """
    FUNCTION: find, in general.py.
    """
    assert_array_equal(general.find(np.array([[True, False]])), np.array([[0]]))
    assert_array_equal(general.find(np.array([[False, True, True, False]])), np.array([[1, 2]]))
    assert_array_equal(general.find(np.array([[False, False]])), np.array([[]]))

def test_mk_stochastic():
    """
    FUNCTION: mk_stochastic, in general.py
    """
    """Define input values"""
    mat = np.random.random((5, 5, 5, 5, 5))

    """Execute function"""
    mat = general.mk_stochastic(mat)

    """Assert that the sum over the last dimension is 1"""
    ans = []
    for i in range(0, 5):
        for j in range(0, 5):
            for k in range(0, 5):
                for l in range(0, 5):
                    ans.append(np.sum(mat[i, j, k, l, :]))

    assert not sum(np.round(np.array(ans), 3) != 1)

def test_compute_counts():
    """
    FUNCTION: compute_counts, in general.py.
    """
    """Create data required to test function"""
    data = read_samples('lawn_samples.txt')
    sz = np.array([2, 2, 2, 2])

    """Execute function"""
    count = general.compute_counts(data, sz)

    """Assert the function executed correctly"""
    assert_array_equal(count, np.array([[[[7, 0], \
                                          [0, 1]],\
                                         [[1, 14], \
                                          [0, 1]]],
                                        [[[4, 0], \
                                          [3, 14]],\
                                         [[0, 1], \
                                          [0, 4]]]]))

def test_subv2ind():
    """
    FUNCTION: subv2ind, in general.py.
    """
    """Create data required to test function"""
    sz = np.array([ 2.,  2.])
    sub = np.array([[0, 1],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [1, 0],
                    [1, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 0],
                    [1, 1],
                    [1, 0],
                    [1, 0],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                    [0, 1],
                    [0, 0],
                    [0, 1],
                    [0, 1],
                    [0, 0],
                    [1, 0],
                    [1, 0],
                    [0, 0],
                    [1, 1]])

    """Execute function"""
    index = general.subv2ind(sz, sub)

    """Createn expected output for comparison"""
    t_index = np.array([[ 2.],
                        [ 2.],
                        [ 1.],
                        [ 1.],
                        [ 2.],
                        [ 1.],
                        [ 2.],
                        [ 2.],
                        [ 1.],
                        [ 2.],
                        [ 2.],
                        [ 1.],
                        [ 2.],
                        [ 1.],
                        [ 2.],
                        [ 3.],
                        [ 0.],
                        [ 2.],
                        [ 1.],
                        [ 3.],
                        [ 1.],
                        [ 3.],
                        [ 1.],
                        [ 1.],
                        [ 2.],
                        [ 1.],
                        [ 1.],
                        [ 1.],
                        [ 2.],
                        [ 2.],
                        [ 1.],
                        [ 1.],
                        [ 0.],
                        [ 3.],
                        [ 1.],
                        [ 1.],
                        [ 0.],
                        [ 1.],
                        [ 0.],
                        [ 1.],
                        [ 0.],
                        [ 2.],
                        [ 0.],
                        [ 2.],
                        [ 2.],
                        [ 0.],
                        [ 1.],
                        [ 1.],
                        [ 0.],
                        [ 3.]])

    """Assert the function executed correctly"""
    assert_array_equal(index, t_index)

def test_mk_multi_index():
    """
    FUNCTION: mk_multi_index, in general.py.
    """
    """Execute function"""
    index = general.mk_multi_index(3, [1], [1])

    """Assert the function executed correctly"""
    assert index == [slice(None, None, None), slice(1, 2, None), \
                     slice(None, None, None)]

def test_mk_undirected():
    """
    FUNCTION: mk_multi_index, in general.py.
    """
    model_graph = np.array([[0, 1, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]])
    
    """Execute function"""
    model_graph = general.mk_undirected(model_graph)

    """Create expected output for comparison"""
    t_model_graph = np.array([[0, 1, 1, 0],
                            [1, 0, 0, 1],
                            [1, 0, 0, 1],
                            [0, 1, 1, 0]])
    
    """Assert the function executed correctly"""
    assert_array_equal(model_graph, t_model_graph)


