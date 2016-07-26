# Author: Almero Gouws <14366037@sun.ac.za>
"""
This module contains the classes used to perform inference on various
graphical models.
"""
__docformat__ = 'restructuredtext'

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.sparse import sparse
import general
import graph
import cliques
import potentials
import models
import cpds
import pylab
from utilities import create_all_evidence
    
def test_mrf_exact_sum_product():
    """EXAMPLE: Junction tree sum-product on MRF"""
    """Define MRF graph structure"""
    C = 0
    S = 1
    R = 2
    W = 3
    nodes = 4
    adj_mat = sparse.lil_matrix((nodes, nodes), dtype=int)
    adj_mat[C, [R, S]] = 1
    adj_mat[R, W] = 1
    adj_mat[S, W] = 1
    adj_mat[R, S] = 1

    """Define clique domains and node sizes"""
    ns = 2 * np.ones((1, nodes))
    clq_doms = [[0, 1, 2], [1, 2, 3]]

    """Define cliques and potentials"""
    clqs = []
    T = np.zeros((2, 2, 2))
    T[:, :, 0] = np.array([[0.2, 0.2], [0.09, 0.01]])
    T[:, :, 1] = np.array([[0.05, 0.05], [0.36, 0.04]])
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2, 2, 2]), T))
    T[:, :, 0] = np.array([[1, 0.1], [0.1, 0.01]])
    T[:, :, 1] = np.array([[0, 0.9], [0.9, 0.99]])
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2, 2]), T))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs)
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    results = []
    for evidence in evidences:
        net.sum_product(evidence)
        result = []
        result.append(np.max(net.marginal_nodes([0]).T))
        result.append(np.max(net.marginal_nodes([1]).T))
        result.append(np.max(net.marginal_nodes([2]).T))
        result.append(np.max(net.marginal_nodes([3]).T))
        results.append(result)

    results = np.array(results)

    """Get the expected results"""
    exp_results = np.array(pylab.load('./Data/mrf_exact_sum_product_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(results, exp_results)

def test_mrf_exact_max_sum():
    """EXAMPLE: Junction tree max-sum on MRF"""
    """Define MRF graph structure"""
    C = 0
    S = 1
    R = 2
    W = 3
    nodes = 4
    adj_mat = sparse.lil_matrix((nodes, nodes), dtype=int)
    adj_mat[C, [R, S]] = 1
    adj_mat[R, W] = 1
    adj_mat[S, W] = 1
    adj_mat[R, S] = 1

    """Define clique domains and node sizes"""
    ns = 2 * np.ones((1, nodes))
    clq_doms = [[0, 1, 2], [1, 2, 3]]

    """Define cliques and potentials"""
    clqs = []
    T = np.zeros((2, 2, 2))
    T[:, :, 0] = np.array([[0.2, 0.2], [0.09, 0.01]])
    T[:, :, 1] = np.array([[0.05, 0.05], [0.36, 0.04]])
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2, 2, 2]), T))
    T[:, :, 0] = np.array([[1, 0.1], [0.1, 0.01]])
    T[:, :, 1] = np.array([[0, 0.9], [0.9, 0.99]])
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2, 2]), T))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs)
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([0, 0, 0, 0])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))

    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/mrf_exact_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)

def test_mrf_approx_sum_product():
    """EXAMPLE: Loopy belief sum-product on MRF"""
    """Define MRF graph structure"""
    C = 0
    S = 1
    R = 2
    W = 3
    nodes = 4
    adj_mat = sparse.lil_matrix((nodes, nodes), dtype=int)
    adj_mat[C, [R, S]] = 1
    adj_mat[R, W] = 1
    adj_mat[S, W] = 1
    adj_mat[R, S] = 1

    """Define clique domains and node sizes"""
    ns = 2 * np.ones((1, nodes))
    clq_doms = [[0, 1, 2], [1, 2, 3]]

    """Define cliques and potentials"""
    clqs = []
    T = np.zeros((2, 2, 2))
    T[:, :, 0] = np.array([[0.2, 0.2], [0.09, 0.01]])
    T[:, :, 1] = np.array([[0.05, 0.05], [0.36, 0.04]])
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2, 2, 2]), T))
    T[:, :, 0] = np.array([[1, 0.1], [0.1, 0.01]])
    T[:, :, 1] = np.array([[0, 0.9], [0.9, 0.99]])
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2, 2]), T))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs)
    net.init_inference_engine(exact=False)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    results = []
    for evidence in evidences:
        net.sum_product(evidence)
        result = []
        result.append(np.max(net.marginal_nodes([0]).T))
        result.append(np.max(net.marginal_nodes([1]).T))
        result.append(np.max(net.marginal_nodes([2]).T))
        result.append(np.max(net.marginal_nodes([3]).T))
        results.append(result)

    results = np.array(results)

    """Get the expected results"""
    exp_results = np.array(pylab.load('./Data/mrf_approx_sum_product_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(results, exp_results)

def test_mrf_approx_max_sum():
    """EXAMPLE: Loopy belief max-sum on MRF"""
    """Define MRF graph structure"""
    C = 0
    S = 1
    R = 2
    W = 3
    nodes = 4
    adj_mat = sparse.lil_matrix((nodes, nodes), dtype=int)
    adj_mat[C, [R, S]] = 1
    adj_mat[R, W] = 1
    adj_mat[S, W] = 1
    adj_mat[R, S] = 1

    """Define clique domains and node sizes"""
    ns = 2 * np.ones((1, nodes))
    clq_doms = [[0, 1, 2], [1, 2, 3]]

    """Define cliques and potentials"""
    clqs = []
    T = np.zeros((2, 2, 2))
    T[:, :, 0] = np.array([[0.2, 0.2], [0.09, 0.01]])
    T[:, :, 1] = np.array([[0.05, 0.05], [0.36, 0.04]])
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2, 2, 2]), T))
    T[:, :, 0] = np.array([[1, 0.1], [0.1, 0.01]])
    T[:, :, 1] = np.array([[0, 0.9], [0.9, 0.99]])
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2, 2]), T))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs)
    net.init_inference_engine(exact=False)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([0, 0, 0, 0])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))

    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/mrf_approx_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)


def test_bnet_exact_sum_product():
    """EXAMPLE: Junction tree sum-product on BNET"""
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
    net = models.bnet(dag, ns, node_cpds)
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    results = []
    for evidence in evidences:
        net.sum_product(evidence)
        result = []
        result.append(np.max(net.marginal_nodes([0]).T))
        result.append(np.max(net.marginal_nodes([1]).T))
        result.append(np.max(net.marginal_nodes([2]).T))
        result.append(np.max(net.marginal_nodes([3]).T))
        results.append(result)

    results = np.array(results)

    """Get the expected results"""
    exp_results = np.array(pylab.load('./Data/bnet_exact_sum_product_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(results, exp_results)
    
def test_bnet_exact_max_sum():
    """EXAMPLE: Junction tree max-sum on BNET"""
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
    net = models.bnet(dag, ns, node_cpds)
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([0, 0, 0, 0])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))

    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/bnet_exact_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)
    

def test_bnet_approx_sum_product():
    """EXAMPLE: Loopy belief sum-product on BNET"""
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
    net = models.bnet(dag, ns, node_cpds)
    net.init_inference_engine(exact=False)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    results = []
    for evidence in evidences:
        net.sum_product(evidence)
        result = []
        result.append(np.max(net.marginal_nodes([0]).T))
        result.append(np.max(net.marginal_nodes([1]).T))
        result.append(np.max(net.marginal_nodes([2]).T))
        result.append(np.max(net.marginal_nodes([3]).T))
        results.append(result)

    results = np.array(results)

    """Get the expected results"""
    exp_results = np.array(pylab.load('./Data/bnet_approx_sum_product_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(results, exp_results)
    

def test_bnet_approx_max_sum():
    """EXAMPLE: Loopy belief max-sum on BNET"""
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
    net = models.bnet(dag, ns, node_cpds)
    net.init_inference_engine(exact=False)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([0, 0, 0, 0])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))

    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/bnet_approx_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)
