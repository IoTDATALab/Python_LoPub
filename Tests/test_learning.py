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
       
def test_bnet_mle():
    """EXAMPLE: MLE learning on a BNET"""
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

    """Instantiate the model"""
    net = models.bnet(dag, ns, [])

    """Learn the parameters"""
    samples = np.array(pylab.load('./Data/lawn_samples.txt')) - 1
    net.learn_params_mle(samples.copy())
   
    """Initialize the inference engine"""
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([[0, 0, 0, 0]])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))

    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/bnet_mle_exact_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)
    
def test_bnet_EM():
    """EXAMPLE: EM learning on a BNET"""
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

    """Instantiate the model"""
    net = models.bnet(dag, ns, [])


    """
    Load the samples, and set one sample of one node to be unobserved, this
    should not effect the learnt parameter much, and will demonstrate that
    the algorithm can handle unobserved samples.
    """
    samples = (np.array(pylab.load('./Data/lawn_samples.txt')) - 1).tolist()
    samples[0][0] = []

    """Learn the parameters"""
    net.learn_params_EM(samples[:])
   
    """Initialize the inference engine"""
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([[0, 0, 0, 0]])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))

    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/bnet_mle_exact_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)
    
def test_mrf_mle():
    """EXAMPLE: MLE learning on a MRF"""
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
    clq_doms = [[0], [0, 1], [0, 2], [1, 2, 3]]

    """Define cliques and potentials"""
    clqs = []
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2])))
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2])))
    clqs.append(cliques.discrete_clique(2, clq_doms[2], np.array([2, 2])))
    clqs.append(cliques.discrete_clique(3, clq_doms[3], np.array([2, 2, 2])))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs)
    
    """Learn the parameters"""
    samples = np.array(pylab.load('./Data/lawn_samples.txt')) - 1
    net.learn_params_mle(samples[:])
   
    """Initialize the inference engine"""
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([[0,  0, 0, 0]])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))
   
    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/mrf_mle_exact_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)

def test_mrf_EM():
    """EXAMPLE: EM learning on a MRF"""
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
    clq_doms = [[0], [0, 1], [0, 2], [1, 2, 3]]

    """Define cliques and potentials"""
    clqs = []
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2])))
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2])))
    clqs.append(cliques.discrete_clique(2, clq_doms[2], np.array([2, 2])))
    clqs.append(cliques.discrete_clique(3, clq_doms[3], np.array([2, 2, 2])))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs)
    
    """
    Load the samples, and set one sample of one node to be unobserved, this
    should not effect the learnt parameter much, and will demonstrate that
    the algorithm can handle unobserved samples.
    """
    samples = (np.array(pylab.load('./Data/lawn_samples.txt')) - 1).tolist()
    samples[0][0] = []

    """Learn the parameters"""
    net.learn_params_EM(samples[:])
   
    """Initialize the inference engine"""
    net.init_inference_engine(exact=True)

    """Create and enter evidence"""
    evidences = create_all_evidence(4, 2)
    mlcs = np.array([[0, 0, 0, 0]])
    for evidence in evidences:
        mlc = net.max_sum(evidence)
        mlcs = np.vstack((mlcs, mlc))
   
    """Read in expected values"""
    exp_mlcs = np.array(pylab.load('./Data/mrf_em_exact_max_sum_res.txt'))

    """Assert that the output matched the expected values"""
    assert_array_equal(mlcs, exp_mlcs)
