# Author: Almero Gouws <14366037@sun.ac.za>
"""
This is a tutorial on how to create a Markov random field, and perform
exact SUM-PRODUCT inference on it.
"""
"""Import the required numerical modules"""
import numpy as np

"""Import the GrMPy modules"""
import models
import inference
import cliques

if __name__ == '__main__':
    """
    This example is based on the lawn sprinkler example, and the Markov
    random field has the following structure.

                            Cloudy - 0
                             /  \
                            /    \
                           /      \
                   Sprinkler - 1  Rainy - 2
                           \      /
                            \    /
                             \  /
                           Wet Grass -3                
    """
    """Assign a unique numerical identifier to each node"""
    C = 0
    S = 1
    R = 2
    W = 3

    """Assign the number of nodes in the graph"""
    nodes = 4

    """
    The graph structure is represented as a adjacency matrix, dag.
    If adj_mat[i, j] = 1, then there exists a undirected edge from node
    i and node j.
    """
    adj_mat = np.matrix(np.zeros((nodes, nodes)))
    adj_mat[C, [R, S]] = 1
    adj_mat[R, W] = 1
    adj_mat[S, W] = 1

    """
    Define the size of each node, which is the number of different values a
    node could observed at. For example, if a node is either True of False,
    it has only 2 possible values it could be, therefore its size is 2. All
    the nodes in this graph has a size 2.
    """
    ns = 2 * np.ones((1, nodes))

    """
    Define the clique domains. The domain of a clique, is the indices of the
    nodes in the clique. A clique is a fully connected set of nodes.
    Therefore, for a set of node to be a clique, every node in the set must
    be connected to every other node in the set.
    """
    clq_doms = [[0, 1, 2], [1, 2, 3]]

    """Define potentials for the cliques"""
    clqs = []
    T = np.zeros((2, 2, 2))
    T[:, :, 0] = np.array([[0.2, 0.2], [0.09, 0.01]])
    T[:, :, 1] = np.array([[0.05, 0.05], [0.36, 0.04]])
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2, 2, 2]), T))
    T[:, :, 0] = np.array([[1, 0.1], [0.1, 0.01]])
    T[:, :, 1] = np.array([[0, 0.9], [0.9, 0.99]])
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2, 2]), T))

    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs, lattice=False)

    """
    Intialize the MRF's inference engine to use EXACT inference, by
    setting exact=True.
    """
    net.init_inference_engine(exact=True)

    """Create and enter evidence ([] means that node is unobserved)"""
    evidence = [[], 0, [], []]

    """Execute max-sum algorithm"""
    net.sum_product(evidence)

    """
    Print out the marginal probability of each node.
    """
    marginal = net.marginal_nodes([C])
    print 'Probability it is cloudy:     ', marginal.T[1]*100, '%'
    marginal = net.marginal_nodes([S])
    print 'Probability the sprinkler is on:  ', 0, '%'   #Observed node
    marginal = net.marginal_nodes([R])
    print 'Probability it is raining:      ',marginal.T[1]*100, '%'
    marginal = net.marginal_nodes([W])
    print 'Probability the grass is wet:  ', marginal.T[1]*100, '%'
