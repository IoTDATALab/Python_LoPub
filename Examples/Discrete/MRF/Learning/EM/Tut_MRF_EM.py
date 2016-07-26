# Author: Almero Gouws <14366037@sun.ac.za>
"""
This is a tutorial on how to create a Markov random field, learn its
parameters from partially observed data via EM, and perform exact MAX-SUM
inference on it.
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
                1 - Sprinkler----Rainy - 2
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
    clq_doms = [[0], [0, 1], [0, 2], [1, 2, 3]]

    """Create blank cliques with the required domains and sizes"""
    clqs = []
    clqs.append(cliques.discrete_clique(0, clq_doms[0], np.array([2])))
    clqs.append(cliques.discrete_clique(1, clq_doms[1], np.array([2, 2])))
    clqs.append(cliques.discrete_clique(2, clq_doms[2], np.array([2, 2])))
    clqs.append(cliques.discrete_clique(3, clq_doms[3], np.array([2, 2, 2])))


    """Create the MRF"""
    net = models.mrf(adj_mat, ns, clqs, lattice=False)

    """Define the samples that will be used to train the models parameters"""
    samples = \
           [[0, 1, 0, []],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]]

    """Train the models parameters using the defined samples"""
    net.learn_params_EM(samples[:])
    
    """Intialize the MRF's inference engine to use EXACT inference"""
    net.init_inference_engine(exact=True)

    """Create and enter evidence ([] means that node is unobserved)"""
    evidence = [[], 0, [], []]
    mlc = net.max_sum(evidence)

    """
    mlc contains the most likely configuaration for all the nodes in the MRF
    based in the input evidence.
    """
    print 'Cloudy node:     ', bool(mlc[C])
    print 'Sprinkler node:  ', bool(mlc[S])
    print 'Rainy node:      ', bool(mlc[R])
    print 'Wet grass node:  ', bool(mlc[W])
