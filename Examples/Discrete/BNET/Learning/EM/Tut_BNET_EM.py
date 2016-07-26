# Author: Almero Gouws <14366037@sun.ac.za>
"""
This is a tutorial on how to create a Bayesian network, learn its parameters
from partially observed data via EM, and perform exact MAX-SUM inference on it.
"""
"""Import the required numerical modules"""
import numpy as np

"""Import the GrMPy modules"""
import models
import inference
import cpds

if __name__ == '__main__':
    """
    This example is based on the lawn sprinkler example, and the Bayesian
    network has the following structure, with all edges directed downwards:

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
    If dag[i, j] = 1, then there exists a directed edge from node
    i and node j.
    """
    dag = np.zeros((nodes, nodes))
    dag[C, [R, S]] = 1
    dag[R, W] = 1
    dag[S, W] = 1

    """
    Define the size of each node, which is the number of different values a
    node could observed at. For example, if a node is either True of False,
    it has only 2 possible values it could be, therefore its size is 2. All
    the nodes in this graph has a size 2.
    """
    ns = 2 * np.ones((1, nodes))

    """Create the BNET"""
    net = models.bnet(dag, ns)

    """Define the samples to train the models parameters with"""
    samples = \
           [[0, 1, 0, 0],
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
    
    """Intialize the BNET's inference engine to use EXACT inference"""
    net.init_inference_engine(exact=True)

    """Create and enter evidence ([] means that node is unobserved)"""
    evidence = [[], 0, [], []]
    mlc = net.max_sum(evidence)

    """
    mlc contains the most likely configuaration for all the nodes in the BNET
    based in the input evidence.
    """
    print 'Cloudy node:     ', bool(mlc[C])
    print 'Sprinkler node:  ', bool(mlc[S])
    print 'Rainy node:      ', bool(mlc[R])
    print 'Wet grass node:  ', bool(mlc[W])
