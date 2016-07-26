# Author: Almero Gouws <14366037@sun.ac.za>
"""
This is a tutorial on how to create a Bayesian network, and perform
approximate MAX-SUM inference on it.
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

    """
    We now need to assign a conditional probability distribution to each
    node.
    """
    node_cpds = [[], [], [], []]

    """Define the CPD for node 0"""
    CPT = np.array([0.5, 0.5])
    node_cpds[C] = cpds.tabular_CPD(C, ns, dag, CPT)

    """Define the CPD for node 1"""
    CPT = np.array([[0.8, 0.2], [0.2, 0.8]])
    node_cpds[R] = cpds.tabular_CPD(R, ns, dag, CPT)

    """Define the CPD for node 2"""
    CPT = np.array([[0.5, 0.5], [0.9, 0.1]])
    node_cpds[S] = cpds.tabular_CPD(S, ns, dag, CPT)

    """Define the CPD for node 3"""
    CPT = np.array([[[1, 0], [0.1, 0.9]], [[0.1, 0.9], [0.01, 0.99]]])
    node_cpds[W] = cpds.tabular_CPD(W, ns, dag, CPT)

    """Create the Bayesian network"""
    net = models.bnet(dag, ns, node_cpds)

    """
    Intialize the BNET's inference engine to use approximate inference, by
    setting exact=False.
    """
    net.init_inference_engine(exact=False)

    """Create and enter evidence ([] means that node is unobserved)"""
    evidence = [[], 0, [], []]

    """Execute the max-sum algorithm"""
    mlc = net.max_sum(evidence)

    """
    mlc contains the most likely configuaration for all the nodes in the BNET
    based in the input evidence.
    """
    print 'Cloudy node:     ', bool(mlc[C])
    print 'Sprinkler node:  ', bool(mlc[S])
    print 'Rainy node:      ', bool(mlc[R])
    print 'Wet grass node:  ', bool(mlc[W])
