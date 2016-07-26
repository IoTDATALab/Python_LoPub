#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module provides general functions that are used by the Python
based Bayesian network toolbox.
"""

__docformat__ = 'restructuredtext'

import numpy as np
import pylab

def determine_observed(evidence):
    """
    Determine which nodes are observed and which are hidden.

    Parameters
    ----------
    evidence: List
        A list of any observed evidence. If evidence[i] = [], then
        node i is unobserved (hidden node), else if evidence[i] =
        SomeValue then, node i has been observed as being SomeValue.
    """
    hidden = []
    observed = []
    ev = evidence[:]

    """Find the hidden nodes"""
    while ev.count([]) != 0:
        ind = ev.index([])
        ev[ind] = 0
        hidden.append(ind)

    """Find the observed nodes"""
    observed = np.setdiff1d(np.array(range(0, len(evidence))),\
                            np.array(hidden)).tolist()

    return [hidden, observed]

def determine_pot_type(model, onodes):
    """
    Determines the potential type of the model, based on which nodes are
    hidden.

    Parameters
    ----------
    Model: bnet object
        A pobabalistic graphical model.

    onodes: List
        A list of indices of the observed nodes in the model.
    """
    nodes = model.order
    hnodes = mysetdiff(np.array(nodes), np.array(onodes))
    if len(np.intersect1d(np.array(model.cnodes[:]), \
                          np.array(hnodes).tolist())) == 0:
        """If all hidden nodes are discrete nodes"""
        pot_type = 'd'
    elif issubset(np.array(hnodes), np.array(model.cnodes[:])):
        """If all the hidden nodes are continuous"""
        pot_type = 'g'
    else:
        pot_type = 'cg'

    return pot_type

def mysetdiff(A, B):
    """
    Returns the difference between 2 sets.

    Parameters
    ----------
    A: List
        A list defining a set.
    B: List
        A list defining a set.
    """
    if A.shape == (0,):
        return np.array(([]))
    elif B.shape == (0,):
        return A
    else:
        return np.setdiff1d(A,B)

def issubset(small, large):
    """
    Returns true if the set in 'small' is a subset of the set
    in 'large'.

    Parameters
    ----------
    small: List
        A list defining a set.
    large: List
        A list defining a set.
    """
    if small.shape == (0,):
        issubset = True
    else:
        temp = np.intersect1d(np.array(small[:]), np.array(large[:]))
        issubset = temp.shape[0] == small.shape[0]

    return issubset

def find(U):
    """
    Returns the indices of the elements that are True in U.

    Parameters
    ----------
    U: A flat 2D array filled with ones and zeros.
    """
    args = np.argwhere(U)
    U = args[:,1]
    return np.array([U])

def draw_graph(adj_mat, names=[], directed=False, text=''):
    """
    Draws the graph represented by adj_mat to a figure.

    Parameters
    ----------
    adj_mat: Numpy ndarray
        An adjacency matrix representing a graph, e.g. adj_mat[i, j] == 1,
        then node i and node j are connected.

    names: List
        A list of the names of the nodes, e.g names[i] == 'Rain', then node
        i is labeled 'Rain'.
    """
    import networkx as nx
    pylab.figure()
    pylab.title(text)
    if directed == False:
        if names != []:
            g = nx.Graph()
            g.add_nodes_from(names)
            edges = np.argwhere(adj_mat == 1)
            for i in edges:
                g.add_edge((names[i[0]], names[i[1]]))
        else:
            g = nx.Graph(adj_mat)
    else:
        if names != []:
            g = nx.DiGraph()
            g.add_nodes_from(names)
            edges = np.argwhere(adj_mat == 1)
            for i in edges:
                g.add_edge((names[i[0]], names[i[1]]))
        else:
            g = nx.DiGraph(adj_mat)
    nx.draw(g)

def mk_stochastic(mat):
    """
    Ensure that the sum over the last dimension is one. If mat is a 3D
    matrix, then this function will ensure that sum_k(i, j, k) = 1 for
    all i and j.

    Parameters
    ----------
    mat: numpy ndarray
        The matrix to convert.
    """
    if mat.squeeze().ndim == 1:
        mat = mat.squeeze()

    ns = mat.shape
    mat = mat.reshape(np.prod(ns[0:-1]), ns[len(ns) - 1], order='FORTRAN')
    s = np.sum(mat, 1)
    s = np.array([s + (s==0)])
    norm = np.repeat(s, ns[len(ns) - 1], 0)
    mat = mat / norm.T
    mat = mat.reshape(ns, order='FORTRAN')

    return mat

def compute_counts(data, sz):
    """
    Counts the number of times each combination of discrete assignments
    occurs. For instance, if sz = [2, 2], that means there are two binary
    nodes, which can be in 2**2 = 4 possible states: [0, 0], [1, 0], [0, 1]
    and [1, 1]. The output of this function would be a 2-by-2 matrix,
    containing that tally of each time a certain discrete combination occured
    in data. Therefore counts[0, 1] = the amount of times the combination
    of the first possible value for node 1 and the second possible value for
    node 2 occurred, in this binary case, the combination [0, 1].

    Parameters
    ----------
    data: numpy ndarray
        data(i,t) is the value of variable i in case t.

    sz:
        The values for variable i are assumed to be in range(0, sz(i))
    """
    P = np.prod(sz)
    indices = subv2ind(sz, data)
    count = np.histogram(indices, P, (0, P-1))
    count = count[0]
    count = count.reshape(sz)
    count = count.T
    return count

def subv2ind(sz, sub):
    """
    Linear index from subscript vector.

    Parameters
    ----------
    sz: List
        The size of the array we want to create a linear index for.

    sub: numpy ndarray
        The subscript vector
    """
    cum_size = np.cumprod(sz[0:-1])
    prev_cum_size = np.mat((np.hstack((1, cum_size))))
    index = (sub + 1)*prev_cum_size.T - np.sum(prev_cum_size)
    return index

def mk_multi_index(n, dims, vals):
    """
    Creates a list of slices, named index. The list can be used to slice an
    array, for example:
        index = mk_multi_index(3, [0, 2], [3, 2])
        gives index = [slice(3,4), slice(None), slice(2, 3)],
        which will select out dim 0 the 3rd entry, out of dim 1 everything,
        and out of dim 2 the 2nd entry.

        So if A[:,:,1]=[[1 2], [3 4], [5 6]]
              A[:,:,2]=[[7 8], [9 10], [11 12]]

        then A(index{:}) = [11 12].

    Parameters
    ----------
    n: Int
        The number of dimensions the matrix to be sliced has.

    dims: List
        The dimensions we wish to slice from.

    vals: List
        Which entries to select out of the desired dimensions.

    """
    index = []
    for i in range(0, n):
        if i in dims:
            val = vals[dims.index(i)]
            index.append(slice(val, val + 1))
        else:
            index.append(slice(None))

    return index


def mk_undirected(model_graph):
    """
    Converts an adjacency matrix representing edges in a directed graph,
    by making all the edges undirected.

    Parameters
    ----------
    model_graph: Numpy array
        The adjacency matrix representing the directed graph.
    """
    for i in range(0, model_graph.shape[0]):
        for j in range(0, model_graph.shape[1]):
            if model_graph[i, j] == 1:
                model_graph[j, i] = 1
    return model_graph

def block(blocks, block_sizes):
    """
    Return a vector of subscripts corresponding to specified blocks.
    """
    skip = np.cumsum(block_sizes).tolist()
    skip.insert(0, 0)
    start = np.array(skip)[blocks]
    fin = start + block_sizes[blocks]

    if type(blocks) == int:
        len_blocks = 1
        start = [start]
        fin = [fin]
    else:
        len_blocks = len(blocks)

    sub = []
    for i in range(0, len_blocks):
        sub.append(range(start[i], fin[i]))

    return sub

def gaussian_prob(x, m, C, use_log=False):
    """
    Evaluate a multivariate Gaussian density.
    """
    [N, d] = x.shape
    M = m.flatten().T * np.matrix(np.ones((1, N)))
    denom = (2 * np.pi)**(float(d)/2) * np.sqrt(np.abs(np.linalg.det(C)))
    mahal = np.array((x.T - M).T * np.linalg.inv(C))
    mahal = np.sum(mahal * np.array((x.T - M)).T, 1)
    eps = 2.2204 * np.exp(-16)
    if use_log:
        p = -0.5 * mahal - np.log(denom)
    else:
        p = (np.exp(-0.5 * mahal)) / (denom + eps)
    return p


