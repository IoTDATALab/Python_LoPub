#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module supplies functions used to implement graph theory.
"""
__docformat__ = 'restructuredtext'

import numpy as np
from dfs import dfs
from general import find, issubset

def parents(adj_mat, i):
    """
    Returns the indices of the parent nodes of the input node, i, in the
    given adjacency matrix.
    Parameters
    ----------
    adj_mat: Numpy ndarray
        Adjacency matrix. If adj_mat[i, j] = 1, there exists a directed
        edge from node i to node j.
    i: Int
        The index of the node whose parents are to be found.
    """
    """Check if this is perhaps a sparse matrix"""
    if type(adj_mat) != np.ndarray:
        posi = np.array((adj_mat[:, i].todense() == 1)).squeeze()
    else:
        posi = np.array((adj_mat[:, i] == 1))
    
    vals = []
    while np.sum(posi)!=0:
        t_pos = np.argmax(posi)
        posi[t_pos]=0
        vals.append(t_pos)

    return vals

def children(adj_mat, i):
    """
    Returns the indices of the children nodes of the input node, i, in the
    given adjacency matrix.
    Parameters
    ----------
    adj_mat: Numpy ndarray
        Adjacency matrix. If adj_mat[i, j] = 1, there exists a directed
        edge from node i to node j.
    i: Int
        The index of the node whose parents are to be found.
    """
    """Check if this is perhaps a sparse matrix"""
    i=int(i)
    if type(adj_mat) != np.ndarray:
        adj_mat = adj_mat.tocsr()
        posi = np.array((adj_mat[i, :].todense() == 1)).squeeze()
        adj_mat = adj_mat.tocsc()
    else:
        posi = np.array((adj_mat[i, :] == 1))
        
    vals = []
    while np.sum(posi)!=0:
        t_pos = np.argmax(posi)
        posi[t_pos]=0
        vals.append(t_pos)

    
    return vals

def neighbours(adj_mat, i):
    """
    Returns the indices of the neighbours nodes of the input node, i, in the
    given adjacency matrix.
    Parameters
    ----------
    adj_mat: Numpy ndarray
        Adjacency matrix. If adj_mat[i, j] = 1, there exists a directed
        edge from node i to node j.
    i: Int
        The index of the node whose parents are to be found.
    """
    i=int(i)
    kids = np.array(children(adj_mat, i))
    folks = np.array(parents(adj_mat,i))
    if issubset(kids, folks) and issubset(folks, kids):
        nbrs = kids
    else:
        nbrs = np.hstack((kids, folks)).tolist()

    return nbrs

def family(adj_mat, i):
    """
    Returns the indices of the family nodes of the input node, i, in the
    given adjacency matrix.
    Parameters
    ----------
    adj_mat: Numpy ndarray
        Adjacency matrix. If adj_mat[i, j] = 1, there exists a directed
        edge from node i to node j.
    i: Int
        The index of the node whose parents are to be found.
    """
    f = parents(adj_mat, i)
    f.append(i)

    return f

def topological_sort(A):
    """
    Returns the indices of the nodes in the graph defined by the adjacency
    matrix A in topological order.
    Parameters
    ----------
    A: Scipy sparse csc_matrix
        Adjacency matrix. If A[i, j] = 1, there exists a directed edge from
        node i to node j.
    """
   
    n = A.shape[0]
    indeg = []
    zero_indeg = []
    for i in range(0,n):
        indeg.append(len(parents(A,i)))
        if indeg[i] == 0:
            zero_indeg.append(i)

    zero_indeg.reverse()

    t = 1
    order = []
    while len(zero_indeg)!=0:
        v = zero_indeg.pop()
        order.append(v)
        t = t + 1
        cs = children(A, v)
        
        for j in range(0,len(cs)):
            c = cs[j]
            indeg[c] = indeg[c] - 1
            if indeg[c] == 0:
                zero_indeg.insert(0,c)

    return order

def moralize(G):
    """
    Converts a directed graph to an undirected graph, by connecting the
    parents of every node together.
    Parameters
    ----------
    G: Numpy ndarray
        Adjacency matrix. If A[i, j] = 1, there exists a directed edge from
        node i to node j.
    """
    M = G.copy()
    n = M.shape[0]
    for i in range(0,n):
        fam = family(G,i)
        for j in fam:
            M[j, fam] = 1

    """Make sure no node has an edge to itself"""
    M = setdiag(M, 0)
    moral_edges = np.triu(M-G,0)

    return [M, moral_edges]

def setdiag(G, val):
    """
    Sets the diagonal elements of a matrix to a specified value.
    Parameters
    ----------
    G: A 2D matrix or array.
        The matrix to modify.
    val: Int or float
        The value to which the diagonal of 'G' will be set.
    """
    n = G.shape[0]
    for i in range(0,n):
        G[i,i] = val

    return G

def graph_to_jtree(model_graph, ns):
    """
    This function triangulates a moral graph and obtains a junction tree
    from the cliques of the triangulated graph by computing the maximum
    spanning tree for those cliques.
    Parameters
    ----------
    model_graph: Numpy ndarray
        MG[i,j] = 1 iff there is an edge between node i and node j.
    ns: List
        The node sizes, where ns[i] = the number of discrete values node i
        can take on [1 if observed].
    Output
    ------
    jtree: Numpy ndarray
        A matrix reprsenting the edges in the junction tree. jtree(i,j)=1
        iff there is an edge between clique i and clique j.
    root: Int
        The index of the root clique.
    cliques: List
        A list of lists of the indices of each clique. cliques[i] = the
        indices of the nodes in clique i.
    B: Numpy ndarray
        A map of which clique each node appears in, B[i,j] = 1 iff node j
        occurs in clique i.
    w: List
        The weights of the cliques, w[i] = weight of clique i.
    """
    """Make sure that no node has a edge connecting to itself."""
    model_graph = setdiag(model_graph, 0)

    """Determine the elimination order"""
    elim_order = best_first_elim_order(model_graph.copy(), ns)

    """
    Using the elimination order and the moral graph, obtain the new cliques
    using triangulation.
    """
    [triangulated_graph, cliques] = triangulate(model_graph.copy(), elim_order)
    #print(triangulated_graph)

    """
    Obtain a junction tree from the set of cliques.
    """
    [jtree, root, B, w] = cliques_to_jtree(cliques, ns)

    return [triangulated_graph,jtree, root, cliques, B, w]

def best_first_elim_order(G, node_sizes, stage=[]):
    """
    This function greedily searches for an optimal elimination order.
    Find an order in which to eliminate nodes from the graph in such a way
    as to try and minimize the weight of the resulting triangulated graph.
    The weight of a graph is the sum of the weights of each of its cliques;
    the weight of a clique is the product of the weights of each of its
    members; the weight of a node is the number of values it can take on.
    Since this is an NP-hard problem, we use the following greedy heuristic:
    At each step, eliminate that node which will result in the addition of
    the least number of fill-in edges, breaking ties by choosing the node
    that induces the lighest clique.
    For details, see
    - Kjaerulff, "Triangulation of graphs -- algorithms giving small total
        state space", Univ. Aalborg tech report, 1990 (www.cs.auc.dk/~uk)
    - C. Huang and A. Darwiche, "Inference in Belief Networks: A procedural
        guide", Intl. J. Approx. Reasoning, 11, 1994
    Parameters
    ----------
    G: Numpy ndarray
        G[i,j] = 1 iff there is an edge between node i and node j.
    node_sizes: List
        The node sizes, where ns[i] = the number of discrete values
        node i can take on [1 if observed].
    stage: List
        stage[i] is a list of the nodes that must be eliminated at i'th
        stage.
    """
    """Obtain the number of nodes in the graph"""
    n = G.shape[0]
    if stage == []:
        stage = [range(0, n)]
    MG = G.copy()
    uneliminated = np.ones((1, n))
    order = np.zeros((1, n))
    t = 0

    """For each node in the graph"""
    for i in range(0, n):
        """Find the indices of the unelminated elements"""
        U = find(uneliminated == 1)

        """Find which nodes can be removed in this stage."""
        #valid = np.intersect1d_nu(np.array(U), np.array([stage[t]]))###################################################################################################################################
        valid = np.intersect1d(np.array(U), np.array([stage[t]]))

        """
        Determine which of the valid nodes will add the least number of fill in
        edges once eliminated. If 2 or more nodes add the same least number of
        fill in edges, then choose the one that results in the lightest clique.
        """
        min_fill = np.zeros((1, len(valid)))
        min_weight = np.zeros((1, len(valid)))
        """For each node that is valid for elimination"""
        for j in range(0, len(valid)):
            k = valid[j]
            
            """Obtain the uneliminated neighbours of the node to be eliminated"""
            nbrs = neighbours(G, k)
            #nbrs = np.intersect1d_nu(np.array([nbrs]), np.array(U))####################################################################################################################################
            nbrs = np.intersect1d(np.array([nbrs]), np.array(U))
            l = len(nbrs)
            M = np.zeros((l, l))
            count = 0
            for x in nbrs:
                for y in range(0, len(nbrs)):
                    M[count, y] = MG[x, nbrs[y]]
                count = count + 1

            """Save the number of fill-in edges required to eliminate node j"""
            min_fill[0, j] = l**2 - np.sum(M)
            nbrs = nbrs.tolist()
            nbrs.insert(0, k)
            """Save the clique weight obtained by eliminating node j"""
            min_weight[0, j] = np.prod(node_sizes[0, nbrs])

        """Determine which of the nodes create the lightest clique."""
        lightest_nbrs = find(min_weight == np.min(min_weight))
        
        """
        Determine which of nodes found in the step above, require the least
        number of fill-in edges to eliminate.
        """
        best_nbr_ndx = np.argmin(min_fill[0, lightest_nbrs.tolist()])
        j = lightest_nbrs[0, best_nbr_ndx]
        
        """
        Determine which of the nodes found in the step above are valid for
        elimination, these are the nodes to be eliminated.
        """
        k = valid[j]
        uneliminated[0, k] = 0
        
        """Add the nodes to be eliminated to the elimination order"""
        order[0, i] = k
        
        """Determine the nieghbours of the nodes to be eliminated"""
        ns = neighbours(G, k)
        #ns = np.intersect1d_nu(np.array([ns]), np.array(U))############################################################################################################################################
        ns = np.intersect1d(np.array([ns]), np.array(U))
        
        """Eliminate the nodes"""
        if len(ns) != 0:
            for x in ns:
                for y in ns:
                    G[x, y] = 1
            G = setdiag(G, 0)

        """
        If all the nodes valid for elimination in this stage have been
        eliminated, then advance to the next stage.
        """
        if np.sum(np.abs(uneliminated[0, stage[t]])) == 0:
            t = t + 1
    return order

def triangulate(G, order):
    """
    This function ensures that the input graph is triangulated (chordal),
    i.e., every cycle of length > 3 has a chord. To find the maximal
    cliques, we save each induced cluster (created by adding connecting
    neighbors) that is not a subset of any previously saved cluster. (A
    cluster is a complete, but not necessarily maximal, set of nodes.)
    Parameters
    ----------
    G: Numpy ndarray
        G[i,j] = 1 iff there is an edge between node i and node j.
    order: List
        The order in which to eliminate the nodes.
    """
    MG = G.copy()
    
    """Obtain the the number of nodes in the graph"""
    n = G.shape[0]
    eliminated = np.zeros((1,n))
    cliques = []
    for i in range(0,n):
        """Obtain the index of the next node to be eliminated"""
        u = order[0,i]
        u = int(u)
        U = find(eliminated == 0)
        #nodes = np.intersect1d_nu(neighbours(G, u), U)#################################################################################################################################################
        nodes = np.intersect1d(neighbours(G, u), U)
        nodes = np.union1d(nodes, np.array([u]))
        """
        Connect all uneliminated neighbours of the node to be eliminated
        together.
        """
        for i in nodes:
            i=int(i)
            for j in nodes:
                j=int(j)
                G[i, j] = 1
        G = setdiag(G, 0)

        """Mark the node as 'eliminated'"""
        eliminated[0, u] = 1

        """
        If the generated clique is a subset of an existing clique, then it is
        not a maximal clique, so it is excluded from the list if cliques.
        """
        exclude = False
        for c in range(0, len(cliques)):
            if issubset(nodes, np.array(cliques[c])):
                exclude = True
                break

        if not exclude:
            cliques.append(nodes)

    return [G, cliques]

def cliques_to_jtree(cliques, ns):
    """
    This function produces an optimal junction tree from a set of cliques.
    A junction tree is a tree that satisfies the jtree property, which says:
    for each pair of cliques U, V with intersection S, all cliques on the
    path between U and V contain S. (This ensures that local propagation
    leads to global consistency.)
    The best jtree is the maximal spanning tree which minimizes the sum of
    the costs on each edge. The cost on an edge connecting cliques i and j,
    is the weight of the seperator set between the two cliques, defined as
    the intersection between cliques i and j.
    Therefore, to determine the cost of an edge connecting 2 cliques:
    C[i] = clique i, and
    C[j] = clique j,
    S[i, j] = Intersection(C[i], C[j]), is the seperator set between i
    and j,
    w[S[i, j]]= weight of the seperator set, which is the product of the
    weights of each node in S, where the weight of a node is the number of
    values that node can take on. Therefore the cost of an edge connecting
    clique i and clique j is: cost[i, j] = W[S[i, j]].
    For details, see
    - Jensen and Jensen, "Optimal Junction Trees", UAI 94.
    Parameters
    ----------
    cliques: List
        cliques[i] contains the indices of the nodes in clique i.
    ns: List
        The node sizes, ns[i] is the number of values node i can take on.
    Ouput
    -----
    jtree: Numpy ndarray
        A matrix reprsenting the edges in the junction tree. jtree(i,j)=1
        iff there is an edge between clique i and clique j.
    root: Int
        The index of the root clique.
    cliques: List
        A list of lists of the indices of the nodes in each clique. cliques[i] =
        the indices of the nodes in clique i.
    B: Numpy ndarray
        A map of which clique each node appears in, B[i,j] = 1 iff node j
        occurs in clique i.
    w: List
        The weights of the cliques, w[i] = weight of clique i.
    """
    num_cliques = len(cliques)
    w = np.zeros((num_cliques, 1))
    B = np.zeros((num_cliques, ns.shape[1]))

    for i in range(0, num_cliques):
        i=int(i)
        cliquetemp=map(int,cliques[i])
        B[i, cliquetemp] = 1
        w[i] = np.prod(ns[0, cliquetemp])
        #B[i, cliques[i].tolist()] = 1
        #w[i] = np.prod(ns[0, cliques[i].tolist()])

    C1 = np.mat(B) * np.mat(B).T
    C1 = setdiag(C1, 0)

    W = np.repeat(w, num_cliques, 1)
    C2 = W + np.mat(W).T
    C2 = setdiag(C2, 0)

    """Using -C1 gives us the maximum spanning tree"""
    jtree = minimum_spanning_tree(-1*C1, C2)

    return [jtree, num_cliques, B, w]

def minimum_spanning_tree(C1, C2):
    """
    This function finds the minimum spanning tree using Prim's algorithm.
    We assume that absent edges have 0 cost. To find the maximum spanning
    tree, use -1*C.
    We partition the nodes into those in U and those not in U.
    closest[i] is the vertex in U that is closest to i in V-U.
    lowcost[i] is the cost of the edge [i, closest[i]], or infinity if i has
    been used.
    For details see
        - Aho, Hopcroft & Ullman 1983, "Data structures and algorithms",
        p 237.
    Parameters
    ----------
    C1: Numpy matrix
        C1[i,j] is the primary cost of connecting i to j.
    C2: Numpy matrix
        C2[i,j] is the (optional) secondary cost of connecting i to j, used
        to break ties.
    """
    n = C1.shape[0]
    A = np.zeros((n,n))

    closest = np.zeros((1,n))
    used = np.zeros((1,n))
    used[0,0] = 1
    C1 = C1 + np.nan_to_num((C1 == 0) * np.Inf )
    C2 = C2 + np.nan_to_num((C2 == 0) * np.Inf )
    lowcost1 = C1[0,:]
    lowcost2 = C2[0,:]

    for i in range(1,n):
        ks = find(np.array(lowcost1) == np.min(lowcost1))
        k = ks[0, np.argmin(lowcost2[0, ks])]
        k=int(k)
        A[int(k), int(closest[0,int(k)])] = 1
        A[int(closest[0,k]), k] = 1
        lowcost1[0,k] = np.nan_to_num(np.Inf)
        lowcost2[0,k] = np.nan_to_num(np.Inf)
        used[0,k] = 1
        NU = find(used == 0)

        for ji in range(0, NU.shape[1]):
            j = NU[0, ji]
            if C1[k, j] < lowcost1[0, j]:
                lowcost1[0, j] = float(C1[k, j])
                lowcost2[0, j] =  float(C2[k, j])
                closest[0, j] =  float(k)

    return A

def mk_rooted_tree(G, root):
    """
    This function reproduces G as a directed tree pointing away from the
    root.
    Parameters
    ----------
    G: Numpy ndarray
        G[i,j] = 1 iff there is an edge between node i and node j.
    root: Int
        The index of the root node.
    Output
    ------
    T: Numpy ndarray
        The rooted tree, T[i,j] = 1 iff there is an edge between node i and
        node j.
    pre: List
        The pre visting order.
    post: List
        The post visting order.
    cycle: Int
        Equals 1 if there is a cycle in the rooted tree.
    """
    n = G.shape[0]
    T = np.zeros((n, n))
    directed = 0

    """Preform depth first search"""
    searched = dfs(G, root, directed)

    for i in range(0, searched.pred.shape[1]):
        if searched.pred[0, i]!=-1:
            T[searched.pred[0, i], i] = 1

    return [T, searched.pre, searched.post, searched.cycle]