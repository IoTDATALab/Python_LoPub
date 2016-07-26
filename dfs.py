#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module functions used to implement a Depth-first search on a graph.
The functions contained in this module are:

    'dfs': Performs a depth first search on a graph.
    'dfs_visit': Performs a depth first search visit on a node in a graph.
"""

__docformat__ = 'restructuredtext'

import numpy as np
import graph

class dfs:
    def __init__(self, adj_mat, start=-1, directed=0):
        """
        Performs a depth first search on a graph.

        Parameters
        ----------
        -'adj_mat': Numpy ndarray
            Adjacency matrix. If adj_mat[i, j] = 1, there exists
            a directed edge from node i to node j.
        -'start': Int
            The index of the node to begin the search from.
        -'directed': Int
            Equals 1 if it is a directed graph, 0 otherwise.
        """
        n = adj_mat.shape[0]
        self.white = 0
        self.gray = 1
        self.black = 2
        self.color = np.zeros((1,n))
        self.time_stamp = 0
        self.d = np.zeros((1,n))
        self.f = np.zeros((1,n))
        self.pred = np.ones((1,n))*-1
        self.cycle = 0
        self.pre = []
        self.post = []

        if start != -1:
            self.dfs_visit(start, adj_mat, directed)

        for u in range(0, n):
            if self.color[0,u] == self.white:
                self.dfs_visit(u, adj_mat, directed)

    def dfs_visit(self, u, adj_mat, directed):
        """
        Performs a depth first serach visit on a node in a graph.

        Parameters
        ----------
        -'u': Int
            The index of the node to visit.
        -'adj_mat': Numpy ndarray
            Adjacency matrix. If adj_mat[i, j] = 1, there exists
            a directed edge from node i to node j.
        -'directed': Int
            Equals 1 if it is a directed graph, 0 otherwise.
        """
        self.pre.append(u)
        self.color[0, u] = self.gray
        self.time_stamp = self.time_stamp + 1
        self.d[0, u] = self.time_stamp

        if directed == 1:
          ns = graph.children(adj_mat, u)
        else:
          ns = graph.neighbours(adj_mat, u)
          ns = np.unique(ns)
          ns = np.setdiff1d(np.array(ns), np.array([self.pred[0, u]]))
          ns = ns.tolist()

        for v in ns:
            if self.color[0, v] == self.white:
                self.pred[0, v] = u
                self.dfs_visit(v, adj_mat, directed)
            elif self.color[0, v] == self.gray:
                self.cycle = 1

        self.color[0, u] = self.black
        self.post.append(u)
        self.time_stamp = self.time_stamp + 1
        self.f[0, u] = self.time_stamp
