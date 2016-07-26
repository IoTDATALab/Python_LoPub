#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module defines classes and functions used to implement potentials.
"""

__docformat__ = 'restructuredtext'

import numpy as np
from general import find, mk_multi_index

class dpot(object):
    """
    A discrete potential object
    """
    def __init__(self, domain, sizes, T=None):
        """
        Initializes a discrete potential object

        Parameters
        ----------
        domain: List
            A list of indices of the nodes defining the domain of
            the potential.

        sizes: List
            A list of the node sizes corresponding to all the nodes in
            the domain of the potential.

        T: Numpy ndarray
            A table defining the potential. Optional, defaults to a table
            of ones for the specified domain.
        """
        self.domain = domain[:]
        self.observed_domain = []
        self.sizes = sizes.copy()
        if T == None:
            sizes = np.array(sizes, dtype='int').tolist()
            self.T = np.ones(sizes)
        else:
            self.T = T.copy()
    
    def copy(self):
        """
        Creates a fresh copy of this potential.
        """
        new_pot = dpot(self.domain, self.sizes, self.T)
        new_pot.observed_domain = self.observed_domain[:]
        return new_pot

    def arithmatic(self, pot, op='+'):
        """
        Perfrom addition, subtraction, multiplication or division between
        this potential and another.

        Parameters
        ----------
        pot: dpot object
            The potential object to add, subtract, multiply or divide with
            this one.

        op: str
            The operation to perform, '+' for addition, '-' for subtraction,
            '*' for multiplication and '/' for division.
        """
        """
        Extend the domain of the potential with the smaller domain
        so that both have the same domain.
        """
        if len(self.sizes) != 1:
            pos = []
            for i in pot.domain:
                pos.append(self.domain.index(i))

            sz = np.ones((1, len(self.domain)), dtype='int')
            sz[0, pos] = pot.sizes
            sz = sz.tolist()
            sz = sz[0]
            Ts = pot.T.reshape(sz)

            for i in xrange(0, len(self.sizes)):
                if (i not in pos):
                    Ts = np.repeat(Ts, self.sizes[i], i)
        else:
            Ts = pot.T.copy()

        Ts = Ts.reshape(standardize_sizes(self.sizes))

        """Perform the required arithmatic operation"""
        if op == '+':
            self.T = self.T + Ts
        elif op == '-':
            self.T = self.T - Ts
        elif op == '*':
            self.T = self.T * Ts
        else:
            self.T = self.T / (Ts + (Ts == 0))
        
    def normalize_pot(self):
        """
        This method converts the discrete potential Pr(X,E) into Pr(X|E)
        and returns log(Pr(E))
        """
        s = np.sum(self.T)
        s = s + (s == 0)
        self.T = self.T / s

        return np.log(s)

    def enter_evidence(self, evidence):
        """
        Enters observed evidence into potential by taking 'slices' out
        of its look up table.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.

        """
        """
        Convert the potential to a table, and obtain a list of the observed
        nodes.
        """
        odom = self.convert_to_table(evidence)

        """
        Observed nodes can only have one value, so the size of an observed
        node is always 1.
        """
        odom = np.intersect1d(self.domain, odom).tolist()
        self.observed_domain = odom[:]
        odom_index =  []
        for i in odom:
            if i in self.domain:
                odom_index.append(self.domain.index(i))

        self.sizes[odom_index] = 1

        self.T = self.T.reshape(standardize_sizes(self.sizes))

    def convert_to_table(self, evidence):
        """
        This function evaluates a tabular CPT using observed evidence,
        by taking 'slices' of the CPT. Returns the 'sliced' CPT and a list
        of the observed nodes.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence, where evidence[i] = [] if
            node i is hidden, or evidence[i] = SomeValue if node i has been
            observed as having SomeValue.
        """
        odom = []
        vals = []
        positions = []
        count = 0
        for i in self.domain:
            if type(evidence[i]) != list:
                odom.append(i)
                vals.append(evidence[i])
                positions.append(count)
            count = count + 1

        """
        The following code has the effect of 'slicing' the table. The idea is to
        select a certain slice out of each dimension, in this case, select slice
        vals[i] out of dimension positions[i]. So if positions = [0, 2, 3], and
        we were slicing 4-D array T, the resulting slice would be equal to
        T[vals[0], :, vals[1], vals[2]].
        """
        index = mk_multi_index(len(self.domain), positions, vals)
        self.T = self.T[index]
        self.T = self.T.squeeze()

        return odom

    def marginalize_pot(self, onto, maximize=False):
        """
        This method marginalizes (or maximizes) a discrete potential onto
        a smaller domain.

        Parameters
        ----------
        onto: List
            The list of nodes specifying the domain onto which to
            marginalize.

        maiximize: Bool
            This value is false if the function must marginalize
            the potential over a domain, and true if it must maximize
            the potential over a domain.
        """        
        ns = np.zeros((1, np.max(self.domain)+1))
        ns[0, self.domain] = self.sizes

        """Marginalize the table"""
        smallT = self.T

        """Determine which dimensions to sum/max over"""
        sum_over = np.setdiff1d(np.array(self.domain), np.array(onto))
        ndx = []
        for i in sum_over:
            temp = find(np.array([np.array(self.domain) == i]))
            if temp.shape != (1,):
                ndx.append(temp[0,0])
        ndx = np.array(ndx)

        maximizers = dict()
        if maximize:
            """
            Determine which variables to resulting argmax values will
            be dependants on. These values are used for back tracking.
            """
            dependants = np.setdiff1d(np.array(self.domain[:]),
                np.array(self.observed_domain[:])).squeeze().tolist()
            if type(dependants) != list:
                dependants = [dependants]

            count = 0
            for i in xrange(0, len(ndx)):
                if ndx[i]<smallT.ndim:
                    """
                    If this node is unobserved, save its backtracking info.
                    """
                    if sum_over[count] not in self.observed_domain[:]:
                        """Save backtracking information"""
                        if sum_over[count] in dependants:
                            dependants.remove(sum_over[count])

                        """Determine which values maximized the array"""
                        argmax = np.argmax(smallT, ndx[i]).squeeze()
                        if argmax.shape == ():
                            argmax = np.array(argmax).tolist()

                        """Save backtracking data"""
                        maximizers[sum_over[count]] = \
                                [dependants[:], argmax]

                    """Maximize out the required dimensions"""
                    smallT = np.max(smallT, ndx[i])

                    """Compensate for reduced dimensions of smallT"""
                    ndx = ndx - 1
                    count = count + 1
        else:
            for i in xrange(0, len(ndx)):
                if ndx[i]<smallT.ndim:
                    """Sum over the dimension ndx[i]"""
                    smallT = np.sum(smallT, ndx[i])
                    """Compensate for reduced dimensions of smallT"""
                    ndx = ndx - 1

        """Create marginalized potential"""
        smallpot = dpot(onto, ns[0, onto], smallT)
        return [smallpot, maximizers]

    def find_most_prob_entry(self):
        """
        Determines the most probable entry for this potential.
        """
        m = np.max(self.T)
        indices = np.argwhere(self.T.reshape(self.sizes) == m)
        self.T =  np.zeros(self.sizes)
        self.T[tuple(indices.flatten().tolist())] = m

        return indices

    def __str__(self):
        """
        Prints the values of various the members of a discrete potential
        object to the console.
        """
        print 'domain: \n', self.domain, '\n'
        print 'T: \n', self.T, '\n'
        print 'sizes: \n', self.sizes, '\n'

        return ''


def mk_initial_pot(pot_type, dom, ns, cnodes, onodes):
    """
    Creates blank potential objects with a specified domain.

    Parameters
    ----------
    pot_type: String
        A character indicating which type of potential to create,
        for instance 'd' for discrete.

    dom: List
        A list of indices of the nodes defining the domain of the new
        potential.

    ns: List
        A list of the node sizes corresponding to all the nodes in the
        network.

    cnodes: List
        A list of indices of the contiuous nodes within the domain.

    onodes: List
        A list of indices of the observed nodes within the domain.
    """
    if pot_type == 'd':
        """Observed can only have one possible value."""
        ns[0, onodes] = 1
        inds = np.array(dom, dtype='int').tolist()
        
        """Create the blank potential object"""
        pot = dpot(dom, ns[0, inds])

    return pot

def standardize_sizes(sizes):
    """
    Removes trailing ones from a list.

    Parameters
    ----------
    sizes: List
        A list of integers.
    """
    while (sizes[-1] == 1) and len(sizes)>2:
        sizes = sizes[0:-1]

    return sizes
