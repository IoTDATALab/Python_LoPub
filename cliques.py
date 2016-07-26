#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module supplies the classes used to implement different types
of cliques.
"""
    
__docformat__ = 'restructuredtext'

import numpy as np
import potentials
import general

class discrete_clique(object):
    """
    A clique with an attached discete potential.
    """    
    def __init__(self, id_num=None, domain=None, sizes=None, T=None):
        """
        Creates and initializes a clique object.

        Parameters
        ----------
        id_num : Integer
            A identifier number for this clique. A tool to identify it among
            a list of cliques. It is best to ensure that each clique object
            has a unique identifier.

        domain: List of integers
            The list of nodes that this clique encompasses.

        sizes: List of integers
            A list of the size of each node in the clique. If sizes[2] = 10,
            then node 2 can assume 1 of 10 different states.

        T : Numpy array
            The look up table of discrete probabalities assigned to this
            clique.
        """
        """Check if this is supposed to be a blank clique"""
        if (domain == None) and (sizes == None):
            self.pot = None
            self.unobserved_pot = None
        else:
            """Set the identifier and the domain"""
            self.id = id_num
            self.domain = domain[:]
            
            """
            Set the unobserved potential of the clique, this potential
            will not be changed in any of the inference algorithm's, and
            can only be changed explicitly. It is used to initialize the
            observed potential based on any observed evidence.
            """
            self.unobserved_pot =  potentials.dpot(domain, sizes, T)

            """
            This pontential will be changed by entering evidence and running
            inference algorithms.
            """
            self.pot = self.unobserved_pot.copy()

        """Initialize expected sufficient statistics to 1"""
        self.ess = np.array([1])
        
        """
        The dictionary nbrs stores the I.D.'s of the cliques that are
        neighbours to this clique, these are used as keys to find the
        variable nodes that seperate this clique from its neighbours.
        Therefore if clique i is a nighbour of this clique, and
        self.nbrs[i] = [3, 5], then the variable nodes with ID's
        3 and 5 seperate clique i from this clique.
        """
        self.nbrs = dict()

    def enter_evidence(self, evidence, maximize=False):
        """
        Enter observed evidence into this clique's working potential.

        Parameters
        ----------
        evidence: List
            A list of observed values for the nodes in this clique.
            [] represents a hidden node.
        """
        """Reinitialize the working potential with the unobserved potential"""
        self.pot = self.unobserved_pot.copy()

        """
        If the potential is being used in a max-sum algorithm, log the
        working potential.
        """
        if maximize:
            self.pot.T = np.log(self.pot.T)
        
        """Enter the evidence to the working potential"""
        self.pot.enter_evidence(evidence)
      

    def init_sep_pots(self, node_sizes, onodes, max_sum=False):
        """
        Intialize the seperator potentials, which are the messages stored
        at the variable nodes, before being sent to the cliques.

        Parameters
        ----------
        node_sizes: Array
            A list of the sizes of each node in the model. If sizes[2] = 10,
            then node 2 can assume 1 of 10 different states.

        onodes: List
            A list of all the observed nodes in the model.

        max_sum: Bool
            Max_sum is true, if the max_sum algorithm is going to be used
            on this clique, or false otherwise. It indicates whether the
            potentials must be initialized to ones (for sum-product),
            or zeros (for max-sum). Since max-sum uses log's to evaluate the
            maximum likely configuration.
        """
        for i in self.nbrs.iterkeys():
            """Create intial potential object"""
            sep_pot = potentials.mk_initial_pot('d', self.nbrs[i][0], \
                                                node_sizes,\
                                                [], onodes)
            """
            If this clique is being used in a max-sum algorithm,
            initialize the tables message to zero instead of one.
            """
            if max_sum == True:
                sep_pot.T = sep_pot.T * 0
                
            """Assign potential object to the neighbour"""
            self.nbrs[i][1] = sep_pot

    def reset_ess(self):
        """
        Reset the expected sufficient statistics for this clique.
        """
        self.ess = np.zeros((1, np.prod(self.unobserved_pot.T.shape)))

    def update_ess(self, sample, expected_vals, node_sizes):
        """
        Update the expected sufficient statistics for this clique.

        Parameters
        ----------
        sample: List
            A partially observed sample of the all the nodes in the model
            this clique is part of. sample[i] = [] if node i in unobserved.

        expected_vals: marginal
            A marginal object containing the expected values for any unobserved
            nodes in this clique.

        node_sizes: Array
            A list of the sizes of each node in the model. If sizes[2] = 10,
            then node 2 can assume 1 of 10 different states.
        """
        [hidden, observed] = general.determine_observed(sample)

        if general.issubset(np.array(self.domain), np.array(hidden)):
            """
            If the entire domain of the clique was unobserved in
            the last sample. Then the marginal over the cliques domain will
            be just the cliques entire potential. Therefore we can add this
            directly to the cliques expected sufficient statistics.
            """
            self.ess = self.ess + expected_vals.T.flatten()
        else:
            """
            If any part of the cliques domain was observed, the expected values
            for the observed domain has been marginalized out. Therefore
            we need to pump the marginal up to its correct dimensions based on
            the observed evidence, and place the observed values where the
            'expected' values were.
            """
            expected_vals.add_ev_to_dmarginal(sample, node_sizes)

            """
            Add the new values to the cliques expected sufficient statistics.
            """
            self.ess = self.ess + expected_vals.T.flatten()

    def maximize_params(self):
        """
        Maximize the parameters from the expected sufficent statistics.
        """
        ess = np.array(self.ess).reshape(self.unobserved_pot.T.shape)
        self.unobserved_pot.T = general.mk_stochastic(ess)
        

    def copy(self):
        """
        Creates am exact copy of this clique.
        """
        copy_clq = discrete_clique()
        copy_clq.id = self.id
        copy_clq.domain = self.domain[:]
        copy_clq.pot = self.pot.copy()
        copy_clq.unobserved_pot = self.unobserved_pot.copy()
        copy_clq.ess = self.ess.copy()
        copy_clq.nbrs = dict()
        for nbr in self.nbrs:
            copy_clq.nbrs[nbr] = [[], []]
            copy_clq.nbrs[nbr][0] = self.nbrs[nbr][0]
            if self.nbrs[nbr][1] != None:
                copy_clq.nbrs[nbr][1] = self.nbrs[nbr][1].copy()
    
        return copy_clq
