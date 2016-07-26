#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module contains the classes used to implement the conditional probability
distributions for nodes in a Bayesian network.
"""

__docformat__ = 'restructuredtext'

import numpy as np
import general
import potentials
import graph

class tabular_CPD(object):
    """
    A discrete conditional probability distribution, represented by
    a conditional probability table.
    """
    def __init__(self, node_id, node_sizes, dag, CPT=None):
        """
        Initializes the CPD.

        Parameters
        ----------
        node_id: Int
            The index of the node this CPD is assigned to.

        node_size: Numpy array
            A list of the sizes of the nodes in Bayesian network that this
            CPD is part of.

        DAG: Numpy matrix
            An adjacency matrix representing the directed acyclic graph that
            from the bayesian network this CPD is part of.
        """
        """Set the data members"""
        self.ess = None

        """Determine the nodes parents"""
        parents = graph.parents(dag, node_id)
        self.parents = parents[:]

        """
        The domain sizes of the CPT can be determined by the number of
        values each node forming the CPT can assume. For instance, the
        CPT for P(a|b,c) is formed by the 3 discrete nodes a, b and c. The
        CPT will therefore have 3 dimensions. The size of each dimension
        is determined by the number of values each variable, a, b and c,
        can assume. If a and b can assume 1 of 2 possible values, and c
        can assume 1 of 4 possible values then dimension 1 and 2 are both
        of size 2, and dimension 3 is of size 4. Therefore, this CPT would
        be a 3-D array, with dimensions (2, 2, 4).
        """
        parents.insert(0, node_id)
        self.fam_size = node_sizes[0, parents]

        if CPT != None:
            """If a CPT was defined, assign it to the CPD"""
            self.CPT = CPT
        else:
            """
            If no CPT was defined, create an array of ones of the correct size.
            """
            self.CPT = np.ones((self.fam_size))

    def convert_to_pot(self, domain, evidence=[]):
        """
        Converts a tabular CPD object to a discrete clique potential.

        Parameters
        ----------
        domain: List
            The domain the of the desired clique potential, which should
            be a subset of the CPD's domain.

        evidence: List
            A list of any observed evidence. If any of the nodes in the CPDs
            domain have been observed, they will be clamped at the observed
            value before being incorporated into the potential.
        """
        """Convert the CPD to a table based on the evidence"""
        [T, odom] = self.convert_to_table(domain, evidence)

        """Set the size of observed nodes to 1"""
        for node in odom:
            if node in domain:
                ndx = domain.index(node)
            self.fam_size[ndx] = 1

        """Reshape table to correct size"""
        T = T.reshape(potentials.standardize_sizes(self.fam_size))

        """Create the potential object"""
        pot = potentials.dpot(domain, self.fam_size, T)

        return pot
        

    def convert_to_table(self, domain, evidence=[]):
        """
        This function evaluates a tabular CPD's CPT using observed evidence,
        by taking 'slices' of the CPT. Returns the 'sliced' CPT and a list
        of the observed nodes.

        Parameters
        ----------
        domain: List
            The domain over which the CPD is defined, which is a list of
            indices of the nodes that the CPD encompasses.

        evidence: List
            A list of any observed evidence, where evidence[i] = [] if
            node i is hidden, or evidence[i] = SomeValue if node i has been
            observed as having SomeValue.
        """
        odom = []
        vals = []
        positions = []
        count = 0
        
        """If there is any observed evidence"""
        if len(evidence) != 0:
            """For every node in the CPDs domain"""
            for i in domain:
                """If this node has been observed"""
                if evidence[i] != []:
                    """Add it to the list of observed nodes"""
                    odom.append(i)
                    vals.append(evidence[i])
                    positions.append(count)
                count = count + 1

        """
        The following code has the effect of 'slicing' the table. The idea is to
        select a certain slice out of each dimension, in this case, select slice
        vals[i] out of dimension positions[i]. So if positions = [0, 2, 3], and
        we were slicing 4-D array T, the resulting slice would be equal to
        T[vals[0], :, vals[1], vals[2]]. This has the effect of clamping
        observed variables at their observed values.
        """
        index = general.mk_multi_index(len(domain), positions, vals)
        T = self.CPT[index]
        T = T.squeeze()

        return [T, odom]

    def learn_params_mle(self, samples):
        """
        Maximum liklihood estimation (MLE) parameter learing for a tabular
        CPD.

        Parameters
        ----------
        samples: List
            A list of fully observed samples for the spanning the total domain
            of this CPD. Where samples[i][n] is the i'th sample for node n.
        """
        """Compute the counts of the samples"""
        counts = general.compute_counts(samples, self.fam_size)

        """Reshape the counts into a CPT"""
        self.CPT = general.mk_stochastic(np.array(counts, dtype=float))

    def reset_ess(self):
        """
        Reset the Expected Sufficient Statistics of this CPD
        """
        self.ess = np.zeros((1, np.prod(self.CPT.shape)))

    def update_ess(self, sample, expected_vals, node_sizes):
        """
        Update the expected sufficient statistics for this CPD.

        Parameters
        ----------
        sample: List
            A partially observed sample of the all the nodes in the model
            this CPD is part of. sample[i] = [] if node i in unobserved.

        expected_vals: marginal
            A marginal object containing the expected values for any unobserved
            nodes in this CPD.

        node_sizes: Array
            A list of the sizes of each node in the model. If sizes[2] = 10,
            then node 2 can assume 1 of 10 different states.
        """      
        """Determine which nodes were observed in this sample"""
        [hidden, observed] = general.determine_observed(sample)

        """If the entire domain of the CPD is hidden"""
        if general.issubset(np.array(expected_vals.domain), np.array(hidden)):
            """
            If the entire domain of the CPD was unobserved in
            the last sample. Then the marginal over the CPD domain will
            be just the CPD's entire CPT. Therefore we can add this
            directly to the CPD's expected sufficient statistics.
            """
            self.ess = self.ess + expected_vals.T.flatten()
        else:
            """
            If any part of the CPD's domain was observed, the expected values
            for the observed domain has been marginalized out. Therefore
            we need to pump the marginal up to its correct dimensions based
            on the observed evidence, and place the observed values where the
            'expected' values were for the observed nodes.
            """
            expected_vals.add_ev_to_dmarginal(sample, node_sizes)

            """
            Add the new values to the CPD's expected sufficient statistics.
            """
            self.ess = self.ess + expected_vals.T.flatten()

    def maximize_params(self):
        """
        Maximize the parameters from the expected sufficent statistics.
        """
        ess = np.array(self.ess).reshape(self.CPT.shape)
        self.CPT = general.mk_stochastic(ess)
            
            


        
        
        
        

