#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module provides classes that implement Markov random field and
Bayesian network objects.
"""
__docformat__ = 'restructuredtext'

import numpy as np
from scipy import sparse
import graph
import cpds
import general
import inference

class model(object):
    """
    A graphical model.
    """
    def sum_product(self, evidence):
        """
        Execute the propagation phase of the sum-product algortihm on this
        graphical model.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        loglik = self.engine.sum_product(evidence)

        return loglik

    def max_sum(self, evidence):
        """
        Execute the propagation phase of the max-sum algortihm on this
        graphical model.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        mlc = self.engine.max_sum(evidence)

        return mlc

    def marginal_nodes(self, query, maximize=False):
        """
        Marginalize a set of nodes out of a clique.

        Parameters
        ----------
        query: List
            A list of the indices of the nodes to marginalize onto. This set
            of nodes must be a subset of one of the triangulated cliques (exact)
            or input user cliques (approximate) within the inference engine.
            Marginalizing onto a single node will always work, because every
            node is the subset of some clique.

        maximize: Bool
            This value is set to true if we wish to maximize instead of
            marginalize, and False otherwise.
        """
        m = self.engine.marginal_nodes(query, maximize)

        return m

    
class mrf(model):
    """
    A markov random field.
    """
    def __init__(self, model_graph, node_sizes, clqs, lattice=False):
        """
        Initializes MRF object.

        model_graph: Numpy array or Scipy.sparse matrix
            A matrix defining the edges between nodes in the network. If
            graph[i, j] = 1 there exists a undirected edge from node i to j.

        node_sizes: List or Int
            A list of the possible number of values a discrete
            node can have. If node_sizes[i] = 2, then the discrete node i
            can have one of 2 possible values, such as True or False. If
            this parameter is passed as an integer, it indicates that all
            nodes have the size indicated by the integer.

        clqs: List of clique objects (cliques.py)
            A list of the cliques in the MRF.

        lattice: Bool
            Lattice is true if this MRF has a lattice graph structure, and
            false otherwise.
        """
        """Assign the input values to their respective internal data members"""
        self.lattice = lattice
        self.num_nodes = model_graph.shape[0]
        self.cliques = clqs
        self.node_sizes = node_sizes

        """Convert the graph to a sparse matrix"""
        if ((type(model_graph) == type(np.matrix([0]))) or
           (type(model_graph) == type(np.array([0])))):
            model_graph = sparse.lil_matrix(model_graph)

        """In an MRF, all edges are bi-directional"""
        self.model_graph = model_graph - \
                           sparse.lil_diags([sparse.extract_diagonal(\
                               model_graph)], [0], (model_graph.shape[0], \
                                                    model_graph.shape[0]))\
                                                    + model_graph.T
        
        """
        Obtain elimination order, which is just the input order in the case
        of a lattice.
        """
        if self.lattice == True:
            self.order = range(0, self.model_graph.shape[0])
        else:
            self.order = graph.topological_sort(self.model_graph)

    def init_inference_engine(self, exact=True, max_iter=10):
        """
        Determine what type of inference engine to create, and intialize it.

        Parameters
        ----------
        exact: Bool
            Exact is TRUE if the type of inference must be exact, therefore,
            using the junction tree algorithm. And exact is FALSE if the type
            of inference must be approximate, therefore, using the loopy belief
            algorithm.

        max_iter: Int
            If the type of inference is approximate, then this value is maximum
            number of iterations the loopy belief algorithm can execute.
        """
        if exact:
            if self.lattice:
                print 'WARNING: Exact inference on lattice graphs not recommened'
            self.engine = inference.jtree_inf_engine(self)
        elif self.lattice:
            """
            This version of the approximate inference engine has been
            optimized for lattices.
            """
            self.engine = inference.belprop_mrf2_inf_engine(self, max_iter)
        else:
            self.engine = inference.belprop_inf_engine(self, max_iter)

    def learn_params_mle(self, samples):
        """
        Maximum liklihood estimation (MLE) parameter learing for a MRF.

        Parameters
        ----------
        samples: List
            A list of fully observed samples for the spanning the total domain
            of this MRF. Where samples[i][n] is the i'th sample for node n.
        """
        samples = np.array(samples)
        """For every clique"""
        for clq in self.cliques:
            """Obtain the evidence that is within the domain of this clique"""
            local_samples = samples[:, clq.domain]
            
            """If there is evidence for this clique"""
            if len(local_samples.tolist()) != 0:
                """Compute the counts of the samples"""
                counts = general.compute_counts(local_samples, clq.pot.sizes)

                """Reshape the counts into a potentials lookup table"""
                clq.unobserved_pot.T =\
                        general.mk_stochastic(np.array(counts, dtype=float))
                clq.pot.T = clq.unobserved_pot.T.copy()
                

    def learn_params_EM(self, samples, max_iter=10, thresh=np.exp(-4), \
                        exact=True, inf_max_iter=10):
        """
        EM algorithm parameter learing for a MRF, accepts partially
        observed samples.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this MRF. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.    
        """
        """Set all the cliques parameters to ones"""
        for i in range(0, len(self.cliques)):
            self.cliques[i].unobserved_pot.T = \
                            np.ones(self.cliques[i].unobserved_pot.T.shape)

        """Create data used in the EM algorithm"""
        loglik = 0
        prev_loglik = -1*np.Inf
        converged = False
        num_iter = 0

        """Init the training inference engine for the new BNET"""
        self.init_inference_engine(exact, inf_max_iter)
       
        while ((not converged) and (num_iter < max_iter)):
            
            """Perform an EM iteration and gain the new log likelihood"""
            loglik = self.EM_step(samples)
  
            """Check for convergence"""
            delta_loglik = np.abs(loglik - prev_loglik)
            avg_loglik = np.nan_to_num((np.abs(loglik) + \
                                        np.abs(prev_loglik))/2)
            if (delta_loglik / avg_loglik) < thresh:
                 """Algorithm has converged"""
                 break
            prev_loglik = loglik
            
            """Increase the iteration counter"""
            num_iter = num_iter + 1
            
    def EM_step(self, samples):
        """
        Perform an expectation step and a maximization step of the EM
        algorithm.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this MRF. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.       
        """
        """Reset every cliques's expected sufficient statistics"""
        for clique in self.cliques:
            clique.reset_ess()

        """
        Set the log liklihood to zero, and loop through every sample in the
        sample set.
        """
        loglik = 0
        for sample in samples:
            """Enter the sample as evidence into the inference engine"""
            sample_loglik = self.sum_product(sample[:])
            loglik = loglik + sample_loglik

            """For every clique in the MRF"""
            for clique in self.cliques:
                """
                Perform a marginalization over the entire cliques domain.
                This will result in a marginal containing the information
                for any nodes that were unobserved in the last entered sample,
                and will remove the 'expected' values for nodes that have been
                observed. Therefore, we are determining probability of the
                hidden nodes given the observed nodes and the current
                model parameters.
                """
                expected_vals = self.engine.marginal_nodes(clique.domain)

                """Update this cliques expected sufficient statistics"""
                clique.update_ess(sample[:], expected_vals, self.node_sizes)

        """Maximize the parameters"""
        for clique in self.cliques:
            clique.maximize_params()

        return loglik
       
    def __str__(self):
        """
        Prints the values of the various members of a mrf object
        to the console.
        """
        print 'Model Graph:\n', self.model_graph
        print '\nNumber of nodes: \n', self.num_nodes
        print '\nNode Sizes: \n', self.node_sizes
        print '\nCliques: \n', self.cliques
        print '\nPotentialss: \n', self.pots
        print '\nLattice \n', self.lattice
        print '\nOrder: \n', self.order
        return ''

class bnet(model):
    """
    A Bayesian network object.
    """
    def __init__(self, model_graph, node_sizes, node_cpds=[]):
        """
        Initializes BNET object.

        model_graph: Numpy array or Scipy.sparse matrix
            A matrix defining the edges between nodes in the network. If
            graph[i, j] = 1 there exists a directed edge from node i to j.

        node_sizes: List or Int
            A list of the possible number of values a discrete
            node can have. If node_sizes[i] = 2, then the discrete node i
            can have one of 2 possible values, such as True or False. If
            this parameter is passed as an integer, it indicates that all
            nodes have the size indicated by the integer.

        node_cpds: List of CPD objects (cpds.py)
            A list of the CPDs for each node in the BNET.
        """
        """Set the data members to the input values"""
        self.model_graph = model_graph
        self.num_nodes = model_graph.shape[0]
        self.node_sizes = node_sizes.copy()
        self.cpds = node_cpds

        """Convert the graph to a sparse matrix"""
        if ((type(model_graph) == type(np.matrix([0]))) or
           (type(model_graph) == type(np.array([0])))):
            model_graph = sparse.lil_matrix(model_graph)
            
        """Obtain topological order"""
        self.order = graph.topological_sort(self.model_graph)

    def init_inference_engine(self, exact=True, max_iter=10):
        """
        Determine what type of inference engine to create, and intialize it.

        Parameters
        ----------
        exact: Bool
            Exact is TRUE if the type of inference must be exact, therefore,
            using the junction tree algorithm. And exact is FALSE if the type
            of inference must be approximate, therefore, using the loopy belief
            algorithm.

        max_iter: Int
            If the type of inference is approximate, then this value is maximum
            number of iterations the loopy belief algorithm can execute.
        """
        if exact:
            self.engine = inference.jtree_inf_engine(self, mrf=False)
        else:
            self.engine = inference.belprop_inf_engine(self, mrf=False, \
                                                       max_iter=10)

    def learn_params_mle(self, samples):
        """
        Maximum liklihood estimation (MLE) parameter learing for a BNET.

        Parameters
        ----------
        samples: List
            A list of fully observed samples for the spanning the total domain
            of this BNET. Where samples[i][n] is the i'th sample for node n.
        """
        """Convert the samples list to an array"""
        samples = np.array(samples)
        
        """If the CPD's have not yet been initialized"""
        if len(self.cpds) == 0:
            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """Create a blank CPD for the node"""
                family = graph.family(self.model_graph, i)
                self.cpds.append(cpds.tabular_CPD(i, self.node_sizes, \
                                                  self.model_graph))
                """Get the samples within this nodes CPDs domain"""
                local_samples = samples[:, family]

                """Learn the node parameters"""
                if len(local_samples.tolist()) != 0:
                    self.cpds[i].learn_params_mle(local_samples)
        else:
            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """Get the samples within this nodes CPDs domain"""
                family = graph.family(self.model_graph, i)
                local_samples = samples[:, family]
                
                """Learn the node parameters"""
                if len(local_samples.tolist()) != 0:
                    self.cpds[i].learn_params_mle(local_samples)

    def learn_params_EM(self, samples, max_iter=10, thresh=np.exp(-4), \
                        exact=True, inf_max_iter=10):
        """
        EM algorithm parameter learing for a BNET, accepts partially
        observed samples.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this BNET. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.    
        """
        """If the CPDs have not yet been defined, then create them"""
        if len(self.cpds) == 0:
            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """Create a blank CPD for the node"""
                family = graph.family(self.model_graph, i)
                self.cpds.append(cpds.tabular_CPD(i, self.node_sizes, \
                                                  self.model_graph))
        else:
            """If they have been defined, reset the CPT's"""
            for i in range(0, self.num_nodes):
                self.cpds[i].CPT = np.ones(self.cpds[i].CPT.shape)

        """Create data used in the EM algorithm"""
        loglik = 0
        prev_loglik = -1*np.Inf
        converged = False
        num_iter = 0

        """Init the training inference engine for the new BNET"""
        self.init_inference_engine(exact, inf_max_iter)

        while ((not converged) and (num_iter < max_iter)):
            
            """Perform an EM iteration and gain the new log likelihood"""
            loglik = self.EM_step(samples)

            """Check for convergence"""
            delta_loglik = np.abs(loglik - prev_loglik)
            avg_loglik = np.nan_to_num((np.abs(loglik) + \
                                        np.abs(prev_loglik))/2)
            if (delta_loglik / avg_loglik) < thresh:
                 """Algorithm has converged"""
                 break
            prev_loglik = loglik
            
            """Increase the iteration counter"""
            num_iter = num_iter + 1

            
    def EM_step(self, samples):
        """
        Perform an expectation step and a maximization step of the EM
        algorithm.

        Parameters
        ----------
        samples: List
            A list of partially observed samples for the spanning the total
            domain of this BNET. Where samples[i][n] is the i'th sample for
            node n. samples[i][n] can be [] if node n was not observed in the
            i'th sample.       
        """
        """Reset every CPD's expected sufficient statistics"""
        for cpd in self.cpds:
            cpd.reset_ess()

        """
        Set the log liklihood to zero, and loop through every sample in the
        sample set.
        """
        loglik = 0
        for sample in samples:
            """Enter the sample as evidence into the inference engine"""
            sample_loglik = self.sum_product(sample[:])
            loglik = loglik + sample_loglik

            """For every node in the BNET"""
            for i in range(0, self.num_nodes):
                """
                Perform a marginalization over the entire CPDs domain.
                This will result in a marginal containing the information
                for any nodes that were unobserved in the last entered sample,
                and will remove the 'expected' values for nodes that have been
                observed. Therefore, we are determining probability of the
                hidden nodes given the observed nodes and the current
                model parameters.
                """
                expected_vals = self.engine.marginal_family(i)

                """Update this nodes CPD's expected sufficient statistics"""
                self.cpds[i].update_ess(sample[:], expected_vals,\
                                        self.node_sizes)

        """Perform maximization step"""
        for cpd in self.cpds:
            cpd.maximize_params()
            
        return loglik

        

    def __str__(self):
        """
        Prints the values of the various members of a bnet object
        to the console.
        """
        print 'Model Graph:\n', self.model_graph
        print '\nNumber of nodes: \n', self.num_nodes
        print '\nNode Sizes: \n', self.node_sizes
        print '\CPDs: \n', self.cpds
        print '\nOrder: \n', self.order
        return ''

        
        
