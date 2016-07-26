#Copyright 2009 Almero Gouws, <14366037@sun.ac.za>
"""
This module contains the classes used to perform inference on various
graphical models.
"""
__docformat__ = 'restructuredtext'

import numpy as np
from numpy.testing import assert_array_almost_equal
import general
import graph
import cliques
import potentials
import models

class jtree_inf_engine(object):
    """
    An exact inference engine for both MRF's and BNETs.
    """
    def __init__(self, model, mrf=True):
        """
        Intializes the inference engine.

        Parameters
        ----------
        Model: Either a MRF or BNET object (models.py)
            The model to perfrom infernece on.
        """
        """
        Initialize the engine for the specific model it is being assigned to.
        """
        if mrf:
            self.init_for_mrf(model)
        else:
            self.init_for_bnet(model)
            
    def init_for_mrf(self, mrf):
        """
        Initializes a junction tree inference engine for specific 
        markov random field. The process in this function is known as the
        'Compilation phase' of the junction tree algorithm.

        Parameters
        ----------
        mrf: mrf object
            The markov random field to perform inference on.
        """
        """Initialize all the data members"""
        self.jtree = []
        self.model = mrf

        """
        Now perform triagulation and obtain the junction tree from the input
        undirected graph.
        """
        [self.jtree, root, clq_doms, B, w] = \
                     graph.graph_to_jtree(self.model.model_graph,\
                                          self.model.node_sizes)

        """
        Store the clique bitvector, which is used to lookup which nodes
        are in which cliques, and store the clique weights, of how many
        possible configuarations the nodes within each clique can assume.
        """
        self.cliques_bitv = B
        self.clique_weight = w
        
        """
        Create a set of discrete cliques representing the cliques in the
        triangulated graph.
        """
        self.tri_cliques = []
        count = 0
        for clq_dom in clq_doms:
            """Create a clique with a blank potential"""
            clq_dom = np.array(clq_dom, dtype=int).tolist()
            self.tri_cliques.append(cliques.discrete_clique(count, clq_dom[:], \
                                    self.model.node_sizes[0, clq_dom]))
            
            """
            Now create the new potential for the triangulated clique
            by combining the input clique potentials that it consists of.
            """
            tri_pot = self.tri_cliques[-1].unobserved_pot
            for clique in self.model.cliques:
                """
                Determine if this original clique can contribute to the
                new clique.
                """
                if len(clique.domain) < len(clq_dom):
                    inter = np.intersect1d(np.array(clique.domain), \
                                           np.array(clq_dom))
                else:
                    inter = np.intersect1d(np.array(clq_dom), \
                                           np.array(clique.domain))

                """
                If there is an intersection, incorporate the required
                information from this clique to the new clique.
                """
                if len(inter) == len(clique.domain):
                    """
                    Make a copy of the original potential and marginalize out
                    the unrequired nodes.
                    """
                    temp_pot = clique.unobserved_pot.marginalize_pot(inter)[0]

                    """
                    Multiply the marginalized original potential into the
                    new triangulated potential.
                    """
                    tri_pot.arithmatic(temp_pot, '*')
            count = count + 1

        """
        Now find the seperation set between all the cliques, start by finding
        the indices of all the edges in the jtree graph matrix.
        """
        xs = []
        ys = []
        for x in range(0, self.jtree.shape[0]):
            temp = graph.find(np.array([self.jtree[x, :]]) > 0)
            if temp.shape[1] != 0:
                xs.append(x)
                ys.append(temp[0, :].tolist())

        """Determine the intersection between each set of connected cliques"""
        for x in xs:
            for y in ys[x]:
                """
                Determine the seperating domain between clique x and clique y.
                """
                sep_dom = graph.find(np.array([((B[x, :] + B[y, :]) \
                                                == 2)])).tolist()[0]

                """
                Clique x is seperated from clique y by the nodes indexed in
                sep_dom.
                """
                self.tri_cliques[x].nbrs[y] = [sep_dom, None]

        """
        Now that we have the the jtree, we convert it to a directed jtree
        pointing away from its root. This is done using a depth-first search,
        and the pre and post visting order obtained from the DFS is used in the
        progagation phase of the algorithm.
        """
        rooted_tree = graph.mk_rooted_tree(self.jtree, len(self.tri_cliques)-1)
        self.jtree = rooted_tree[0]
        self.preorder = rooted_tree[1]
        self.postorder = rooted_tree[2]

        
        self.postorder_parents = []
        for n in range(0, len(self.postorder)):
            self.postorder_parents.insert(int(n), \
                                          (graph.parents(self.jtree, n)))

        self.preorder_children = []
        for n in range(0, len(self.preorder)):
            self.preorder_children.insert(int(n), \
                                          (graph.children(self.jtree, n)))
    def init_for_bnet(self, bnet):
        """
        Initializes a junction tree inference engine for specific 
        bayesian network. The process in this function is known as the
        'Compilation phase' of the junction tree algorithm.

        Parameters
        ----------
        bnet: bnet object
            The bayesian network to perform inference on.
        """
        """Initialize all the data members"""
        self.jtree = []
        self.model = bnet

        """Obtained the moralized verion of the directed acyclic graph"""
        self.model_graph = graph.moralize(bnet.model_graph)[0]

        """
        Now perform triabgulation and obtain the junction tree from the
        graph moralized graph.
        """
        [self.jtree, root, clq_doms, B, w] = \
                     graph.graph_to_jtree(self.model_graph,\
                                          self.model.node_sizes)

        """
        Store the clique bitvector, which is used to lookup which nodes
        are in which cliques, and store the clique weights, of how many
        possible configuarations the nodes within each clique can assume.
        """
        self.cliques_bitv = B
        self.clique_weight = w

        """
        A node can be a member of many cliques, but is assigned to exactly one,
        to avoid double-counting its CPD. We assign node i to clique c if c is
        the "lightest" clique that contains i's family, so it can accomodate
        its CPD.
        """
        self.clq_ass_to_node = np.zeros((1, self.model.num_nodes), dtype=int)
        for i in range(0, self.model.num_nodes):
            """Find the domain this node belongs in"""
            f = graph.family(self.model.model_graph, i)

            """Find which cliques its domain is a subset of"""
            clqs_containing_family = graph.find(np.array( \
                [np.all(B[:, f] == 1, 1)]) == 1)

            """Determine which of the discovered cliques are the lightest"""
            c = clqs_containing_family[0 , np.argmin(w[\
                clqs_containing_family.tolist(), 0])]

            """Assign the node to a clique"""
            self.clq_ass_to_node[0, i] = c

        self.clq_ass_to_node = self.clq_ass_to_node.squeeze().tolist()
       
        """
        Create a set of discrete cliques representing the cliques in the
        triangulated graph. Start by converting all the CPD's to
        unobserved potentials.
        """
        node_pots = []
        for i in range(0, self.model.num_nodes):
            """Determine the domain of the new clique potential"""
            domain = graph.family(self.model.model_graph, i)

            """Create the clique potentials from the CPD"""
            node_pots.append(self.model.cpds[i].convert_to_pot(domain, []))       

        """
        Initialize the triangulated clique potentials to the right dimensions.
        """
        count = 0
        self.tri_cliques = []
        for clq_dom in clq_doms:
            """Create a clique with a blank potential"""
            clq_dom = np.array(clq_dom, dtype=int).tolist()
            self.tri_cliques.append(cliques.discrete_clique(count, clq_dom[:], \
                                    self.model.node_sizes[0, clq_dom]))
            """
            Now create the new potential for the triangulated clique
            by combining the input clique potentials that it consists of.
            """
            tri_pot = self.tri_cliques[-1].unobserved_pot
            node_ass = self.clq_ass_to_node[:]
            while count in node_ass:
                ndx = node_ass.index(count)
                tri_pot.arithmatic(node_pots[ndx], '*')
                node_ass[ndx] = -1
            count = count + 1

        """
        Now find the seperation set between all the cliques, start by finding
        the indices of all the edges in the jtree graph matrix.
        """
        xs = []
        ys = []
        for x in range(0, self.jtree.shape[0]):
            temp = graph.find(np.array([self.jtree[x, :]]) > 0)
            if temp.shape[1] != 0:
                xs.append(x)
                ys.append(temp[0, :].tolist())

        """Determine the intersection between each set of connected cliques"""
        for x in xs:
            for y in ys[x]:
                """
                Determine the seperating domain between clique x and clique y.
                """
                sep_dom = graph.find(np.array([((B[x, :] + B[y, :]) \
                                                == 2)])).tolist()[0]

                """
                Clique x is seperated from clique y by the nodes indexed in
                sep_dom.
                """
                self.tri_cliques[x].nbrs[y] = [sep_dom, None]

        """
        Now that we have the the jtree, we convert it to a directed jtree
        pointing away from its root. This is done using a depth-first search,
        and the pre and post visting order obtained from the DFS is used in the
        progagation phase of the algorithm.
        """
        rooted_tree = graph.mk_rooted_tree(self.jtree, len(self.tri_cliques)-1)
        self.jtree = rooted_tree[0]
        self.preorder = rooted_tree[1]
        self.postorder = rooted_tree[2]

        self.postorder_parents = []
        for n in range(0, len(self.postorder)):
            self.postorder_parents.insert(int(n), \
                                          (graph.parents(self.jtree, n)))

        self.preorder_children = []
        for n in range(0, len(self.preorder)):
            self.preorder_children.insert(int(n), \
                                          (graph.children(self.jtree, n)))

    def sum_product(self, evidence):
        """
        Execute the propagation phase of the sum-product algortihm.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        """Determine which nodes are observed"""
        [hnodes, onodes] = general.determine_observed(evidence)
        
        """
        Enter the evidence into the clique potentials, and prepare its
        seperator potentials.
        """
        for clique in self.tri_cliques:
            """Make a copy of the unobserved clique potential to work with"""
            clique.pot = clique.unobserved_pot.copy()

            """Enter the evidence into this clique"""
            clique.enter_evidence(evidence)

            """Initialize this cliques seperator potentials"""
            clique.init_sep_pots(self.model.node_sizes.copy(), onodes, False)

        """
        Begin propagation by sending messages from the leaves to the root of the
        jtree, updating the clique and seperator potentials.
        """
        for n in self.postorder:
            n = int(n)
            for p in self.postorder_parents[n]:
                """
                Get handles to neighbouring cliques for easy readability.
                """
                clique_n = self.tri_cliques[n]
                clique_p = self.tri_cliques[p]
                
                """
                Send message from clique n to variable nodes seperating
                clique n and clique p.
                """
                clique_n.nbrs[p][1] = clique_n.pot.marginalize_pot(\
                    clique_n.nbrs[p][0], False)[0]
                
                """Send message from variable nodes to clique p"""
                clique_p.pot.arithmatic(clique_n.nbrs[p][1], '*')

        """
        The jtree is now in an incosistant state, restore consistancy and
        complete the propagation phase by sending messages from the root to the
        leaves of the jtree, updating the clique and seperator potentials.
        """
        for n in self.preorder:
            n = int(n)
            for c in self.preorder_children[n]:
                """
                Get handles to neighbouring cliques for easy readability.
                """
                clique_n = self.tri_cliques[n]
                clique_c = self.tri_cliques[c]

                """
                Before sending a message from clique n to clique c, divide
                out the old message from clique n to clique c
                """
                clique_c.pot.arithmatic(clique_n.nbrs[c][1], '/')
                
                """
                Send message from clique n to variable nodes seperating
                clique n and clique c.
                """
                clique_n.nbrs[c][1] = clique_n.pot.marginalize_pot(\
                    clique_n.nbrs[c][0], False)[0]

                """Send message from variable nodes to clique c"""
                clique_c.pot.arithmatic(clique_n.nbrs[c][1], '*')

        """Normalize the clique potentials"""
        for clique in self.tri_cliques:
            loglik = clique.pot.normalize_pot()

        return loglik

    def max_sum(self, evidence):
        """
        Execute the propagation phase of the max_sum algortihm.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        """Determine which nodes are observed"""
        [hnodes, onodes] = general.determine_observed(evidence)
        
        """
        Enter the evidence into the clique potentials, and prepare its
        seperator potentials.
        """
        for clique in self.tri_cliques:
            """Make a copy of the unobserved clique potential to work with"""
            clique.pot = clique.unobserved_pot.copy()

            """Enter the evidence into this clique, and log its potential"""
            clique.enter_evidence(evidence, True)

            """Initialize this cliques seperator potentials"""
            clique.init_sep_pots(self.model.node_sizes.copy(), onodes, True)


        """Check for special case where there is only one clique"""
        if len(self.tri_cliques) == 1:
            """
            If there is only one clique, we simply find the maximum of that
            cliques potential.
            """
            mlc = evidence[:]
            max_track = np.argwhere(self.tri_cliques[0].pot.T == \
                                    np.max(self.tri_cliques[0].pot.T))[0]

            """Assign the maximimizing values to the nodes"""
            for i in range(0, self.model.num_nodes):
                if type(mlc[i]) == list:
                    mlc[i] = max_track[i]
        else:
            """
            Initialize maximizing tracker, used to perform back tracking once the
            algorithm has converged.
            """
            max_track = dict()
            
            """
            Begin propagation by sending messages from the leaves to the root of the
            jtree, updating the clique and seperator potentials.
            """
            order = []
            for n in self.postorder:
                n = int(n)
                for p in self.postorder_parents[n]:
                    """
                    Get handles to neighbouring cliques for easy readability.
                    """
                    clique_n = self.tri_cliques[n]
                    clique_p = self.tri_cliques[p]
                    
                    """
                    Send message from clique n to variable nodes seperating
                    clique n and clique p.
                    """               
                    ans = clique_n.pot.marginalize_pot(clique_n.nbrs[p][0], True)
                    clique_n.nbrs[p][1] = ans[0]
                    for key in ans[1]:
                        if key not in order:
                            order.insert(0, key)
                        max_track[key] = ans[1][key]
                            
                    """Send message from variable nodes to clique p"""
                    clique_p.pot.arithmatic(clique_n.nbrs[p][1], '+')

            """
            Send messages to all the non-separation nodes of the root clique
            in the tree to finalize that data for the back-tracking. First
            determine which nodes are not separation nodes.
            """
            non_sep_nodes = clique_p.domain[:]
            for nbr in clique_p.nbrs:
                for node in clique_p.nbrs[nbr][0]:
                    if node in non_sep_nodes:
                        non_sep_nodes.remove(node)

            """
            Now send a message from the root clique to each one of its nodes.
            """
            for node in non_sep_nodes:
                ans = clique_p.pot.marginalize_pot([node], True)
                for key in ans[1]:
                    if key not in order:
                        order.insert(0, key)    
                    max_track[key] = ans[1][key]

            """
            Begin backtracking by setting the MAP values of all observed nodes
            to their observed values, if the last node is unobserved, set it to
            its calculated MAP value.
            """               
            mlc = evidence
            if type(mlc[node]) == list:
                mlc[node] = np.argmax(self.marginal_nodes([node], True).T)

            """
            Perform backtracking by traversing back through the order, and
            setting the MAP values of each unobserved node along the way. 
            """
            for node in order:
                if type(mlc[node]) == list:
                    dependants = max_track[node][0]
                    argmax = max_track[node][1]
                    ndx = []
                    for dependant in dependants:
                        ndx.append(mlc[dependant])
                    if len(ndx) != 0:
                        mlc[node] = argmax[tuple(ndx)]
                    else:
                        mlc[node] = argmax

        return mlc
           
    def marginal_nodes(self, query, maximize=False):
        """
        Marginalize a set of nodes out of a clique.

        Parameters
        ----------
        query: List
            A list of the indices of the nodes to marginalize onto. This set
            of nodes must be a subset of one of the triangulated cliques.

        maximize: Bool
            This value is set to true if we wish to maximize instead of
            marginalize, and False otherwise.
        """
        """Determine which clique, if any, contains the nodes in query"""
        c = self.clq_containing_nodes(query)
        if c == -1:
            print 'There is no clique containing node(s): ', query
            return None
        else:
            """Perform marginalization"""
            m = self.tri_cliques[c].pot.marginalize_pot(query, maximize)[0]
            m = marginal(m.domain, m.T)

        return m

    def marginal_family(self, query):
        """
        """
        """Get the family of the query node"""
        fam = graph.family(self.model.model_graph, query)

        """Determine which clique the query node was assigned to"""
        c = int(self.clq_ass_to_node[query])

        """Marginalize the clique onto the domain of the family"""
        m = self.tri_cliques[c].pot.marginalize_pot(fam)[0]
        m = marginal(m.domain, m.T)
        
        return m

    def clq_containing_nodes(self, query):
        """
        Finds the lightest clique (if any) that contains the set of nodes.
        Returns c = -1 if there is no such clique.

        Parameters
        ----------
        query: List
            A list of node the indices we wish to find the containing
            clique of.
        """
        if type(query) != list:
            query = [query]
            
        B = self.cliques_bitv
        w = self.clique_weight
        clqs = graph.find(np.array([np.all(B[:, query] == 1, 1)]) == 1)
        if clqs.shape[1] == 0:
            c = -1
        else:
            c = clqs[0, np.argmin(w[clqs, 0])]

        return int(c)
    
class belprop_inf_engine(object):
    """
    An approximate inference engine, using Loopy belief propagation, for both
    MRFs and BNETs.
    """
    def __init__(self, model, mrf=True, max_iter=10):
        """
        Intializes the inference engine.

        Parameters
        ----------
        model: Either a MRF or BNET object (models.py)
            The model to perfrom infernece on.

        max_iter: Int
            The maximum number of iterations the algorithm can run for.
        """
        """
        Initialize the engine for the specific model it is being assigned to.
        """
        if mrf:
            self.init_for_mrf(model, max_iter)
        else:
            self.init_for_bnet(model, max_iter)
    
    def init_for_mrf(self, mrf, max_iter=10):
        """
        Initializes the inference engine.  

        Parameters
        ----------
        mrf: Markov random field object, class located in models.py.
            The markov random field on which to perform inference. .

        max_iter: Int
            The maximum number of iterations of belief propogation allowed,
            (Default: 10)
        """
        """Initialize object data members"""
        self.model = mrf
        self.max_iter = max_iter
        self.cliques = self.model.cliques
                    
        self.num_cliques = len(self.cliques)
        self.nbrs = []       

        """
        Find the seperation between all the cliques using the clique domains.
        """
        for clique_i in self.cliques:
            """Find all cliques with domains that intersect with clique i"""
            for clique_j in self.cliques:
                if clique_i.id  != clique_j.id:
                    """
                    Determine the intersection between clique i and clique j.
                    """
                    sep_nodes = np.intersect1d(np.array(clique_i.domain), \
                                         np.array(clique_j.domain)).tolist()

                    """
                    If there is an intersection between the two cliques, then
                    they are neighbours. Save the seperator set between in
                    clique i and clique j in clique i's neighbour list. 
                    """
                    if len(sep_nodes) != 0:
                        clique_i.nbrs[clique_j.id] = [sep_nodes, None]

    def init_for_bnet(self, bnet, max_iter=10):
        """
        Initializes the inference engine.  

        Parameters
        ----------
        mrf: Bayesian network object, class located in models.py.
            The Bayesian network on which to perform inference. 

        max_iter: Int
            The maximum number of iterations of belief propogation allowed,
            (Default: 10)
        """
        """Initialize object data members"""
        self.model = bnet
        self.max_iter = max_iter

        """Convert conditional probability densitys to clique potentials"""
        self.cliques = []
        for i in range(0, self.model.num_nodes):
            """Determine the domain of the new clique potential"""
            domain = graph.family(self.model.model_graph, i)

            """Create the clique potential from the CPD"""
            clq_pot = self.model.cpds[i].convert_to_pot(domain, [])
            
            self.cliques.append(cliques.discrete_clique(i, clq_pot.domain, \
                                                        clq_pot.sizes,\
                                                        clq_pot.T))
            
        self.num_cliques = len(self.cliques)
        self.nbrs = []       

        """
        Find the seperation between all the cliques using the clique domains.
        """
        for clique_i in self.cliques:
            """Find all cliques with domains that intersect with clique i"""
            for clique_j in self.cliques:
                if clique_i.id  != clique_j.id:
                    """
                    Determine the intersection between clique i and clique j.
                    """
                    sep_nodes = np.intersect1d(np.array(clique_i.domain), \
                                         np.array(clique_j.domain)).tolist()

                    """
                    If there is an intersection between the two cliques, then
                    they are neighbours. Save the seperator set between in
                    clique i and clique j in clique i's neighbour list. 
                    """
                    if len(sep_nodes) != 0:
                        clique_i.nbrs[clique_j.id] = [sep_nodes, None]

    def sum_product(self, evidence):
        """
        Perform the loopy sum-product algorithm.
        
        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        """
        Enter the observed evidence, start by determining which
        nodes are observed.
        """
        [hnodes, onodes] = general.determine_observed(evidence)

        """Create handle to the cliques for better code readability"""
        clqs = self.cliques
        for clq in clqs:
            """Enter the observed evidence"""
            clq.enter_evidence(evidence)
            
            """
            Initialize the separator potentials, which are the intermediate
            messages between cliques are stored. The potentials represent the
            information stored at a variable node.
            """
            clq.init_sep_pots(self.model.node_sizes.copy(), onodes, False)
     
        """Initialize storage for the belief at each clique"""
        if self.num_cliques != 1:
            bel = np.zeros((1, self.num_cliques)).squeeze().tolist()
        else:
            bel = [[]]

        """
        Initialize maximizing tracker, used to perform back tracking once the
        algorithm has converged.
        """
        max_track = dict()
        
        """
        Perform message passing for loopy-belief propagation, iterate until
        the algorithm has converged, or the maximum number iterations has been
        reached.
        """
        converged = False
        iteration = 0
        print 'Start message passing routine...'
        while (not converged) and (iteration < self.max_iter):
##            print '     Running iteration:', iteration,'...'
            iteration = iteration + 1

            """
            Running the 'collection phase', in which each clique receives the
            messages from its neighbours, and sums them into its belief.
            """
            old_clqs = []
            for clq in clqs:
                """
                The belief at the current clique, is the result of its
                observed potential summed with the sum of all the messages
                from its neighbouring cliques. Start by extracting the
                observed potential.
                """
                bel[clq.id] = clq.pot.copy()
                
                """
                Now ,sum the messages passed to this clique from this cliques
                neighbours with its belief.
                """
                for nbr in clq.nbrs:                   
                    bel[clq.id].arithmatic(clqs[nbr].nbrs[clq.id][1], '*')
                bel[clq.id].normalize_pot()
                
                """
                Make copy of old clique before introducing new info, this is
                used in the 'distribution phase' to remove old messages from
                beliefs before sending the new messages out.
                """
                old_clqs.append(clq.copy())                      

            """
            Now perform the 'distribution phase', where each clique updates
            its outgoing messages to its neighbours, which can then be recieved
            by the neighbours in the next collection phase.
            """
            for clq in clqs:
                for nbr in clq.nbrs:
                    """
                    Find the seperation set between this clique and its
                    neighbour.
                    """
                    sep_set = clq.nbrs[nbr][0]

                    """
                    Remove the old message sent from this clique to its
                    neighbour, using the copy of this clique from the last
                    iteration.
                    """
                    msg_to_nbr = bel[clq.id].copy()
                    msg_to_nbr.arithmatic(old_clqs[nbr].nbrs[clq.id][1], '/')
                  
                    """
                    Update the seperator messages, which are the messages the
                    cliques will recieve in the next iteration. These are
                    messages sent from cliques to variable nodes, and are
                    determined via marginalization.
                    """
                    ans = msg_to_nbr.marginalize_pot(sep_set, False)

                    """Save the new message"""
                    clq.nbrs[nbr][1] = ans[0].copy()           
            
            """
            Save a copy of the beliefs, which is used to determine most likely
            configuration once the algorithm has converged.
            """
            self.marginal_domains = bel

            if iteration < 3:
                """
                If this is the first iteration, find the current most likely
                configuration, and save it.
                """
                old_bel = []
                for b in bel:
                    old_bel.append(b.copy())
            else:
                """
                If more that 3 iterations have passed, check for convergence
                by determining whether the most likely configuration has
                changed between the last iteration and this one. If it hasn't,
                then assume the algorithm has converged.
                """
                try:
                    for i in xrange(0, len(bel)):
                        assert_array_almost_equal(bel[i].T, old_bel[i].T, 3)
                    break
                except:
                    pass
                
                old_bel = []
                for b in bel:
                    old_bel.append(b.copy())
          
    def max_sum(self, evidence):
        """
        Find the most likely configuration of the all nodes in the network
        given some evidence.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        """
        Enter the observed evidence, start by determining which
        nodes are observed.
        """
        [hnodes, onodes] = general.determine_observed(evidence)

        """Create handle to the cliques for better code readability"""
        clqs = self.cliques
        for clq in clqs:
            """Enter the observed evidence"""
            clq.enter_evidence(evidence)
           
            """
            Initialize the separator potentials, which are the intermediate
            messages between cliques are stored. The potentials represent the
            information stored at a variable node.
            """
            clq.init_sep_pots(self.model.node_sizes.copy(), onodes, True)
     
        """Initialize storage for the belief at each clique"""
        if self.num_cliques != 1:
            bel = np.zeros((1, self.num_cliques)).squeeze().tolist()
        else:
            bel = [[]]

        """
        Initialize maximizing tracker, used to perform back tracking once the
        algorithm has converged.
        """
        max_track = dict()
        
        """
        Perform message passing for loopy-belief propagation, iterate until
        the algorithm has converged, or the maximum number iterations has been
        reached.
        """
        converged = False
        iteration = 0
        while (not converged) and (iteration < self.max_iter):
##            print '     Running iteration:', iteration,'...'
            iteration = iteration + 1

            """
            Running the 'collection phase', in which each clique receives the
            messages from its neighbours, and sums them into its belief.
            """
            old_clqs = []
            for clq in clqs:
                """
                The belief at the current clique, is the result of its
                observed potential summed with the sum of all the messages
                from its neighbouring cliques. Start by extracting the
                observed potential.
                """
                bel[clq.id] = clq.pot.copy()
                
                """
                Now ,sum the messages passed to this clique from this cliques
                neighbours with its belief.
                """
                for nbr in clq.nbrs:                   
                    bel[clq.id].arithmatic(clqs[nbr].nbrs[clq.id][1])
                    
                
                """
                Make copy of old clique before introducing new info, this is
                used in the 'distribution phase' to remove old messages from
                beliefs before sending the new messages out.
                """
                old_clqs.append(clq.copy())                      

            """
            Now perform the 'distribution phase', where each clique updates
            its outgoing messages to its neighbours, which can then be recieved
            by the neighbours in the next collection phase.
            """
            for clq in clqs:
                for nbr in clq.nbrs:
                    """
                    Find the seperation set between this clique and its
                    neighbour.
                    """
                    sep_set = clq.nbrs[nbr][0]

                    """
                    Remove the old message sent from this clique to its
                    neighbour, using the copy of this clique from the last
                    iteration.
                    """
                    msg_to_nbr = bel[clq.id].copy()
                    msg_to_nbr.arithmatic(old_clqs[nbr].nbrs[clq.id][1], '-')
                  
                    """
                    Update the seperator messages, which are the messages the
                    cliques will recieve in the next iteration. These are
                    messages sent from cliques to variable nodes, and are
                    determined via marginalization.
                    """
                    ans = msg_to_nbr.marginalize_pot(sep_set, True)

                    """Save the new message"""
                    clq.nbrs[nbr][1] = ans[0].copy()

                    """
                    Save the maximizing values of the nodes seperating these
                    cliques. These values are important as they are used to
                    perform the back-tracking to find the most likely
                    configuration.
                    """
                    for key in ans[1]:
                        max_track[key] = ans[1][key]
                        
                for node in clq.domain:
                    ans = clq.pot.marginalize_pot([node], True)
                    for key in ans[1]:
                        if key not in max_track.keys():
                            max_track[key] = ans[1][key]
                        
            """
            Save a copy of the beliefs, which is used to determine most likely
            configuration once the algorithm has converged.
            """
            self.marginal_domains = bel
            if iteration == 1:
                """
                If this is the first iteration, find the current most likely
                configuration, and save it.
                """
                mlc = self.back_track(evidence, clqs, max_track)
                old_mlc = mlc[:]
            elif iteration > 3:
                """
                If more that 3 iterations have passed, check for convergence
                by determining whether the most likely configuration has
                changed between the last iteration and this one. If it hasn't,
                then assume the algorithm has converged.
                """
                """
                Perform the back-tracking to find the most likely
                configuaration of the hidden nodes.
                """
                mlc = self.back_track(evidence, clqs, max_track)

                """Check for convergence"""
                if np.sum(np.array(mlc) != np.array(old_mlc)) == 0:
                    break
                old_mlc = mlc[:]
        
        return mlc

    def back_track(self, evidence, clqs, max_track):
        """
        Perform's back-tracking to determine the most likely configuaration
        any hidden nodes in the model. This is required because there could
        be multiple maximizing configuarations, this process ensures all nodes
        are set to the values from only one of these configuarations.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
            
        clqs: List of clique objects
            A list of the cliques in the model.
            
        max_track: Dictionary
            The dictionary containing the different maximizing configurations,
            obtained by running the max-sum algorithm.
        """
        mlc = evidence[:]

        [hnodes, set_nodes] = general.determine_observed(evidence)
        
        """
        Determine the best node to begin backtracking from, which is the node
        with that is has the most nodes dependant on it, and is dependant on
        the most nodes.
        """
        score = np.zeros((1, \
                          self.model.num_nodes), dtype=int).squeeze().tolist()
        for node in max_track:
            if node in hnodes:
                for vals in max_track.values():
                    if node in vals[0]:
                        score[node] = score[node] + 1
                score[node] = score[node] + len(max_track[node][0])

        node = np.argmax(np.array(score))
        set_nodes.append(node)       

        if type(mlc[node]) == list:
            mlc[node] = np.argmax(self.marginal_nodes([node], True).T)
     
        """
        Now set all the other varaible nodes to the values corresponding to
        the same maximum configuaration that the first node has been set to.
        Begin by looping through all the cliques.
        """
        not_set_nodes = np.setdiff1d(np.array(range(0, self.model.num_nodes)),
                                     np.array(set_nodes)).tolist()

        while len(not_set_nodes) != 0:
            for node in max_track:
                if type(mlc[node]) == list and \
                   general.issubset(np.array(max_track[node][0]),\
                                    np.array(set_nodes)):
                    
                    dependants = max_track[node][0]
                    argmax = max_track[node][1]

                    ndx = []
                    for dependant in dependants:
                        if type(mlc[dependant]) != list:
                            ndx.append(mlc[dependant])
                    
                    if len(ndx) != 0:
                        mlc[node] = argmax[tuple(ndx)]
                    else:
                        mlc[node] = argmax

                    not_set_nodes.remove(node)
                    set_nodes.append(node)
                    
        return mlc
                        
    def marginal_nodes(self, query, maximize=False):
        """
        Computes the marginal on the specified query nodes.

        Parameters
        ----------
        query: List
            A list of nodes to marginalize onto, must be a subset of some
            clique in the graph.

        maximize: Bool
            Maximize is equal to True if the function must perform maximization
            instead of marginalization, and false otherwise.
        """
        if type(query) != list:
            query = [query]

        """Find which clique the node/s belongs to"""
        found = False
        for i in xrange(0, len(self.cliques)):
            if general.issubset(np.array(query), \
                                np.array(self.cliques[i].domain)):
                found = True
                break

        if found == True:
            """Marginalize over the query nodes"""
            pot = self.marginal_domains[i].marginalize_pot(query, maximize)[0]
            m = marginal(pot.domain, pot.T)
        else:
            print 'ERROR: The query nodes are not a subset of any clique!'
            m = None
        
        return m


class belprop_mrf2_inf_engine(object):
    """
    An approximate inference engine, using Loopy belief propagation, for Markov
    Random fields with pair-wise defined cliques. 
    """
    def __init__(self, mrf, max_iter=10):
        """
        Initializes the inference engine.  

        Parameters
        ----------
        mrf: Markov random field object, class located in models.py.
            The markov random field on which to perform inference. The MRF
            is only allowed to have cliques consisting of 2 nodes each.

        max_iter: Int
            The maximum number of iterations of belief propogation allowed,
            (Default: 10)
        """
        """Initialize object data members"""
        self.model = mrf
        self.max_iter = max_iter
        self.num_cliques = len(self.model.cliques)
        self.nbrs = []

        """
        Find the seperation between all cliques, since the cliques are
        pairwise, there will only ever be one node seperating any 2
        cliques. Start by collecting all the clique domains into a list.
        """
        clqs = []
        for i in self.model.cliques:
            clqs.append(i.domain)
        clqs = np.array(clqs)       

        """
        Find the seperation between all the cliques using the clique domains.
        """
        for i in xrange(0, self.num_cliques):
            """
            Find all cliques that contain the first of the two nodes in
            this clique.
            """
            node = clqs[i, 0]
            sep_1 = np.argwhere(clqs[:, 0] == node).squeeze().tolist()
            sep_2 = np.argwhere(clqs[:, 1] == node).squeeze().tolist()
            
            if type(sep_1) == int:
                sep_1 = [sep_1]
            if type(sep_2) == int:
                sep_2 = [sep_2]

            sep_1.extend(sep_2)

            """
            Make sure that this clique does not appear in the list of
            neighbouring cliques.
            """
            if i in sep_1:
                sep_1.remove(i)

            """Assign the list of neighbours and seperators to clique i"""
            for j in sep_1:
                """
                The spot for the seperator potential is intialized to None.
                """
                self.model.cliques[i].nbrs[j] = [[node], None]
            
            """
            Now find all cliques that contain the second of the two nodes in
            this clique.
            """
            node = clqs[i, 1]
            sep_1 = np.argwhere(clqs[:, 0] == node).squeeze().tolist()
            sep_2 = np.argwhere(clqs[:, 1] == node).squeeze().tolist()
            
            if type(sep_1) == int:
                sep_1 = [sep_1]
            if type(sep_2) == int:
                sep_2 = [sep_2]

            sep_1.extend(sep_2)

            """
            Once again, make sure that this clique does not appear in the
            list of neighbouring cliques.
            """
            if i in sep_1:
                sep_1.remove(i)
                
            """Assign the list of neighbours and seperators to clique i"""
            for j in sep_1:
                """
                The spot for the seperator potential is intialized to None.
                """
                self.model.cliques[i].nbrs[j] = [[node], None]

    def sum_product(self, evidence):
        """
        Perform the loopy sum-product algorithm.
        
        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        """
        Enter the observed evidence, start by determining which
        nodes are observed.
        """
        [hnodes, onodes] = general.determine_observed(evidence)

        """Create handle to the cliques for better code readability"""
        clqs = self.model.cliques
        for clq in clqs:
            """Enter the observed evidence"""
            clq.enter_evidence(evidence)
            
            """
            Initialize the separator potentials, which are the intermediate
            messages between cliques are stored. The potentials represent the
            information stored at a variable node.
            """
            clq.init_sep_pots(self.model.node_sizes.copy(), onodes, False)
     
        """Initialize storage for the belief at each clique"""
        if self.num_cliques != 1:
            bel = np.zeros((1, self.num_cliques)).squeeze().tolist()
        else:
            bel = [[]]

        """
        Initialize maximizing tracker, used to perform back tracking once the
        algorithm has converged.
        """
        max_track = dict()
        
        """
        Perform message passing for loopy-belief propagation, iterate until
        the algorithm has converged, or the maximum number iterations has been
        reached.
        """
        converged = False
        iteration = 0
        print 'Start message passing routine...'
        while (not converged) and (iteration < self.max_iter):
            print '     Running iteration:', iteration,'...'
            iteration = iteration + 1

            """
            Running the 'collection phase', in which each clique receives the
            messages from its neighbours, and sums them into its belief.
            """
            old_clqs = []
            for clq in clqs:
                """
                The belief at the current clique, is the result of its
                observed potential summed with the sum of all the messages
                from its neighbouring cliques. Start by extracting the
                observed potential.
                """
                bel[clq.id] = clq.pot.copy()
                
                """
                Now ,sum the messages passed to this clique from this cliques
                neighbours with its belief.
                """
                for nbr in clq.nbrs:                   
                    bel[clq.id].arithmatic(clqs[nbr].nbrs[clq.id][1], '*')
                bel[clq.id].normalize_pot()
                
                """
                Make copy of old clique before introducing new info, this is
                used in the 'distribution phase' to remove old messages from
                beliefs before sending the new messages out.
                """
                old_clqs.append(clq.copy())                      

            """
            Now perform the 'distribution phase', where each clique updates
            its outgoing messages to its neighbours, which can then be recieved
            by the neighbours in the next collection phase.
            """
            for clq in clqs:
                for nbr in clq.nbrs:
                    """
                    Find the seperation set between this clique and its
                    neighbour.
                    """
                    sep_set = clq.nbrs[nbr][0]

                    """
                    Remove the old message sent from this clique to its
                    neighbour, using the copy of this clique from the last
                    iteration.
                    """
                    msg_to_nbr = bel[clq.id].copy()
                    msg_to_nbr.arithmatic(old_clqs[nbr].nbrs[clq.id][1], '/')
                  
                    """
                    Update the seperator messages, which are the messages the
                    cliques will recieve in the next iteration. These are
                    messages sent from cliques to variable nodes, and are
                    determined via marginalization.
                    """
                    ans = msg_to_nbr.marginalize_pot(sep_set, False)

                    """Save the new message"""
                    clq.nbrs[nbr][1] = ans[0].copy()           
            
            """
            Save a copy of the beliefs, which is used to determine most likely
            configuration once the algorithm has converged.
            """
            self.marginal_domains = bel

            if iteration < 3:
                """
                If this is the first iteration, find the current most likely
                configuration, and save it.
                """
                old_bel = []
                for b in bel:
                    old_bel.append(b.copy())
            else:
                """
                If more that 3 iterations have passed, check for convergence
                by determining whether the most likely configuration has
                changed between the last iteration and this one. If it hasn't,
                then assume the algorithm has converged.
                """
                try:
                    for i in xrange(0, len(bel)):
                        assert_array_almost_equal(bel[i].T, old_bel[i].T, 3)
                    break
                except:
                    pass
                
                old_bel = []
                for b in bel:
                    old_bel.append(b.copy())
          
    def max_sum(self, evidence):
        """
        Find the most likely configuration of the all nodes in the network
        given some evidence.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
        """
        """
        Enter the observed evidence, start by determining which
        nodes are observed.
        """
        [hnodes, onodes] = general.determine_observed(evidence)

        """Create handle to the cliques for better code readability"""
        clqs = self.model.cliques
        for clq in clqs:
            """Enter the observed evidence"""
            clq.enter_evidence(evidence)
            
            """
            Initialize the separator potentials, which are the intermediate
            messages between cliques are stored. The potentials represent the
            information stored at a variable node.
            """
            clq.init_sep_pots(self.model.node_sizes.copy(), onodes, True)
     
        """Initialize storage for the belief at each clique"""
        if self.num_cliques != 1:
            bel = np.zeros((1, self.num_cliques)).squeeze().tolist()
        else:
            bel = [[]]

        """
        Initialize maximizing tracker, used to perform back tracking once the
        algorithm has converged.
        """
        max_track = dict()
        
        """
        Perform message passing for loopy-belief propagation, iterate until
        the algorithm has converged, or the maximum number iterations has been
        reached.
        """
        converged = False
        iteration = 0
        while (not converged) and (iteration < self.max_iter):
            print '     Running iteration:', iteration,'...'
            iteration = iteration + 1

            """
            Running the 'collection phase', in which each clique receives the
            messages from its neighbours, and sums them into its belief.
            """
            order = []
            old_clqs = []
            for clq in clqs:
                """
                The belief at the current clique, is the result of its
                observed potential summed with the sum of all the messages
                from its neighbouring cliques. Start by extracting the
                observed potential.
                """
                bel[clq.id] = clq.pot.copy()
                
                """
                Now ,sum the messages passed to this clique from this cliques
                neighbours with its belief.
                """
                for nbr in clq.nbrs:                   
                    bel[clq.id].arithmatic(clqs[nbr].nbrs[clq.id][1])
                    
                
                """
                Make copy of old clique before introducing new info, this is
                used in the 'distribution phase' to remove old messages from
                beliefs before sending the new messages out.
                """
                old_clqs.append(clq.copy())                      

            """
            Now perform the 'distribution phase', where each clique updates
            its outgoing messages to its neighbours, which can then be recieved
            by the neighbours in the next collection phase.
            """
            for clq in clqs:
                for nbr in clq.nbrs:
                    """
                    Find the seperation set between this clique and its
                    neighbour.
                    """
                    sep_set = clq.nbrs[nbr][0]

                    """
                    Remove the old message sent from this clique to its
                    neighbour, using the copy of this clique from the last
                    iteration.
                    """
                    msg_to_nbr = bel[clq.id].copy()
                    msg_to_nbr.arithmatic(old_clqs[nbr].nbrs[clq.id][1], '-')
                  
                    """
                    Update the seperator messages, which are the messages the
                    cliques will recieve in the next iteration. These are
                    messages sent from cliques to variable nodes, and are
                    determined via marginalization.
                    """
                    ans = msg_to_nbr.marginalize_pot(sep_set, True)

                    """Save the new message"""
                    clq.nbrs[nbr][1] = ans[0].copy()

                    """
                    Save the maximizing values of the nodes seperating these
                    cliques. These values are important as they are used to
                    perform the back-tracking to find the most likely
                    configuration.
                    """
                    for key in ans[1]:
                        if key not in order:
                            order.insert(0, key)
                        max_track[key] = ans[1][key]
                        
            """
            Save a copy of the beliefs, which is used to determine most likely
            configuration once the algorithm has converged.
            """
            self.marginal_domains = bel
            if iteration == 1:
                """
                If this is the first iteration, find the current most likely
                configuration, and save it.
                """
                mlc = self.back_track(hnodes, evidence, clqs, max_track)
                old_mlc = mlc[:]
            elif iteration > 3:
                """
                If more that 3 iterations have passed, check for convergence
                by determining whether the most likely configuration has
                changed between the last iteration and this one. If it hasn't,
                then assume the algorithm has converged.
                """
                """
                Perform the back-tracking to find the most likely
                configuaration of the hidden nodes.
                """
                mlc = self.back_track(hnodes, evidence, clqs, max_track)

                """Check for convergence"""
                if np.sum(np.array(mlc) != np.array(old_mlc)) == 0:
                    break
                old_mlc = mlc[:]
        
        return mlc

    def back_track(self, hnodes, evidence, clqs, max_track):
        """
        Perform's back-tracking to determine the most likely configuaration
        any hidden nodes in the model. This is required because there could
        be multiple maximizing configuarations, this process ensures all nodes
        are set to the values from only one of these configuarations.

        Parameters
        ----------
        hnodes: List
            List of the hidden nodes in the model.
            
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.
            
        clqs: List of clique objects
            A list of the cliques in the model.
            
        max_track: Dictionary
            The dictionary containing the different maximizing configurations,
            obtained by running the max-sum algorithm.s
        """
        """
        Set the most probable values of the observed nodes to there
        observed values.
        """
        mlc = evidence[:]
        
        """
        Set the first hidden node to its most likely value via
        marginalization.
        """
        if len(hnodes) != 0:
            node = hnodes[0]
            if type(mlc[node]) == list:
                mlc[node] = np.argmax(self.marginal_nodes([node], True).T)
     
        """
        Now set all the other varaible nodes to the values corresponding to
        the same maximum configuaration that the first node has been set to.
        Begin by looping through all the cliques.
        """
        for clq in clqs:
            """For every neighbour of this clique"""
            for nbr in clq.nbrs:
                """If the SECOND node in this clique is a hidden node"""
                if clq.domain[1] in hnodes:
                    """
                    And if the value of the FIRST node in this clique has
                    already had its maximizing value set.
                    """
                    sep_node = clq.nbrs[nbr][0][0]
                    if (type(mlc[sep_node]) != list) and \
                       (type(mlc[clq.domain[1]]) == list):
                        """
                        Then use the maximized value of the second node, to
                        obtain the maximizing value of first node, that
                        belongs to the same maximzing configuaration, that
                        set the second node.
                        """
                        node = clq.domain[1]
                        dependants = max_track[node][0]
                        argmax = max_track[node][1]
                        if type(argmax) == np.ndarray:
                            mlc[node] = argmax[mlc[sep_node]]
                        else:
                            mlc[node] = argmax
        return mlc
                        
    def marginal_nodes(self, query, maximize=False):
        """
        Computes the marginal on the specified query nodes.

        Parameters
        ----------
        query: List
            A list of nodes to marginalize over, must be a subset of some
            clique in the graph.
        """
        if type(query) != list:
            query = [query]

        """Find which clique the node/s belongs to"""
        found = False
        for i in xrange(0, len(self.model.cliques)):
            if general.issubset(np.array(query), \
                                np.array(self.model.cliques[i].domain)):
                found = True
                break

        if found == True:
            """Marginalize over the query nodes"""
            pot = self.marginal_domains[i].marginalize_pot(query, maximize)[0]
            m = marginal(pot.domain, pot.T)
        else:
            print 'ERROR: The query nodes are not a subset of any clique!'
            m = None
        
        return m
    
class marginal(object):
    """
    Stores information about a marginalized node, or nodes.
    """
    def __init__(self, domain, T, mu=[], sigma=[]):
        self.domain = domain
        self.T = T
        self.mu = mu
        self.sigma = sigma

    def add_ev_to_dmarginal(self, evidence, ns):
        """
        This function 'pumps up' observed nodes back to their original size,
        by introducing zeros into the array positions which are incompatible
        with the evidence.

        Parameters
        ----------
        evidence: List
            A list of any observed evidence. If evidence[i] = [], then
            node i is unobserved (hidden node), else if evidence[i] =
            SomeValue then, node i has been observed as being SomeValue.

        ns: List
            A list of the node sizes of each node in the network.
        """
        dom = self.domain
        odom = []
        equiv = []
        vals = []
        for i in xrange(0, len(dom)):
            if evidence[dom[i]] != []:
                odom.append(dom[i])
                equiv.append(i)
                vals.append(evidence[dom[i]])

        index = general.mk_multi_index(len(dom), equiv, vals)
        T = np.zeros((ns[0, dom]))
        ens = ns.copy()
        ens[0, odom] = 1
        T[index] = np.reshape(self.T, ens[0, dom])
        self.T = T

    def __str__(self):
        """
        Prints the values of the members of the object to the console.
        """
        print 'T: \n', self.T, '\n'
        print 'domain: \n', self.domain, '\n'
        print 'mu: \n', self.mu, '\n'
        print 'sigma: \n', self.sigma, '\n'
        return ''


        

