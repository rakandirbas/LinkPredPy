'''
Created on May 14, 2014
The Supervised Random Walk implementation
@author: rockyrock
'''

import numpy as np
import mathutils as mu
from numpy import linalg as LA
import graph_utils
from graph_utils import DyadFeaturesIndexLocator
import graphsim as gsim
import networkx as nx
from collections import defaultdict

def main():
    pass
    
    
def power_iter(Q, p=None, e=10**-12):
    """
    Uses power iteration to find the stationary page rank scores p.
    
    Parameters:
    -----------
    Q: the transition matrix.
    p: initial page rank scores, it's None by defualt. When it's None, 
    then all scores are defined to be 1/N where N is the #nodes.
    e: the threshold error, as to when to stop power-iter
    """
    
    e_test = 1
    
    N = Q.shape[0]
    
    if p == None:
        p = np.zeros((N))
        p.fill(1.0/N)
        
    p = p.reshape((1,-1))
        
#     for i in xrange(n_iter):
#          p = np.dot(p, Q)

    old_p = p
    
    while(e_test > e):
        p = np.dot(p, Q)
        e_test = p - old_p
        e_test = np.abs(e_test).sum()
        old_p = p
         
    return p.reshape(-1)

    
def computeP_PP(Q, pQ, p=None, pp=None, err=10**-12):
    """
    Computes the stationary page rank scores P and the partial 
    derivatives of the page rank scores PP.
    
    Parameters:
    -----------
    Q: the transition matrix.
    pQ: the partial derivative of the transition matrix with respect
        to a parameter variable w.
    p: initial page rank scores, it's None by defualt. When it's None, 
    then all scores are defined to be 1/N where N is the #nodes.
    pp: the partial derivatives of the page rank scores with 
        respect to the parameter variable w. When None, then it's 
        initialized to zeros.
    e: the threshold error, as to when to stop loop.
    
    Returns:
    --------
    p: the stationary page rank scores.
    pp: the stationary partial derivatives of the page rank scores
    
    """
    N = Q.shape[0]
    
    if p == None:
        p = np.zeros((N))
        p.fill(1.0/N)
    
    p = p.reshape((1,-1))
    
    if pp == None:
        pp = np.zeros((N))
        
    pp = pp.reshape((1,-1))
    
    err_p = 1
    err_pp = 1
    old_p = p
    old_pp = pp
    while( (err_p > err) or (err_pp > err) ):
        p = np.dot(p, Q)
        pp = np.dot(pp, Q) + np.dot(p, pQ)
        
#         print p, pp
        
        err_p = p - old_p
        err_p = np.abs(err_p).sum()
        old_p = p
        
        err_pp = pp - old_pp
        err_pp = np.abs(err_pp).sum()
        old_pp = pp
        
    p = p.reshape(-1)    
    pp = pp.reshape(-1)
    return p, pp
        
        
class SRW:
    
    #different from original
    def __init__(self, psi, alpha):
        """
        psi: (the Greek character) an object that implements a
         get_feature_vec(source, node1_index, node2_ondex) method, which 
         returns the feature vector of node1 and node2 with respect to a specified source.
         the object also implements get_D(source), get_L(source), num_feats(), get_S() 
         and get_A(source).
        alpha: restart probability
        """
        self.psi = psi
        self.A = None
        self.s = None
        self.w = np.random.normal(0,1,(psi.num_feats()))
        self.alpha = alpha
        self.f = mu.logistic_function
        self.pf = mu.pderv_logistic
        self.h = mu.wmw
        self.ph = mu.derv_wmw

    def get_Q_tick(self):
        """
        Returns Q` as in the paper, i.e. the normalized activations.
        """
        
        N = self.A.shape[0]
        Q = self.A.copy()
        
        zero_rows = np.where(~self.A.any(axis=1))[0]
        if zero_rows.size > 0:
            raise Exception("The adjacency matrix has a zero row")
        
        
        for i in xrange(N):
            nonzero = np.nonzero(Q[i].reshape(-1,))
            nonzero = nonzero[0]
            for j in nonzero:
                x = self.psi.get_feature_vec(self.s, i, j)
                a = self.f( np.dot(x, self.w) )
                Q[i,j] = a
                
        normalization = np.sum(Q, axis=1)
        normalization = normalization.reshape(-1,1)
        Q = Q/normalization
        
        return Q
    
    def get_Q(self, Q_tick, s):
        """
        Returns the transition matrix Q after adding the random jump probabilities.
        
        Parameters:
        -----------
        Q_tick: Q` as in the paper (normalized activations).
        s: the index of the source node
        """
        
        Q = (1-self.alpha) * Q_tick
        Q[:, s] += self.alpha
    
         
        return Q
    
    #different from original
    def get_pQ(self, w_index):
        """
        Returns the partial derivative of the transition matrix
        with respect to one parameter variable w.
        
        Parameters:
        -----------
        w_index: the index of the parameter variable that 
                we are computing the partial derivative for.
        """
        N = self.A.shape[0]
        Q = self.A.copy()
        pQ = self.A.copy()
        
        for i in xrange(N):
            nonzero = np.nonzero(Q[i].reshape(-1,))
            nonzero = nonzero[0]
            for j in nonzero:
                x = self.psi.get_feature_vec(self.s, i, j)
                a = self.f( np.dot(x, self.w) )
                Q[i,j] = a
                
        normalizer = np.sum(Q, axis=1)
        
        for i in xrange(N):
            nonzero = np.nonzero(pQ[i].reshape(-1,))
            nonzero = nonzero[0]
            
            sum_act = normalizer[i]
            sum_pf = 0 #sum of the partial dervs of activations for the row
            dot_prods = {}
            p_fs = {}
            
            for k in nonzero:
                x = self.psi.get_feature_vec(self.s, i, k)
                c_w = x[w_index]#the coefficent of the w variable
                _dotprod = np.dot(x, self.w)
                k_p_f = self.pf(_dotprod, c_w)
                sum_pf += k_p_f
                
                dot_prods[k]=_dotprod
                p_fs[k]=k_p_f
            
            for j in nonzero:
                dotprod = dot_prods[j]
                a = self.f( dotprod )
                p_f = p_fs[j]
                
                pQ[i,j] = (1-self.alpha) * ( ( p_f * sum_act ) - (a*sum_pf) ) / (sum_act)**2
        
        return pQ
    
    #different from origial
    def compute_grad(self, lmbda=1):
        """
        Computes the partial derivatives of the objective function with 
        respect to each weight parameter w.
        
        Parameters:
        -----------
        lmbda: the regularization parameter.
        
        Returns:
        :type pw: 1D numpy array
        :param pw: the partial derivatives of the objective function
                    with respect to each weight parameter w.
        """
        num_w = self.psi.num_feats()
        pw = np.zeros(( num_w ))
        
        
        for w_index in xrange(num_w):
            w = self.w[w_index]
            sum_factor = 0
            countter = 1
#             print 'Computing a W'
            for s in self.psi.get_S():
#                 print 'S number', countter
                countter += 1
                
                self.s = s
                self.A = self.psi.get_A(s)
                s_index = self.psi.get_s_index(s)
                Q_tick = self.get_Q_tick()
                Q = self.get_Q( Q_tick, s_index ) 
                pQ = self.get_pQ(w_index)
                p, pp = computeP_PP(Q,pQ)
                D = self.psi.get_D(s)
                L = self.psi.get_L(s)
                
                for d in D:
                    for l in L:
                        delta = p[l] - p[d]
                        sum_factor += self.ph(delta) * ( pp[l] - pp[d] )
                        
            pw[w_index] = (2 * w) + (lmbda * sum_factor)

            
        return pw
    
    def compute_cost(self):
        #it needs to be updated to the changes!!!
        """
        Computes the cost of the objective function.
        
        Returns:
        --------
        cost: a float that shows the cost
        """
#         Q_tick = self.get_Q_tick()
#         cost = 0
#         for s in self.psi.get_S():
#             Q = self.get_Q( Q_tick , s)
#             p = power_iter(Q)
#             D, L = self.T[s]
#             
#             for d in D:
#                 for l in L:
#                     delta = p[l] - p[d]
#                     cost += self.h(delta)
#                     
#         cost += LA.norm(self.w)**2
        
        return -9999999999999
                
    def optimize(self, n_iter, lmbda=1, eps=0.1, verbose=False):
        """
        Optimize the objective function.
        
        Parameter:
        ----------
        n_iter: number of iterations for gradient descent
        lmbda: the regularization parameter
        eps: gradient descent step-size
        """
        num_w = self.psi.num_feats()
        
        for i in xrange(n_iter):
            pw = self.compute_grad(lmbda=lmbda)
            if verbose:
                print i, self.compute_cost(), self.w, pw
            for w_index in xrange(num_w):
                w = self.w[w_index]
                w = w - (eps) * pw[w_index]       
                self.w[w_index] = w
                
    def get_P(self, s):
        self.s = s
        self.A = self.psi.get_A(s)
        s_index = self.psi.get_s_index(s)
        Q_tick = self.get_Q_tick()
        Q = self.get_Q( Q_tick, s_index )
        
        return Q
                
class RAW_PSI:
    def get_feature_vec(self, s, i, j):
        """
        Returns the feature vector between two nodes i, j with 
        respect to a source node s.
        
        Parameters:
        -----------
        s: the source node ID or something.
        i: the node i index in the adjacency matrix of the source node graph.
        j: the node j index in the adjacency matrix of the source node graph.
        """
        
    def num_feats(self):
        """
        Returns the number of features in-use.
        """
        
    def get_D(self, s):
        """
        Returns a list of nodes indices that are the destinations for the source node.
        
        Parameters:
        -----------
        s: the souce node ID or something
        
        Returns:
        D: a list that represent the nodes indices (with respect to the souce node graph)
         that are the destinations for the source node
        """
        
    def get_L(self, s):
        """
        Returns a list of nodes indices that are the destinations for the source node.
        
        Parameters:
        -----------
        s: the souce node ID or something
        
        Returns:
        D: a list that represent the nodes indices (with respect to the souce node graph)
         that are the destinations for the source node
        """
        
    def get_S(self):
        """
        Returns a list of all source nodes IDs.
        """
        
    def get_A(self, s):
        """
        Returns the adjacency matrix of a source node s.
        It must always return a numpy array and NOOOT a matrix!!!
        Parameters:
        -----------
        s: the source node ID.
        """
        
    def get_s_index(self, s):
        """
        Returns the index of the source node with respect to its adjacency matrix.
        
        Parameters:
        s: the source node ID
        
        Returns:
        s_index: the source node index in the adjacency matrix.
        """
        
        

class GeneralPSI:
    def __init__(self, G, X, k=10, delta=5):
        """
        Parameters:
        -----------
        G: a networkx graph object.
        X: the nodes features matrix. Each row is the feature vector of a node.
        k: the number of neighbours a node must have in order to be considered a candidate source node.
        delta: the number of edges the candidate source node made in the future that close a triangle. 
               (i.e. the future/destination node is a friend of a friend, so delta is a threshold that sees how many
                 of these future/destination nodes are friends of current friends). A candidate source node
                 that has a degree above k, and made future friends above delta, then becomes a source node
                 for training.
        """
        
        self.G = G
        self.X = X
        self.k = k
        self.delta = delta
        self.num_features = self.X.shape[1] * 2
        self.Global_node_to_index = {}
        
        for i, n in enumerate(self.G.nodes()):
            self.Global_node_to_index[n] = i
        
        degrees = self.G.degree( self.G.nodes() )
        degree_nodes = set()
        for node in degrees:
            d = degrees[node]
            if d >= k:
                degree_nodes.add(node)
                
        #degree_nodes are the candidate source nodes
        
        source_nodes = set()
        self.source_nodes_data = defaultdict(dict)
        print 'Finding the source nodes...'
        
        for i, node in enumerate(degree_nodes):
            G = self.G.copy()
            neighbours = G.neighbors(node)
            kth = len(neighbours)/2
            future_nodes = set( neighbours[kth+1:] )
            for fnode in future_nodes:
                G.remove_edge(node, fnode)
            
            node_1_neiborhood = set(G.neighbors(node))
            node_2_neiborhood = graph_utils.node_neighborhood(G, node, 2)
            node_2_neiborhood = set(node_2_neiborhood)
            
            counter = 0
            D_nodes = set()
            for fnode in future_nodes:
                if fnode in node_2_neiborhood:
                    counter += 1
                    D_nodes.add(fnode)
            
            node_to_index = {}
            index_to_node = {}
            L_nodes = set()
            
            if counter >= delta:
                source_nodes.add(node)
                for n in G.nodes():
                    if (n not in node_1_neiborhood) and (n not in node_2_neiborhood) and (n != node):
                        G.remove_node(n)
                    else:
                        if G.degree(n) == 0:
                            raise Exception("a node has zero degree!!!")
                
                for i, n in enumerate(G.nodes()):
                    node_to_index[n] = i
                    index_to_node[i] = n
                    
                for n in G.nodes():
                    if n not in D_nodes:
                        L_nodes.add(n)
                        
                s_index = node_to_index[node]  
                self.source_nodes_data[node] = \
                    {'G': G, 'D': D_nodes, 'L': L_nodes, 'index': s_index,
                     'node_to_index': node_to_index, 'index_to_node': index_to_node}
                    
        split = len(self.source_nodes_data.keys())
        split = split/2    
        self.training_S = self.source_nodes_data.keys()[0:split]
        self.testing_S = self.source_nodes_data.keys()[split:]
        
    def get_feature_vec(self, s, i, j):
        """
        Returns the feature vector between two nodes i, j with 
        respect to a source node s.
        
        Parameters:
        -----------
        s: the source node ID or something.
        i: the node i index in the adjacency matrix of the source node graph.
        j: the node j index in the adjacency matrix of the source node graph.
        """
        
        index_to_node = self.source_nodes_data[s]['index_to_node']
        node1 = index_to_node[i]
        node2 = index_to_node[j]
        node1_global_index = self.Global_node_to_index[node1]
        node2_global_index = self.Global_node_to_index[node2]
        v = np.concatenate( (self.X[node1_global_index], self.X[node2_global_index]) )
        return v
        
    def num_feats(self):
        """
        Returns the number of features in-use.
        """
        
        return self.num_features
        
    def get_D(self, s):
        """
        Returns a list of nodes indices that are the destinations for the source node.
        
        Parameters:
        -----------
        s: the source node ID.
        
        Returns:
        D: a list that represents the nodes indices (with respect to the source node graph)
         that are the destinations for the source node
        """
        
        D = self.source_nodes_data[s]['D']
        node_to_index = self.source_nodes_data[s]['node_to_index']
        D_ind = []
        
        for d in D:
            D_ind.append( node_to_index[d] )
        
        return D_ind
        
    def get_L(self, s):
        """
        Returns a list of nodes indices that are not destinations for the source node.
        
        Parameters:
        -----------
        s: the source node ID.
        
        Returns:
        D: a list that represents the nodes indices (with respect to the source node graph)
         that are the destinations for the source node
        """
        L = self.source_nodes_data[s]['L']
        node_to_index = self.source_nodes_data[s]['node_to_index']
        L_ind = []
        
        for l in L:
            L_ind.append( node_to_index[l] )
        
        return L_ind
        
    def get_S(self):
        """
        Returns a list of all source nodes IDs to use as training source nodes.
        """
        return self.training_S
    
    def get_testingS(self):
        """
        Returns a list of all source nodes IDs to use as testing source nodes.
        """
        return self.testing_S
        
    def get_A(self, s):
        """
        Returns the adjacency matrix of a source node s.
        It must always return a numpy array and NOOOT a matrix!!!
        Parameters:
        -----------
        s: the source node ID.
        """
        G = self.source_nodes_data[s]['G']
        return np.array(nx.adj_matrix(G))
        
        
    def get_s_index(self, s):
        """
        Returns the index of the source node with respect to its adjacency matrix.
        
        Parameters:
        -----------
        s: the source node ID
        
        Returns:
        s_index: the source node index in the adjacency matrix.
        """
        
        return self.source_nodes_data[s]['index']
    
    
    
def get_y_test(s, psi):
    """
    Parameters:
    s: the source id
    psi: the psi
    """
    D = psi.get_D(s)
    G = psi.source_nodes_data[s]['G']
    N = G.number_of_nodes()
    y_test = np.zeros((N))
    y_test[D] = 1
    
    return y_test
    
    

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    