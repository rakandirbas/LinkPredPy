"""
This module contains functions that calculate similarity scores for each node in the network
 (i.e the un-supervised link prediction methods)
"""

import numpy as np
import numpy.linalg as LA
from scipy.sparse import linalg as sLA 
from scipy import sparse
import scipy
from scipy.sparse.linalg import svds
import networkx as nx

def get_degrees_list(G, degrees = None):
    """
    Returns a list that contains the degree of each node.
    
    Parameters:
    -----------
    G: networkx graph object.
    degrees: a dictionary that maps from a node name to its degree. If omited, the
            this G.degree() will be used.
    """
    
    degrees_list = list()
    if degrees == None:
        for node in G.nodes():
            d = G.degree(node)
            degrees_list.append(d)
    else:
        for node in G.nodes():
            d = degrees[node]
            degrees_list.append(d)
    return degrees_list

def cn(A):
    M = sparse.csr_matrix(A)
    paths = M**2
    paths = paths.toarray()
    S = paths
    return S

def salton(A, degrees):
    N = A.shape[0]
    paths = cn(A)
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            i_degree = degrees[i] 
            j_degree = degrees[j] 
            factor = i_degree * j_degree
            if factor != 0:
                S[i,j] = (1.0/np.sqrt(factor)) * paths[i,j]
    return S

def jacard(A, degrees):
    N = A.shape[0]
    paths = cn(A)
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            if j != i:
                i_degree = degrees[i] 
                j_degree = degrees[j] 
                factor = i_degree + j_degree - paths[i,j]
                if factor != 0:
                    S[i,j] = (1.0/factor) * paths[i,j]
                
    return S

def sorensen(A, degrees):
    N = A.shape[0]
    paths = cn(A)
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            i_degree = degrees[i] 
            j_degree = degrees[j] 
            factor = i_degree + j_degree
            if factor != 0:
                S[i,j] = (2.0/factor) * paths[i,j]
                
    return S

def hpi(A, degrees):
    N = A.shape[0]
    paths = cn(A)
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            i_degree = degrees[i] 
            j_degree = degrees[j] 
            factor = np.min((i_degree, j_degree))
            if factor != 0:
                S[i,j] = (2.0/factor) * paths[i,j]
                
    return S

def hdi(A, degrees):
    N = A.shape[0]
    paths = cn(A)
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            i_degree = degrees[i]
            j_degree = degrees[j]
            factor = np.max((i_degree, j_degree))
            if factor != 0:
                S[i,j] = (2.0/factor) * paths[i,j]
                
    return S

def lhn1(A, degrees):
    N = A.shape[0]
    paths = cn(A)
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            i_degree = degrees[i] 
            j_degree = degrees[j] 
            factor = i_degree * j_degree
            if factor != 0:
                S[i,j] = (1.0/factor) * paths[i,j]
                
    return S

def pa(A, degrees):
    N = A.shape[0]
    S = np.zeros((N,N))
    
    for i in xrange(N):
        for j in xrange(N):
            i_degree = degrees[i] 
            j_degree = degrees[j] 
            factor = i_degree * j_degree
            S[i,j] = factor
                
    return S

def katz(A, h=0.1):
    """
        Calculates the Katz index.
        
        Parameters:
        graph: the graph.
        h: lambda, the normalization factor that must be 0 < h < 1
    """
    N = A.shape[0]
    I = np.identity(N)
    to_inv = (I - h*A)
    S = np.linalg.inv(to_inv) - I
    return S

def katz_h(A):
    """"
        Computes the reciprocal of the largest eigenvalue of the adjacency matrix of the graph.
        Theiprocal should be decreased so the reduced value can be used as the h parameter
        to compute the katz index.
    """
    A = sparse.csr_matrix(A)
    eigvals = sLA.eigsh(A, which='LM', return_eigenvectors=False)
    h = np.sort(eigvals)[-1]
    h = 1.0/h
    return h

def aa(A, degrees):
    """
    Parameters:
        A: numpy 2D adjacency array of the graph.
        degrees: a python list that contains the 
                 degree of each node index.
                 
    Returns: S the similarity matrix.
    """
    N = A.shape[0]
    #D is the degree matrix where D_(i,i) is the degree
    # of node i and zero elsewhere.
    D = np.zeros((N,N)) 
    for i in xrange(N):
        deg = degrees[i]
        if deg > 1:
            D[i,i] = 1.0/np.log10(deg)
            
    S = np.dot(A, D)
    S = np.dot(S, A)
    return S

def ra(A, degrees):
    """
    Parameters:
        A: numpy 2D adjacency array of the graph.
        degrees: a python list that contains the 
                 degree of each node index.
                 
    Returns: S the similarity matrix.
    """
    N = A.shape[0]
    # the degree matrix where D_(i,i) is the degree of node i and zero elsewhere.
    D = np.zeros((N,N)) 
    for i in xrange(N):
        deg = degrees[i]
        if deg > 1:
            D[i,i] = 1.0/deg
            
    S = np.dot(A, D)
    S = np.dot(S, A)
    return S

def lp(A, h=0.1):
    """
        Calculates the Local Path Index.
        
        Parameters:
        h: lambda, the katz normalization factor.
        
    """ 
    M = sparse.csr_matrix(A)
    paths_len2 = M**2
    paths_len3 = M**3
    S = paths_len2 + (h * paths_len3)
    S = S.toarray()
    
    return S

def page_rank(A, alpha, n_iters):
    """
    Calculates PageRank for the nodes.
    
    Parameters:
    ------------
    A: numpy adjacency matrix.
    alpha: teleport probability
    n_iters: number of iterations
    """
    
    N = A.shape[0]
    D = np.sum(A, axis=1)
    D = D.reshape(-1, 1)
    
    zeros = np.where(D==0)[0]
    
    for i in zeros:
        A[i] = A[i] + (1.0/N)
        
    A = np.where(D != 0, A/D, A)
    
    A = A * (1-alpha)
    
    for i in zeros:
        A[i] = A[i] / (1-alpha) #this to undo the effect of the previous line 
     
    A = A + (alpha/N)
    
    ####
    for i in zeros:
        A[i] = A[i] - (alpha/N)
    ####
     
    P = A

    P = LA.matrix_power(P, n_iters)
    
    #Or classical power iteration
#     r = np.zeros((1, N))
#     r[0,0] = 1
#     for i in xrange(n_iters):
#         r = np.dot(r, P)
    
    return P[0]
    

    
def rwr(A, c):
    """
    Calculates Random-Walk with Restart
    
    Parameters:
    -----------
    A: adjacency matrix.
    c: the probability to move to neighbour. (1-c) is the probability
        to return to the initial starting node.
        
    
    Returns:
    --------
    S: the similarity matrix. Each row represents the probability that the walker
        is at a given node/column given starting at the node whose index is the index
        of the row.
    """
    
    N = A.shape[0]

    I = np.identity(N)
    S = np.zeros((N,N))
    degrees = np.sum(A, axis=1)
    degrees = degrees.reshape(-1, 1)
    P = np.where(degrees != 0, A/degrees, 0)
    
    inv = (I - c * P.T)
    inv = LA.inv(inv)
    q_x = (1-c) * inv
    
    for i in xrange(N):
        e_x = np.zeros((N,1))
        e_x[i,0] = 1
        q = np.dot(q_x, e_x)
        S[i] = q.reshape(-1)

    return S

def lrw(A, t):
    """
    Computes Local Random Walk
    
    Parameters:
    -----------
    A: the adjacency matrix
    t: the number of time steps to run the walker
    
    Returns:
    ---------
    S: the similarity matrix
    """
    
    N = A.shape[0]
    S = np.zeros((N,N))
    degrees = np.sum(A, axis=1)
    degrees = degrees.reshape(-1, 1)
    P = np.where(degrees != 0, A/degrees, 0)
    
    for i in xrange(N):
        e_x = np.zeros((N,1))
        e_x[i,0] = 1
        pi_x = e_x
        
        for j in xrange(t):
            pi_x = np.dot(P.T, pi_x)
            
        S[i] = pi_x.reshape(-1)
    
    return S

class RWR_Clf:
    """
    A random walk with restart scorer
    """
    def __init__(self, A, c):
        """
        Parameters:
        -----------
        A: adjacency matrix.
        c: the probability to move to neighbour. (1-c) is the probability
            to return to the initial starting node. (1-c) is the restart prob.
        """
        self.A = A
        self.c = c
        
        self.S = rwr(A, c)
        
    def score(self, U_p, node_to_index=None):
        """
        Gives a RWR score for each dyad in the U_prope test set.
        
        Parameters:
        -----------
        U_p: a list containing tuples of the form (nodei_index, nodej_index)
        node_to_index: a dictionary that maps from a node name to a node index
        
        Returns:
        p: a list that contains the scores for each tuple in U_p
        """
        p = []
        for (i, j) in U_p:
            if node_to_index != None:
                i = node_to_index[i]
                j = node_to_index[j]
            
            s_ij = self.S[i,j]
            s_ji = self.S[j,i]
            p.append(s_ij + s_ji)
            
        p = np.array(p)
            
        return p
    
class LRW_Clf:
    """
    A local random walk scorer
    """
    def __init__(self, A, t, n_edges):
        """
        A: adjacency matrix
        t: number of time steps for the walker
        n_edges: total number of edges in the graph
        """
        self.A = A
        self.t = t
        self.S = lrw(A,t)
        self.degrees = np.sum(A, axis=1)
        self.n_edges = n_edges
        
    def score(self, U_p, node_to_index):
        """
        Gives a LRW score for each dyad in the U_prope test set.
        
        Parameters:
        -----------
        U_p: a list containing tuples of the form (nodei_index, nodej_index)
        node_to_index: a dictionary that maps from a node name to a node index
        
        Returns:
        p: a list that contains the scores for each tuple in U_p
        """
        p = []
        for (i, j) in U_p:
            i = node_to_index[i]
            j = node_to_index[j]
            s_ij = self.S[i,j]
            s_ji = self.S[j,i]
            q_i = self.degrees[i]/self.n_edges #the initial configuration function
            q_j = self.degrees[j]/self.n_edges #the initial configuration function
            s = (q_i * s_ij) + (q_j * s_ji)
            p.append(s)
            
        p = np.array(p)
        return p
    

class SRW_Clf:
    """
    A superposed random walk scorer
    """
    def __init__(self, A, t, n_edges):
        """
        A: adjacency matrix
        t: number of time steps for the walker
        n_edges: total number of edges in the graph
        """
        self.A = A
        self.t = t
#         self.S = lrw(A,t)
        self.degrees = np.sum(A, axis=1)
        self.n_edges = n_edges
        
    def score(self, U_p, node_to_index):
        """
        Gives a SRW score for each dyad in the U_prope test set.
        
        Parameters:
        -----------
        U_p: a list containing tuples of the form (nodei_index, nodej_index)
        
        Returns:
        Sc: a list that contains the scores for each tuple in U_p
        """
        Sc = np.zeros(( len(U_p) ))
        for x in xrange(self.t):
            S = lrw(self.A,x+1)
            p = []
            for (i, j) in U_p:
                i = node_to_index[i]
                j = node_to_index[j]
                s_ij = S[i,j]
                s_ji = S[j,i]
                q_i = self.degrees[i]/self.n_edges #the initial configuration function
                q_j = self.degrees[j]/self.n_edges #the initial configuration function
                s = (q_i * s_ij) + (q_j * s_ji)
                p.append(s)
                
            Sc = Sc + p
        return Sc
        
def predict_scores(U, S, node_to_index):
    """
    Returns the prediction score for each dyad in U using the similarity matrix S.
    
    Parameters:
    -----------
    U: a list of dyads of the form [(node_i, node_j), ... etc] where node_i is the node's name.
    S: the similarity matrix that contains the similarity score between each node.
    node_to_index: a dictionary that maps from a node's name to a node's index.
    
    Returns:
    -----------
    probs: returns the confidence score of being a link for each dyad in U. Note that
            it is a confidence score, not a probability score! i.e. the score could be
            larger than 1 e.g. common neighbours!
    """
    
    probs = np.zeros( len(U) )
    
    for c, (node1, node2) in enumerate(U):
        i = node_to_index[node1]
        j = node_to_index[node2]
        probs[c] = S[i,j]
        
    return probs

def test_pagerank():
#     A = [ [0,1,0,0,0,0], 
#           [0,0,1,1,0,0],
#           [1,0,0,1,1,0],
#           [0,0,0,0,1,0],
#           [0,0,0,0,0,0],
#           [0,0,0,0,1,0] 
#           ]
     
    A = [ [0,1,0],
          [1,0,1],
          [0,1,0]
         ]

    A = np.array(A, dtype='float')
    
    print page_rank(A, 0.5, 40)
    

def test_rwr():
    print 'RWR!'
    A = [ [0,1,0],
          [1,0,1],
          [0,1,0]
         ]
    A = np.array(A, dtype='float')
    print rwr(A, 0.5)
    
def test_rwr_clf():
    A = [ [0,1,0],
          [1,0,1],
          [0,1,0]
         ]
    A = np.array(A, dtype='float')
    print rwr(A, 0.5), "\n\n"
    clf = RWR_Clf(A, 0.5)
    U_p = ((0,1), (0,2))
    print clf.score(U_p)
    
def tet_srw_clf():
    A = [ [0,1,0],
          [1,0,1],
          [0,1,0]
         ]
    A = np.array(A, dtype='float')
    print lrw(A, 1), '\n'
    
    clf = SRW_Clf(A, t=14, n_edges=2)
    U_p = ((0,1), (0,2))
    print clf.score(U_p)






