import numpy as np
import mathutils as mu
from numpy import linalg as LA
import graph_utils
from graph_utils import DyadFeaturesIndexLocator
import graphsim as gsim
import networkx as nx
import srw as srwM
from srw import SRW
import testing_utils
from timer import Timerx

t = Timerx(True)

def main():
    print "Hi"
#     test_basic()
    synthetic_test()
    

def synthetic_test():
    print "started...."
    N, s, alpha, k = 10000, 0, 0.2, 1000
    synData = SynthData(n_nodes=N, n_feats=2)
#     A = synData.get_A()
#     A = np.array(A)
#     w = np.array([1,-1])
#     P = synData.get_P(0, alpha, w)
#     scores, indices = get_top_k(P, s, k, alpha)
#     D_pure = get_D(s, indices, A)
#     D = indices.copy()
#     DL = get_DL(N, D)
#     T = {s: DL}
#     y_test = get_y_test(N, D)
#     probs = scores
#     
#     auc = testing_utils.compute_AUC(y_test, probs)
# #     print "AUC=", auc
# #     print "D:\n", D
# #     print "D_pure:\n", D_pure
# #     print "Top k elements scores:\n", scores[D]
#     
#     srw_ob = SRW( synData.psi, T, A, alpha)
#     print "W:", srw_ob.w, "\n\n"
#     print "Optimizing..."
#     srw_ob.optimize(n_iter=100, lmbda=1, eps=0.1, verbose=False)
#     print "Optimized W:", srw_ob.w, "\n\n"
#     
#     P = srw_ob.get_P(s)
#     probs = srwM.rwr(P, s, alpha)
#     auc = testing_utils.compute_AUC(y_test, probs)
#     print "SRW AUC=", auc

def test_basic():
    A = [ [0,1,0],
          [1,0,1],
          [0,1,0]
         ]
    A = np.array(A, dtype='float')
    G = nx.Graph(A)
    
#     print gsim.rwr(A, 0.5)
#     print nx.pagerank(G, 0.5)
    
    psi = PSI(3, 2)
    T = {}
    T[0] = [[2],[1]]
    srw = SRW(psi, T, A, 0.5)
    
    srw.optimize(12)
    print srwM.rwr(srw.get_P(0), 0, 0.5)
#     print srwM.rwr(srw.get_P(1), 1, 0.5)
#     print srwM.rwr(srw.get_P(2), 2, 0.5)

class PSI:
    def __init__(self, n_row, n_cols):
        self.X = np.random.random((n_row, n_cols))
        self.n_row = n_row
        self.n_cols = n_cols
    
    
    def get_feature_vec(self, node1_index, node2_index):
        """
        return the feature vector for the edge (node1, node2)
        """
        
        if ( (node1_index==0 and node2_index==1) or 
             (node1_index==1 and node2_index==0) ):
#             return self.X[0]
            return np.array([0.1, 0.2])
        elif ( (node1_index==0 and node2_index==2) or 
             (node1_index==2 and node2_index==0) ):
#             return self.X[1]
            return np.array([0.3, 0.4])
        elif ( (node1_index==1 and node2_index==2) or 
             (node1_index==2 and node2_index==1) ):
#             return self.X[2]
            return np.array([0.5, 0.6])
        else:
            raise Exception("NOT FOUND FEATURE VECTOR!")
        
    def num_feats(self):
        """
        returns the number of features
        """
        return self.n_cols

class SynthPSI:
    """
    This is a utility class that helps to return
    the feature vector between two nodes
    """
    def __init__(self, X, feature_index_locator):
        """
        Parameter:
        ----------
        X: the dyads feature matrix
        feature_index_locator: an object that implements get_feature_index(node1_index, node2_index) 
        which returns the index of the feature vector between two nodes.
        
        """
        self.X = X
        self.feature_index_locator = feature_index_locator
        
    def get_feature_vec(self, node1_index, node2_index):
        index = self.feature_index_locator.get_feature_index(node1_index, node2_index)
        return self.X[index]
    
    def num_feats(self):
        return self.X.shape[1]
    

class SynthData:
    def __init__(self, n_nodes, n_feats):
        self.G = nx.powerlaw_cluster_graph(n_nodes, 2, 0.5)
#         self.G = nx.fast_gnp_random_graph(n_nodes, 1)
        t.start()
        print "U:"
        self.U = graph_utils.build_U(self.G.nodes())
        t.stop()
        t.start()
        print "locator:"
        self.feature_index_locator = DyadFeaturesIndexLocator(self.U, self.G.nodes())
        t.stop()
        mu, sigma = 0, 1
        t.start()
        print "else:"
        self.M = np.random.normal(mu, sigma, (len(self.U),n_feats))
        self.psi = SynthPSI(self.M, self.feature_index_locator)
        self.n_feats = n_feats
        t.stop()
        
    def get_P(self, s, alpha, w):
        """
        s: source node index
        alpha: restart probability
        w: a weight vector
        """
        
        if len(w) != self.n_feats:
            raise Exception("The number of weights don't match the number of features!")
        
        P = nx.adj_matrix(self.G)
        P = np.array(P)
        N = P.shape[0]
        zero_rows = np.where(~P.any(axis=1))[0]
        
        if zero_rows.size > 0:
            raise Exception("The adjacency matrix has a zero row")
        
        for i in xrange(N):
            nonzero = np.nonzero(P[i])
            nonzero = nonzero[0]
            for j in nonzero:
                x = self.psi.get_feature_vec(i,j) 
                dotprod = np.dot(x, w)
                a = mu.logistic_function(dotprod)
                P[i,j] = a
                
        normalization = np.sum(P, axis=1)
        normalization = normalization.reshape(-1,1)
        P = P/normalization
        
        P = (1-alpha) * P
        P[:, s] += alpha
        
        return P
    
    def get_A(self):
        return np.array(nx.adj_matrix(self.G))
        
        
def get_top_k(P, s, k, alpha):
    """
    Computes random-walk-with-restart for the source node s,
    and then returns the scores for all the nodes and the indices of the
    top K nodes.
    
    Parameters:
    -----------
    P: the transition matrix.
    s: the index of the source node.
    k: the number of top elements to be returned
    
    Returns:
    --------
    scores: the rwr scores for all nodes.
    indices: the indices of the top k elements.
    """
    
    scores = srwM.rwr(P, s, alpha)
    indices = scores.argsort()[-k:][::-1]
    
    return scores, indices

def get_D(s, top_k_indices, A):
    """
    Returns a list of targets for node s. For that it uses the indices of the top_k nodes after
    running a random-walk with restart. It checks which of those nodes node s doesn't have 
    an edge with and then it adds that node as a target.
    
    Parameters:
    -----------
    s: the source node index
    top_k_indices: the indices of the top nodes after running a random-walk-with-restart
    A: the adjacency matrix
    """
    
    D = []
    
    for i in top_k_indices:
        if A[s, i] == 0:
            D.append(i)
            
    return D
        

def get_DL(N, D):
    """
    Returns a list of two lists [D,L] where D is a list that contains 
    the indices of the target nodes and L is a list that contains the indices of the
    non-links nodes.
    
    Parameters:
    -----------
    N: number of nodes in the graph
    D: a list that contains the indices of the target nodes
    
    Returns:
    --------
    DL: a list of two lists [D,L] where D is a list that contains 
    the indices of the target nodes and L is a list that contains the indices of the
    non-links nodes.
    """
    
    L = []
    
    for i in xrange(N):
        if i not in D:
            L.append(i)
            
    DL = [D, L]
    
    return DL

def get_y_test(N, D):
    """
    Given the D list of a source node s,
    this function returns a list where each element of the list
    has a 1 if the node with the corresponding element index 
    is a destination for the source node s.
    
    Parameters:
    -----------
    N: the number of nodes in the graph
    D: a list that contains the indices of the destinations/[future links] for node s
    """
    
    y_test = np.zeros((N))
    y_test[D] = 1
    
    return y_test

main()




















