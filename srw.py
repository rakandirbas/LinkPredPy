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

def main():
    pass
    A = [ [0,1,0],
          [1,0,1],
          [0,1,0]
         ]
    A = np.array(A, dtype='float')
    psi = PSI(3, 2)
    srw = SRW(psi, None, A, 0.5)
    
    print gsim.page_rank(A, 0.5, 40)
    
#     Q_tick = srw.get_Q_tick()
#     Q = srw.get_Q(Q_tick, s=1)
#     p = power_iter(Q)
#     pQ0 = srw.get_pQ(w_index=0)
    
#     print 'Normalizer:\n', srw.act_normalizer, "\n\n"
#     print 'Q_tick:\n', Q_tick, "\n\n"
#     print 'Q:\n', Q, "\n\n"
#     print "p:\n", p, "\n\n"
#     print "pQ0:\n", pQ0, "\n\n"
#     print computeP_PP(Q, pQ0)
    
def rwr(P, s, c, n_iter=0):
    """
    Computes random walk with restart given:
    P: the transition probability matrix
    s: the index of the starting node
    c: the probability of restart
    n_iter: if > 0, the the method compute RWR using an iterative version,
            otherwise it uses an inverse based method
    
    Returns
    -------
    scores: a vector that represent the probability that 
            the walker is at node i, given that he started from node s.
            
    """
    
    N = P.shape[0]
    q_x = np.zeros((N))
    e_x = np.zeros((N))
    q_x[s] = 1
    e_x[s] = c
    
    if n_iter > 0:
        for i in xrange(n_iter):
            q_x = (1-c) * np.dot(P.T, q_x) + e_x
    else:
        cc = 1-c
        I = np.identity(N)
        e_x[s] = 1
        inv = (I - cc * P.T)
        inv = LA.inv(inv)
        q_x = (1-cc) * inv
        q_x = np.dot(q_x, e_x)
    
    return q_x

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
            print 'FUUUUUUUUUCKKKK'
        
        pass
#         return self.X[node1_inde]
#         return np.random.random((4))
    
    def num_feats(self):
        """
        returns the number of features
        """
        return self.n_cols
    
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

def grad_iter(Q, pQ, p, pp=None, e=10**-12):
    """
    Computes the stationary partial derivative of the 
    page rank scores with respect to one parameter variable w.
    
    Q: the transition matrix
    pQ: the partial derivative of the transition matrix with respect
        to the parameter variable w.
    p: the stationary page rank scores.
    pp: the partial derivatives of the page rank scores with 
        respect to the parameter variable w. When None, then it's 
        initialized to zeros.
    e: the threshold error, as to when to stop power-iter
    
    Returns:
    --------
    pp: the partial derivative of the page rank scores with
        respect to a single parameter variable w.
    """
    N = Q.shape[0]
    if pp == None:
        pp = np.zeros((N))
        
    p = p.reshape((1,-1))
    pp = pp.reshape((1,-1))
        
    e_test = 1
    old_pp = pp
    
    #(e_test > e)
    #####
    count = 0
    stp = 40
    #####
    
    
    while(count < stp):
        print count
        count += 1
        print pp
        #Compute the new pp
        pp = np.dot(pp, Q) + np.dot(p, pQ)

        e_test = pp - old_pp
        e_test = np.abs(e_test).sum()
        old_pp = pp
        
    return pp.reshape(-1)
    
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
    
    def __init__(self, psi, T, A, alpha):
        """
        psi: (the greek chara) an object that implements a
         get_feature_vec(node1_index, node2_ondex) method, which 
         returns the feature vector of node1 and node2
        T: python dictionary that holds as keys the sources,
            and as values a tuple, where one tuple member is a list 
            of the destination nodes (D) and the other is a list
            of non-destination nodes (L).
        A: the adjacency matrix
        alpha: restart probability
        """
        self.psi = psi
        self.T = T
        self.A = A
#         self.w = np.random.random( ( psi.num_feats() ) )
        self.w = np.random.normal(0,1,(psi.num_feats()))
#         self.w = np.array([1.0,-1.0])
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
        
#         activations = mu.sigmoid( np.dot(self.X, self.w) )#for checking for calcs
        
        for i in xrange(N):
            nonzero = np.nonzero(Q[i])
            nonzero = nonzero[0]
            for j in nonzero:
                x = self.psi.get_feature_vec(i,j)
                a = self.f( np.dot(x, self.w) )
                Q[i,j] = a
        
#         self.unQ_tick = Q
        
        normalization = np.sum(Q, axis=1)
        self.act_normalizer = normalization
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
        
#         zero_rows = np.where(~Q.any(axis=1))[0]
#         if len(zero_rows) > 0:
#             raise Exception("The adjacency matrix has a zero row")
#         Q[zero_rows, s] = 1
         
        return Q
    
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
        
        for i in xrange(N):
            nonzero = np.nonzero(Q[i])
            nonzero = nonzero[0]
            
            sum_act = self.act_normalizer[i]
            sum_pf = 0 #sum of the partial dervs of activations for the row
            dot_prods = {}
            p_fs = {}
            
            for k in nonzero:
                x = self.psi.get_feature_vec(i,k)
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
                
                Q[i,j] = (1-self.alpha) * ( ( p_f * sum_act ) - (a*sum_pf) ) / (sum_act)**2
#                 print i, j, dotprod, p_f, sum_act, a, sum_pf, Q[i,j]
        
        return Q
    
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
        
        Q_tick = self.get_Q_tick()
        
        for w_index in xrange(num_w):
            w = self.w[w_index]
            pQ = self.get_pQ(w_index)
            sum_factor = 0
            
            for s in self.T:
                Q = self.get_Q( Q_tick , s)
                p, pp = computeP_PP(Q, pQ)
                D, L = self.T[s]
                
                for d in D:
                    for l in L:
                        delta = p[l] - p[d]
                        sum_factor += self.ph(delta) * ( pp[l] - pp[d] )
            
            pw[w_index] = (2 * w) +  (lmbda * sum_factor)
            
        return pw
    
    def compute_cost(self):
        """
        Computes the cost of the objective function.
        
        Returns:
        --------
        cost: a float that shows the cost
        """
        Q_tick = self.get_Q_tick()
        cost = 0
        for s in self.T:
            Q = self.get_Q( Q_tick , s)
            p = power_iter(Q)
            D, L = self.T[s]
            
            for d in D:
                for l in L:
                    delta = p[l] - p[d]
                    cost += self.h(delta)
                    
        cost += LA.norm(self.w)**2
        
        return cost
                
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
        """
        Returns the transition matrix P to be used for random walk with restart
        after optimizing the objective function.
        
        Parameters:
        s: the index of the source node
        """
        
        Q_tick = self.get_Q_tick()
        Q = self.get_Q(Q_tick, s)
        P = Q
        return P

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    