"""
This is just an example that shows how to optimize a function with numpy.
"""

import numpy as np
import networkx as nx
from scipy import optimize


def f(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f

def gradf(x, *args):
    u, v = x
    a, b, c, d, e, f = args
    gu = 2*a*u + b*v + d
    gv = b*u + 2*c*v + e
    gr = np.asarray((gu, gv))
    
    return gr

args = (2, 3, 7, 8, 9, 10)
x0 = np.asarray((0,0))

res = optimize.fmin_cg(f, x0, fprime=gradf, args=args)

def multM(U, V):
    R = np.zeros(( U.shape[0], V.shape[1] ))
    
    dim = U.shape[1]
    
    for i in range(U.shape[0]):
        for j in range(V.shape[1]):
            for t in xrange(dim):
                R[i,j] += U[i, t] * V[t, j]
    
    return R
            
U = np.random.randint(10,size=(3,4))
V = np.random.randint(10,size=(4,2))

R = multM(U,V)

print 'R=', R, '\n\n'

print 'R=', np.dot(U,V)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    