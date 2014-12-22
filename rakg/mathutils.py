'''
Created on May 14, 2014

@author: rockyrock

Math helper functions
'''

import numpy as np
from numpy import linalg as LA
from scipy.special import expit

def logistic_function(x):
#     x = np.float64(x)
    return 1.0 / (1.0 + np.exp(-x))
#     return expit(x)
#     return .5 * (1 + np.tanh(.5 * x))


def square_loss(x):
    return x**2

def log_loss(x):
    return -np.log(x)

def identity_link(x):
    return x

def norm(x):
    return LA.norm(x)

def descent_step(W, W_grad, alpha):
    W = W - (alpha * W_grad)
    return W

def sigmoid(X):
    s = []
    
    for x in X:
        s.append( logistic_function(x) )
        
    s = np.array(s)
    return s

def wmw(x,b=1):
#     if x > 0:
    return ( 1.0/( 1 + np.exp( -x/b ) ) )
#     else:
#         return 0

def derv_wmw(x,b=1):
#     if x > 0:
    return ( ( np.exp( -x/b ) ) / ( b * ( 1 + np.exp( -x/b ) )**2 ) )
#     else:
#         return 0


def pderv_logistic(dot_prod, x_i):
    """
    This computes the partial derivative of the logistic function with respect to one 
    parameter w.
    
    dot_prod: is the dot product result between the parameter w vector and x vector of the features.
    x_i: is the feature value that is multiplied with the parameter w_i
    """
    return ( ( np.exp(-dot_prod) * x_i ) / ( 1.0 + np.exp(-dot_prod) )**2 )



# print wmw(-0.9)




