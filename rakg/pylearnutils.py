"""
This module provides wrapper around PyLearn2 to use it as SKlearn
"""

import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
import theano


def main():
    print 'hi\n'
    X = np.random.randint(10, size=(5,3))
    print X
    Y = np.random.randint(2, size=(5))
    print Y
    dataset = PylearnDataset(X, Y)
    
    print type(dataset)
    print dataset.X
    print dataset.y
    
#     serial.save("dataset.pkl", dataset)

#     t = serial.load("dataset.pkl")

class PylearnDataset(DenseDesignMatrix):
    
    def __init__(self, X, Y, preprocessor=None):
        """
        :type X: numpy array
        :type Y: numpy 1D array
        """
        Y = np.array(Y, dtype=int)
        classes = np.unique(Y)
        one_hot = np.zeros(( len(Y), classes.size ), dtype=np.int)
        for i in xrange(len(Y)):
            one_hot[i, Y[i]] = 1
        self.original_Y = Y
        super(PylearnDataset, self).__init__(X=X, y=one_hot, preprocessor=preprocessor)
        
class PylearnClf:
    
    def __init__(self, train):
        """
        :type train: Pylearn2 Train object
        :param train: a constructed Train object that has all the parameters initialized
        """
        self.train = train
        
    def fit(self, X, Y):
        dataset = PylearnDataset(X, Y)
        self.train.dataset = dataset
        self.train.main_loop()
        return self
        
    def predict_proba(self, X_test):
        X = self.train.model.get_input_space().make_theano_batch()
        Y = self.train.model.fprop(X)
        f = theano.function([X], Y)
        probs = f(X_test)
        return probs
        

def get_probs(X_test, model):
    """
    :type X_test: numpy array
    :param X_test: test set to classify
    :type model: Pylearn2 model
    :param model: a model to use it to predict the probabilities of the test instances 
    """
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    f = theano.function([X], Y)
    probs = f(X_test)
    return probs

# main()









