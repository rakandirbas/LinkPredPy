"""
Playing with pylearn2
"""

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from rakg import testing_utils as tstu
import rakg.facebook100_parser as fb_parser
import numpy as np
from pylearn2.config import yaml_parse
from rakg import pylearnutils
from pylearn2.models import softmax_regression
from pylearn2.training_algorithms import sgd
import pylearn2.termination_criteria as termination_criteria
from pylearn2 import train

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model

def main():
    args = sys.argv
    choice = int(args[1])
    file_path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    yamls_path = "./2_yamls/"
    SEED = 0
    random_state = SEED
    X1, Y1 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="A" )
    X2, Y2 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="B" )
    X3, Y3 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="C" )
    
     
    X1 = (X1 - np.min(X1, 0)) / (np.max(X1, 0) + 0.0001) # 0-1 scaling
    X2 = (X2 - np.min(X2, 0)) / (np.max(X2, 0) + 0.0001) # 0-1 scaling
    X3 = (X3 - np.min(X3, 0)) / (np.max(X3, 0) + 0.0001) # 0-1 scaling
    
    if choice == 0:
        pass
    elif choice == 1:
        _1(X1, Y1, random_state)
    elif choice == 2:
        _2(X1, Y1, random_state)

def _1(X1, Y1, random_state):
    """
    Trying pylearn2 with logistic regression and topology features only
    """
    model = softmax_regression.SoftmaxRegression(n_classes=2, irange=0.05, nvis=X1.shape[1])
    termination_criterion = termination_criteria.EpochCounter(max_epochs=100)
    algorithm = sgd.SGD(learning_rate=1e-1, batch_size=15, 
                                            termination_criterion=termination_criterion)
    trainOb = train.Train(dataset=None, model=model, algorithm=algorithm)
    
    clf = pylearnutils.PylearnClf(trainOb)
    rocs = []
    roc = tstu.train_with_cross_val(X1, Y1, clf, "logistic_topo", random_state )
    rocs.append(roc)
    tstu.draw_rocs(rocs, "2_1")
    
def _2(X1, Y1, random_state):
    """
    Scikit RBM with 5 hidden units + logistic + All default
    """
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=random_state, verbose=True)
    rbm.n_components = 5
    rbm_logistic_clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]) 
    
    rocs = []
    roc1 = tstu.train_with_cross_val(X1, Y1, rbm_logistic_clf, "rbm_logistic_topo", random_state )
    rocs.append(roc1)
    
    tstu.draw_rocs(rocs, "2_2")
    
def _3(X1, Y1, random_state):
    """
    Autostacked encoder with standard toplogy features
    Parameters: 
        
    """
    
    
    
main()



















