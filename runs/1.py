import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from rakg import testing_utils as tstu
import rakg.facebook100_parser as fb_parser
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def main():
    pass
    args = sys.argv
    choice = int(args[1])
    
    file_path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
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
     

    
    if choice == 1:
        logistic(X1, Y1, X2, Y2, X3, Y3, random_state)
    elif choice == 2:
        random_forest(X1, Y1, X2, Y2, X3, Y3, random_state)
    elif choice == 3:
        rbm_logistic1(X1, Y1, X2, Y2, X3, Y3, random_state)
    elif choice == 4:
        rbm_logistic2(X1, Y1, X2, Y2, X3, Y3, random_state)
        
def logistic(X1, Y1, X2, Y2, X3, Y3, random_state):
    logistic = linear_model.LogisticRegression()
    logistic_rocs = []
    roc1 = tstu.train_with_cross_val(X1, Y1, logistic, "logistic_topo", random_state )
    roc2 = tstu.train_with_cross_val(X2, Y2, logistic, "logistic_atts", random_state )
    roc3 = tstu.train_with_cross_val(X3, Y3, logistic, "logistic_topo_atts", random_state )
    
    logistic_rocs.append(roc1)
    logistic_rocs.append(roc2)
    logistic_rocs.append(roc3)
    tstu.draw_rocs(logistic_rocs, "logistic_rocs")
    
def random_forest(X1, Y1, X2, Y2, X3, Y3, random_state):
    random_forest = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    rocs = []
    roc1 = tstu.train_with_cross_val(X1, Y1, random_forest, "RF_topo", random_state )
    roc2 = tstu.train_with_cross_val(X2, Y2, random_forest, "RF_atts", random_state )
    roc3 = tstu.train_with_cross_val(X3, Y3, random_forest, "RF_topo_atts", random_state )
    
    rocs.append(roc1)
    rocs.append(roc2)
    rocs.append(roc3)
    
    tstu.draw_rocs(rocs, "RF_rocs_10trees")
    
    random_forest.n_estimators=100
    rocs = []
    roc1 = tstu.train_with_cross_val(X1, Y1, random_forest, "RF_topo", random_state )
    roc2 = tstu.train_with_cross_val(X2, Y2, random_forest, "RF_atts", random_state )
    roc3 = tstu.train_with_cross_val(X3, Y3, random_forest, "RF_topo_atts", random_state )
    
    rocs.append(roc1)
    rocs.append(roc2)
    rocs.append(roc3)
    
    tstu.draw_rocs(rocs, "RF_rocs_100trees")
    
    
def rbm_logistic1(X1, Y1, X2, Y2, X3, Y3, random_state):
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=random_state, verbose=True)
    rbm_logistic_clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]) 
    
    rocs = []
    roc1 = tstu.train_with_cross_val(X1, Y1, rbm_logistic_clf, "rbm_logistic_topo", random_state )
    roc2 = tstu.train_with_cross_val(X2, Y2, rbm_logistic_clf, "rbm_logistic_atts", random_state )
    roc3 = tstu.train_with_cross_val(X3, Y3, rbm_logistic_clf, "rbm_logistic_topo_atts", random_state )
    
    rocs.append(roc1)
    rocs.append(roc2)
    rocs.append(roc3)
    
    tstu.draw_rocs(rocs, "rbm_logistic_default")
    
def rbm_logistic2(X1, Y1, X2, Y2, X3, Y3, random_state):
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=random_state, verbose=True, n_iter=50)
    rbm_logistic_clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]) 
    
    rocs = []
    roc1 = tstu.train_with_cross_val(X1, Y1, rbm_logistic_clf, "rbm_logistic_topo", random_state )
    roc2 = tstu.train_with_cross_val(X2, Y2, rbm_logistic_clf, "rbm_logistic_atts", random_state )
    roc3 = tstu.train_with_cross_val(X3, Y3, rbm_logistic_clf, "rbm_logistic_topo_atts", random_state )
    
    rocs.append(roc1)
    rocs.append(roc2)
    rocs.append(roc3)
    
    tstu.draw_rocs(rocs, "rbm_logistic_50iters")
    
main()


