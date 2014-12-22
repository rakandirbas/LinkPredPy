'''
Created on Aug 15, 2014

This module contains the supervised methods (i.e. learners) used with link prediction. 

@author: rockyrock
'''
from __future__ import division
import testing_utils as tstu
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import light_srw, srw
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matrix_fact



def general_learner(clf, X, Y, X_pp, Y_pp, X_pp_flag, test_name, random_state=0):
    """
    Train a random forest classifier.
    
    Parameters:
    ------------
    clf: a sci-kit classifier object.
    X: training dataset.
    Y: labels. (binary)
    test_name: the string that will be used next to the roc cure in the plot.
    """
    all_curves = tstu.train_with_cross_val_updated(X, Y, X_pp, Y_pp, X_pp_flag, clf, test_name, random_state )
    
    return all_curves



#################
"""
A wrapper for supervised random walk
"""

def train_with_srw(G, X, test_name,  k=10, delta=5, alpha=0.5, iter=1):
    """
    Applies the Supervised Random Walk method
    
    Parameters:
    ------------
    G: networkx graph.
    X: node attributes matrix. Each row i is the features vector for node i.
    test_name: the name to be plotted next to the curve.
    k: the number of neighbours a node must have in order to be considered a candidate source node.
    delta: the number of edges the candidate source node made in the future that close a triangle. 
           (i.e. the future/destination node is a friend of a friend, so delta is a threshold that sees how many
             of these future/destination nodes are friends of current friends). A candidate source node
             that has a degree above k, and made future friends above delta, then becomes a source node
             for training.
    alpha: restart probability.
    iter: gradient desecent epochs.
    
    Returns:
    --------
    roc: a roc curve details to be plotted.
    typical_auc: an AUC calculated using the same way of calculating the AUC with cross-validation
    my_auc: an AUC calculated using the mean auc of all the testing source nodes. 
    """
    psi = light_srw.GeneralPSI(G=G, X=X, k=k, delta=delta)
    srw_obj = light_srw.SRW(psi=psi, alpha=alpha)
    srw_obj.optimize(n_iter=iter)
    
    aucs = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    all_aucs_pr = []
    
    all_y_test = []
    all_props = []
    all_prec = {}
    all_rec = {}
    all_aucs_pr_d = {}
    i = 0
    
    mean_recall = mean_prec = 0.0
    
    for s in psi.get_testingS():
        P = srw_obj.get_P(s)
        s_index = psi.get_s_index(s)
        probs = srw.rwr(P, s_index, alpha)
        y_test = light_srw.get_y_test(s, psi)
        
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
        auc_ = tstu.compute_AUC(y_test, probs)
        aucs.append(auc_)
        
        precision, recall, average_precision = tstu.get_PR_curve_values(y_test, probs)
        all_prec[i] =  precision
        all_rec[i] = recall
        all_aucs_pr_d[i] = average_precision
        all_aucs_pr.append( average_precision )
        
        all_y_test.extend(y_test)
        all_props.extend(probs)
        
        i += 1
        
    mean_tpr /= len(psi.get_testingS())
    mean_tpr[-1] = 1.0
    mean_auc1 = auc(mean_fpr, mean_tpr)
    
    roc = (test_name, mean_fpr, mean_tpr, mean_auc1)
    
    aucs = np.array(aucs)
    mean_auc2 = np.mean(aucs)
    
    all_aucs_pr = np.array(all_aucs_pr)
    mean_prec, mean_recall, _ = tstu.get_PR_curve_values(all_y_test, all_props)
    mean_pr_auc = np.mean(all_aucs_pr)
    all_prec['mean'] = mean_prec
    all_rec['mean'] = mean_recall
    all_aucs_pr_d['mean'] = mean_pr_auc
    all_y_test = np.array(all_y_test)
    random_prec = all_y_test[all_y_test.nonzero()].size / all_y_test.size
    
    print "Cross-validation ROC stats: Testname: %s,\
     STD: %f, Variance: %f.\n" % (test_name, np.std(aucs), np.var(aucs))
     
    print "Cross-validation PR auc stats: Testname: %s,\
     STD: %f, Variance: %f.\n" % (test_name, np.std(all_aucs_pr), np.var(all_aucs_pr)) 
     
    the_pr_curve = (test_name, mean_prec, mean_recall, mean_pr_auc)
    all_curves = {"roc":roc, "pr": the_pr_curve, "all_prec": all_prec, "all_rec": all_rec, "all_auc_pr": all_aucs_pr_d, "n_folds": len(psi.get_testingS()), 'random':random_prec} 
    
    return all_curves, mean_auc1, mean_auc2




################
"""
A wrapper for matrix factorization
"""

def train_matrix_fact_normal(G, X, test_name, options):
    """
    Train the normal matrix fact model.
    
    Parameters:
    ------------
    G: networkx object.
    X: node attributes matrix.
    test_name: the name of the roc curve.
    options: a dictionary that holds the following parameters:
            options['mf_n_latent_feats'], options['mf_n_folds'], 
            options['mf_alpha'], options['mf_n_iter'], options['mf_with_sampling']
            
    Returns:
    --------
    all_curves = {"roc":roc_curv, "pr": pr_curv} 
        where a roc curve contains details of the roc to be plotted.
    """

    k = options['mf_n_latent_feats']
    n_folds = options['mf_n_folds']
    alpha = options['mf_alpha']
    n_iter = options['mf_n_iter']
    with_sampling = options['mf_with_sampling']
    
    MF = matrix_fact.Matrix_Factorization(k = k, G=G, X=X )

    roc_data, pr_data, n_folds_curves = MF.train_test_normal_model(n_folds = n_folds, 
                    alpha=alpha, n_iter=n_iter, with_sampling = with_sampling)
    fpr, tpr, auc = roc_data
    mean_prec, mean_recall, mean_pr_auc = pr_data
    roc_curv = (test_name, fpr, tpr, auc)
    pr_curv = (test_name, mean_prec, mean_recall, mean_pr_auc)
    all_curves = {"roc":roc_curv, "pr": pr_curv, "n_folds_curves": n_folds_curves}
    
    return all_curves


def train_matrix_fact_ranking(G, X, test_name, options):
    """
    Train the ranking matrix fact model.
    
    Parameters:
    ------------
    G: networkx object.
    X: node attributes matrix.
    test_name: the name of the roc curve.
    options: a dictionary that holds the following parameters:
            options['mf_n_latent_feats'], options['mf_n_folds'], 
            options['mf_alpha'], options['mf_n_iter']
            
    Returns:
    --------
    roc: a roc curve details to be plotted.
    """

    k = options['mf_n_latent_feats']
    n_folds = options['mf_n_folds']
    alpha = options['mf_alpha']
    n_iter = options['mf_n_iter']
    
    MF = matrix_fact.Matrix_Factorization(k = k, G=G, X=X )

    roc_data, pr_data, n_folds_curves = MF.train_test_ranking_model(n_folds = n_folds, 
                    alpha = alpha, n_iter = n_iter)

    fpr, tpr, auc = roc_data
    mean_prec, mean_recall, mean_pr_auc = pr_data
    roc_curv = (test_name, fpr, tpr, auc)
    pr_curv = (test_name, mean_prec, mean_recall, mean_pr_auc)
    all_curves = {"roc":roc_curv, "pr": pr_curv, "n_folds_curves": n_folds_curves}
    
    return all_curves













