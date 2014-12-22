from __future__ import division
from graph_utils import *
from timer import *
import graphsim as gsim
import numpy as np
import networkx as nx
from itertools import izip
import sys
from memory_profiler import profile
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import cross_validation
import facebook100_parser as fb_parser
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib
matplotlib.use('Agg') 
import pylab as pl
import random
import sys
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import timer
from mpltools import style
style.use('ggplot')

def main():
    pass
    
def draw_rocs(values_list, file_name):
    """
    Draws and outputs a graph of ROC curves.
    
    Parameters
    ----------
    :type values_list: python list
    :param values_list: a list that holds the details of every roc curve to 
                        be painted. [(label1, fpr1, tpr1, roc_auc1),
                         (label2, fpr2, tpr2, roc_auc2), ...etc]
    
    :param file_name: the name of the PNG picture (you can prepend the path as well)
    """
    pl.clf()
    
    for label, fpr, tpr, roc_auc in values_list:
        pl.plot(fpr, tpr, lw=1, label='%s (auc = %0.4f)' % (label, roc_auc))
    
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.tight_layout()
    pl.savefig(file_name,dpi=72)
    
def draw_pr_curves(values_list, file_name):
    """
    Draws a list of Precision-Recall curves
    
    values_list: a list of curves data. Each element is a list of information needed
    to draw one curve. Each row should has: [label, precision, recall, avg_prec]
    """
    pl.clf()
    for label, precision, recall, avg_prec in values_list:
        pl.plot(recall, precision, label='%s (auc = %0.4f)' % (label, avg_prec))
    
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall curve')
    pl.legend(loc="lower right")
    pl.tight_layout()
    pl.savefig(file_name,dpi=72)
    
def draw_pr_curves_n_folds(n_folds, test_name, all_prec, all_rec, all_aucs_pr_d, random_prec, file_name):
    """
    Draws a list of Precision-Recall curves
    
    """
    pl.clf()
    pl.axhline(y=random_prec, c='blue', ls='--', label='random AUPR=%0.4f' % (random_prec)) #plot the random classifier line
    pl.plot(all_rec['mean'], all_prec['mean'], label='mean AUPR=%0.4f' % (all_aucs_pr_d['mean']), linewidth=2.0, color='r')
    
    for i in xrange(n_folds):
        pl.plot(all_rec[i], all_prec[i], color='k', alpha=0.3)
    
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall curve')
    pl.legend(loc="lower right")
    pl.tight_layout()
    pl.savefig(file_name,dpi=72)
    
def get_PR_curve_values(y_test, probs):
    precision, recall, _ =  precision_recall_curve(y_test, probs)
    average_precision = average_precision_score(y_test, probs)
    
    return precision, recall, average_precision

def compute_PR_auc(y_test, probs):
    return average_precision_score(y_test, probs)
    
def prepare_training_set(file_path, file_parser, features_adder, random_state,
                         edge_removal_perc=0.5, enabled_features="A", raw_features=False):

    random.seed(random_state)
    
    G, Nodes_X = file_parser.parse(file_path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = edge_removal_perc
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    
    if enabled_features == "A":
        data = features_adder(data, A, U, degrees, G.nodes(), raw_features)
    elif enabled_features == "B":
        data = file_parser.add_nodes_features(Nodes_X, data, U, G.nodes(), raw_features)
    elif enabled_features == "C":
        data = features_adder(data, A, U, degrees, G.nodes(), raw_features)
        data = file_parser.add_nodes_features(Nodes_X, data, U, G.nodes(), raw_features)
    else:
        sys.exit()
    
    X, y = shuffle(data, Y, random_state=random_state)
    
    return X, y

def build_training_datasets(G, X_nodes, node_to_index, params, 
                            edge_removal_perc=0.5, enabled_features=[[1]], 
                            undersample=False, extend_graph=False, random_state=0):
    """
    This updated version gives the possibility to split the datasets into:
    1- A dataset that holds the features for the dyads that are: negative-to-positive, negative-to-negative.
    2- A dataset that holds the features for the dyads that are: positive-to-positive.
    
    This helps when doing cross-validation. So using some dyads from dataset1 
    for learning and the rest for testing. Dataset2 can be either ignored, or always used for learning 
    combined with some dyads from dataset1.
    
    Parameters:
    -----------
    G: networkx graph object.
    X_nodes: the nodes attributes matrix.
    node_to_index: a dictionary that maps from node name to node index.
    params: a dictionary that holds needed parameters.
            Here we need the following: params['rwr_alpha'], 
            params['lrw_nSteps']. (you can omit this parameter if you don't need the global features)
    edge_removal_perc: how much edges to remove to use them for testing (highest is 1.0 and lowest 0.0).
    enabled_features: a list of lists. Each list represent a dataset. Each list contains ints 
        that control which features to add to the dataset in the given order.
        E.g. [ [1,2,3], [1,4] ] here we defined that we want two datasets. 
        The first dataset will have the features with the flags 1,2,3.
        The second dataset will have the features with the flag 1,4.
        
        The Flags are: (1-local topo features, 2-global topo features, 
            3-node attributes features, 4-raw features).
    undersample: to either undersample the training set to make the negative examples equal the number
            of positive examples or not.
    extend_graph: to either extend the graph with node that represent attributes or not.
                (i.e. to make a social-attribute network)
    
    Returns:
    --------
    Xs: a list of the datasets of the negative-to-positive dyads (the testing dyads) and negative-to-negative dyads.
    Ys: a list of the the labels for Xs.
    X_pps: a list of the datasets of the positive-to-positive dyads.
    Y_pps: a list of the labels for X_pps
    """
    random.seed(random_state)
    
    Gx, U, Y, pp_list, np_list, nn_list = prepare_graph_for_training(G, edge_removal_perc, undersample, random_state=random_state)
    
    
    ##
    if extend_graph:
        Gx, original_degrees = get_extended_graph(Gx, X_nodes)
        original_degrees_list = gsim.get_degrees_list(Gx, original_degrees)
    else:
        original_degrees_list = None
    ##
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    Xs = []
    Ys = []
    X_pps = []
    Y_pps = []
    Y_pp = np.ones( len(pp_list) )
    
    for dataset_config in enabled_features:
        X = None
        X_pp = None
        for feature_flag in dataset_config:
            if feature_flag == 1:#local topo features 
                X = add_local_topo_features(X, A, U, degrees, G.nodes(), original_degrees_list)
                X_pp = add_local_topo_features(X_pp, A, pp_list, degrees, G.nodes(), original_degrees_list)
            elif feature_flag == 2:# global topo features
                X = add_global_features(Gx, X, U, node_to_index, params)
                X_pp = add_global_features(Gx, X_pp, pp_list, node_to_index, params)
            elif feature_flag == 3:#node attributes
                X = add_raw_feature(X, U, X_nodes, G.nodes(), node_to_index=node_to_index)
                X_pp = add_raw_feature(X_pp, pp_list, X_nodes, G.nodes(), node_to_index=node_to_index)
            elif feature_flag == 4:#raw features
                X = add_raw_feature(X, U, A, G.nodes(), node_to_index=node_to_index)
                X_pp = add_raw_feature(X_pp, pp_list, A, G.nodes(), node_to_index=node_to_index)
        
        X, y = shuffle(X, Y, random_state=random_state)
        X_pp, y_pp = shuffle(X_pp, Y_pp, random_state=random_state)
        
        X, X_pp = scale_two_sets(X, X_pp)
        
        Xs.append(X)
        Ys.append(y)
        
        X_pps.append(X_pp)
        Y_pps.append(y_pp)
    
    return Xs, Ys, X_pps, Y_pps

def build_neighborhood_sets(G, X_nodes, node_to_index, params, 
                            edge_removal_perc=0.5, enabled_features=[1,2], random_state=0):
    """
    This method builds 3 training datasets that represent only NP and NN dyads. The
    three datasets are constructed based on geodesic distance. So the first datasets
    contains all NP and NN dyads that are within 2-hubs.
    
    Parameters:
    -----------
    enabled_features: here it's not a list of lists, but only a list that defines which features to add
                    to each neibourhood dataset. So [1,2] will add the local and global features
                    for each neibourhood dataset.
    params: a dictionary that holds needed parameters.
            Here we need the following: params['rwr_alpha'], 
            params['lrw_nSteps'].
                    
    Returns:
    Xs: a list of three datasets
    Ys: the labels for the three datasets.
    """
    random.seed(random_state)
    U = build_U(G.nodes())
    Gx = G.copy()
    
    num_edges_to_remove = int( edge_removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    U_n = build_Un_neighborhoods(Gx, U, [2,3,4])
    Y_2 = build_Y(G, U_n[2])
    Y_3 = build_Y(G, U_n[3])
    Y_4 = build_Y(G, U_n[4])
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    
    Xs = []
    Ys = []
    
    for i in sorted(U_n):
        X = None
        for feature_flag in enabled_features:
            if feature_flag == 1:#local topo features 
                X = add_local_topo_features(X, A, U_n[i], degrees, G.nodes())
            elif feature_flag == 2:# global topo features
                X = add_global_features(Gx, X, U_n[i], node_to_index, params)
            elif feature_flag == 3:#node attributes
                X = add_raw_feature(X, U_n[i], X_nodes, G.nodes(), node_to_index=node_to_index)
            elif feature_flag == 4:#raw features
                X = add_raw_feature(X, U_n[i], A, G.nodes(), node_to_index=node_to_index)
        
        X = scale(X)
        Xs.append(X)
    
    if np.count_nonzero(Y_2) != 0:
        Xs[0], y2 = shuffle(Xs[0], Y_2, random_state=random_state)
        Ys.append(y2)
    else:
        del Xs[0]
    
    if np.count_nonzero(Y_3) != 0:
        Xs[1], y3 = shuffle(Xs[1], Y_3, random_state=random_state)
        Ys.append(y3)
    else:
        del Xs[1]
    
    if np.count_nonzero(Y_4) != 0:
        Xs[2], y4 = shuffle(Xs[2], Y_4, random_state=random_state)
        Ys.append(y4)
    else:
        del Xs[2]
    
        
    return Xs, Ys

def prepare_neighborhood_sets(file_path, file_parser, features_adder, random_state=0,
                         edge_removal_perc=0.5, enabled_features="A", raw_features=False):
    """
    Prepares the neighborhood datasets.
    
    Returns:
    Xs: a list of training datasets. E.g. Xs[0] is the X dataset for 2-hub neighborhood.
    Ys: a list of training labels. E.g. Ys[0] is the Y labels for the 2-hub neighborhood X dataset.
    """
    
    random.seed(random_state)
    
    G, Nodes_X = file_parser.parse(file_path)
    U = build_U(G.nodes())
    
    Gx = G.copy()
    removal_perc = edge_removal_perc
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    U_n = build_Un_neighborhoods(Gx, U, [2,3,4])
    Y_2 = build_Y(G, U_n[2])
    Y_3 = build_Y(G, U_n[3])
    Y_4 = build_Y(G, U_n[4])
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    datas = {}
    
    for i in sorted(U_n):
        data = None
        
        if enabled_features == "A":
            data = features_adder(data, A, U_n[i], degrees, G.nodes(), raw_features)
        elif enabled_features == "B":
            data = file_parser.add_nodes_features(Nodes_X, data, U_n[i], G.nodes(), raw_features)
        elif enabled_features == "C":
            data = features_adder(data, A, U_n[i], degrees, G.nodes(), raw_features)
            data = file_parser.add_nodes_features(Nodes_X, data, U_n[i], G.nodes(), raw_features)
        else:
            sys.exit()
        
        datas[i] = data
        
            
    for key in datas:
        data = datas[key]
        min_max_scaler = preprocessing.MinMaxScaler()
        if data.shape[0] != 0:
            datas[key] = min_max_scaler.fit_transform(data)
 
    X2, y2 = shuffle(datas[2], Y_2, random_state=random_state)
    X3, y3 = shuffle(datas[3], Y_3, random_state=random_state)
    X4, y4 = shuffle(datas[4], Y_4, random_state=random_state)
    
    Xs = [X2, X3, X4]
    Ys = [y2, y3, y4]
    
    return Xs, Ys

def standard_features_adder(data, A, U, degrees, nodes, raw_features):
    if not raw_features:
        data = add_feature(data, U, gsim.cn(A), nodes)
        data = add_feature(data, U, gsim.lp(A), nodes)
        data = add_feature(data, U, gsim.salton(A, degrees), nodes)
        data = add_feature(data, U, gsim.jacard(A, degrees), nodes)
        data = add_feature(data, U, gsim.sorensen(A, degrees), nodes)
        data = add_feature(data, U, gsim.hpi(A, degrees), nodes)
        data = add_feature(data, U, gsim.hdi(A, degrees), nodes)
        data = add_feature(data, U, gsim.lhn1(A, degrees), nodes)
        data = add_feature(data, U, gsim.pa(A, degrees), nodes)
        data = add_feature(data, U, gsim.ra(A, degrees), nodes)
    else:
        data = add_raw_feature(data, U, A, nodes)
    return data

def train_with_cross_val(X, Y, clf, test_name, random_state, n_folds=10):
    kfold = cross_validation.KFold(len(X), n_folds=n_folds, random_state=random_state)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    all_aucs = []
    
    for train, test in kfold:
            probas_ = clf.fit(X[train], Y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            all_aucs.append( auc(fpr, tpr) )
            mean_tpr[0] = 0.0

    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    all_aucs = np.array(all_aucs)
    print "Cross-validation stats: Testname: %s,\
     STD: %f, Variance: %f.\n" % (test_name, np.std(all_aucs), np.var(all_aucs))
    
    return (test_name, mean_fpr, mean_tpr, mean_auc)


def train_with_cross_val_updated(X, Y, X_pp, Y_pp, X_pp_flag, clf, test_name, random_state, n_folds=10):
    """
    Trains a classifier with cross-validation. 
    
    This updated version gives the possibility to use the positive-to-positive dyads or not.
    If use_X_pp is true, then X_pp will be used for training only but not with testing.
    
    Parameters:
    -----------
    X_pp_flag: 0==don't use at all, 1==use for training only, 2-use for training and testing
    
    Returns:
    --------
    all_curves: {"roc":the_roc_curve, "pr": the_pr_curve}
    
    """
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    all_aucs = []
    all_aucs_pr = []
    
    all_y_test = []
    all_props = []
    all_prec = {}
    all_rec = {}
    all_aucs_pr_d = {}
    
    num_skipped = 0
    i = 0
    if X_pp_flag == 0:
        kfold = cross_validation.KFold(len(X), n_folds=n_folds, random_state=random_state)
        for train, test in kfold:
            X_train = X[train]
            y_train = Y[train]
            X_test = X[test]
            y_test = Y[test]
            
            if np.count_nonzero(y_train) == 0 or np.count_nonzero(y_test) == 0:
                print 'skipped one fold because:\n'
                print '#positive in y_train == ' + str(np.count_nonzero(y_train)) + '\n'
                print "#positive in y_test == " + str(np.count_nonzero(y_test)) + '\n'
                num_skipped += 1
                continue
            
            probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            all_aucs.append( auc(fpr, tpr) )
            mean_tpr[0] = 0.0
            
            precision, recall, average_precision = get_PR_curve_values(y_test, probas_[:, 1])
            all_prec[i] =  precision
            all_rec[i] = recall
            all_aucs_pr_d[i] = average_precision
            all_aucs_pr.append( average_precision )
            
            all_y_test.extend(y_test)
            all_props.extend(probas_[:, 1])
            
            i += 1
            
    elif X_pp_flag == 1:
        kfold = cross_validation.KFold(len(X), n_folds=n_folds, random_state=random_state)
        for train, test in kfold:
            X_train = np.vstack(( X_pp, X[train] ))
            y_train = np.concatenate(( Y_pp, Y[train] ))
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
            X_test = X[test]
            y_test = Y[test]
            probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            all_aucs.append( auc(fpr, tpr) )
            mean_tpr[0] = 0.0
            
            precision, recall, average_precision = get_PR_curve_values(y_test, probas_[:, 1])
            all_prec[i] =  precision
            all_rec[i] = recall
            all_aucs_pr_d[i] = average_precision
            all_aucs_pr.append( average_precision )
            
            all_y_test.extend(y_test)
            all_props.extend(probas_[:, 1])
            
            i += 1
            
    elif X_pp_flag == 2:
        X = np.vstack(( X_pp, X ))
        Y = np.concatenate(( Y_pp, Y ))
        X, Y = shuffle(X, Y, random_state=random_state)
        kfold = cross_validation.KFold(len(X), n_folds=n_folds, random_state=random_state)
        for train, test in kfold:
            X_train = X[train]
            y_train = Y[train]
            X_test = X[test]
            y_test = Y[test]
            probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            all_aucs.append( auc(fpr, tpr) )
            mean_tpr[0] = 0.0
            
            precision, recall, average_precision = get_PR_curve_values(y_test, probas_[:, 1])
            all_prec[i] =  precision
            all_rec[i] = recall
            all_aucs_pr_d[i] = average_precision
            all_aucs_pr.append( average_precision )
            
            all_y_test.extend(y_test)
            all_props.extend(probas_[:, 1])
            
            i += 1

    all_aucs = np.array(all_aucs)
    all_aucs_pr = np.array(all_aucs_pr)

    mean_tpr /= len(kfold) - num_skipped
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    mean_prec, mean_recall, _ =  get_PR_curve_values(all_y_test, all_props)
    mean_pr_auc = np.mean(all_aucs_pr)
    all_prec['mean'] = mean_prec
    all_rec['mean'] = mean_recall
    all_aucs_pr_d['mean'] = mean_pr_auc
    all_y_test = np.array(all_y_test)
    random_prec = all_y_test[all_y_test.nonzero()].size / all_y_test.size
    
    print "Cross-validation ROC auc stats: Testname: %s,\
     STD: %f, Variance: %f.\n" % (test_name, np.std(all_aucs), np.var(all_aucs))
     
    print "Cross-validation PR auc stats: Testname: %s,\
     STD: %f, Variance: %f.\n" % (test_name, np.std(all_aucs_pr), np.var(all_aucs_pr))
     
    the_roc_curve = (test_name, mean_fpr, mean_tpr, mean_auc)
    the_pr_curve = (test_name, mean_prec, mean_recall, mean_pr_auc)
    all_curves = {"roc":the_roc_curve, "pr": the_pr_curve, "all_prec": all_prec, "all_rec": all_rec, "all_auc_pr": all_aucs_pr_d, 'random': random_prec}
    
    return all_curves



def train_with_holdout(X, Y, clf, test_name, random_state, test_size=0.33):
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    
    return (test_name, fpr, tpr, roc_auc)

def train_with_given_holdout(X_train, Y_train, X_test, Y_test, clf, test_name):
    clf.fit(X_train, Y_train)
    probas = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    return (test_name, fpr, tpr, roc_auc)

class PlainDataSet:
    def __init__(self):
        self.X = None
        self.Y= None
        self.X_test = None
        self.Y_test = None
        
def compute_AUC(y_test, probas):
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def scale(X):
    """
    Scale the matrix X to [0,1].
    
    Parameters:
    ------------
    X: numpy matrix to be scaled.
    
    Returns:
    ---------
    X: returns X scaled.
    """
    
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)
    return X

def scale_two_sets(X, X_pp):
    """
    Scale X and X_pp to a range between 1-0. But use it uses the same 
    normalizers.
    
    Returns:
    --------
    scaled_X:
    scaled_X_pp
    """
    
    X_mins = np.min(X,0)
    X_maxs = np.max(X,0)
    
    Xpps_mins = np.min(X_pp, 0)
    Xpps_maxs = np.max(X_pp, 0)
    
    all_mins = np.vstack((X_mins, Xpps_mins))
    all_maxs = np.vstack((X_maxs, Xpps_maxs))
    
    the_min = np.min(all_mins, 0)
    the_max = np.max(all_maxs, 0)
    
    scaled_X = ( X - the_min ) / ( the_max + 0.0001 )
    scaled_Xpp = ( X_pp - the_min ) / ( the_max + 0.0001 )
    
    return scaled_X, scaled_Xpp

def scale_three_sets(X_pp, X_np, X_nn):
    """
    Scale X_pp, X_np, X_nn to a range between 1-0. But use it uses the same 
    normalizers.
    
    Returns:
    --------
    scaled_X_pp
    scaled_X_np
    scaled_X_nn
    """
    
    Xpps_mins = np.min(X_pp, 0)
    Xpps_maxs = np.max(X_pp, 0)
    
    Xnp_mins = np.min(X_np,0)
    Xnp_maxs = np.max(X_np,0)
    
    Xnn_mins = np.min(X_nn,0)
    Xnn_maxs = np.max(X_nn,0)
    
    all_mins = np.vstack((Xpps_mins, Xnp_mins, Xnn_mins))
    all_maxs = np.vstack((Xpps_maxs, Xnp_maxs, Xnn_maxs))
    
    the_min = np.min(all_mins, 0)
    the_max = np.max(all_maxs, 0)
    
    scaled_Xpp = ( X_pp - the_min ) / ( the_max + 0.0001 )
    scaled_Xnp = ( X_np - the_min ) / ( the_max + 0.0001 )
    scaled_Xnn = ( X_nn - the_min ) / ( the_max + 0.0001 )
    
    return scaled_Xpp, scaled_Xnp, scaled_Xnn

def normalize(X, axis=0):
    if axis == 0:
        X = X / np.sum(X, 0)
        return X
    else:
        X = X.T
        X = normalize(X)
        X = X.T
        return X

def reduce_dims(X, n_dims, split_index=0):
    """
    Reduces the dimensionality of the given dataset using LSA.
    
    Parameters:
    -----------
    X: the dataset.
    n_dims: the number of dimensions to reduce to.
    split_index: if not zero, then this method takes all the columns of the dataset like this:
                  mini_set = X[:, split_index:], then applies the dimentionality reduction to this mini_set.
                  After that it concatinates
                  the reduced mini_set with X[:, :split_index] and returns this new dataset.
    
    Returns:
    ---------
    X_new: the dataset with the dimensions reduced.
    """
    
    if split_index==0:
        svd = TruncatedSVD(n_components=n_dims, random_state=0)
        return svd.fit_transform(X)
    else:
        mini_set = X[:, split_index:]
        svd = TruncatedSVD(n_components=n_dims, random_state=0)
        mini_set = svd.fit_transform(mini_set)
        X = np.column_stack( ( X[:, :split_index], mini_set ) )
        
        return X
        
def filter_unlinked_nodes(G, X_nodes):
    """
    Removes all nodes that are unlinked to any other node.
    And remove their indices from the node attribute matrix.
    
    Parameters:
    -----------
    G: networkx graph.
    X_nodes: the node attribute matrix.
    
    Returns:
    -----------
    G_new: the graph with the unlinked nodes removed.
    X_nodes_new: the node attribute matrix with the rows of the removed nodes removed.
    """
    node_to_index = get_node_to_index(G)
    nodes = G.nodes()
    deleted_nodes = []
    deleted_nodes_indices = []
    
    for node in nodes:
        if G.degree(node) == 0:
            G.remove_node(node)
            deleted_nodes.append(node)
    
    for node in deleted_nodes:
        node_index = node_to_index[node]
        deleted_nodes_indices.append(node_index)
    
    if X_nodes != None:
        X_nodes = np.delete(X_nodes, deleted_nodes_indices, 0)
            
    return G, X_nodes
        
# @profile
def train_with_stratified_cross_validation(G, X_nodes, node_to_index, clf, n_folds, tests_names,
                                            params, plot_file_name, edge_removal_perc=0.5, 
                                            enabled_features=[[1]], undersample=False, extend_graph=False):
    
    """
    This is like the X_pp flag #3. Here we will only use the Xpp for training
    and X for testing. We do that n_folds times. i.e. we remove some dyads (X) and keep them 
    for testing and the others are used for training. We repeat this process n_folds times
    with different dyads removed in each time.
    
    Parameters:
    -----------
    G: networkx graph object.
    X_nodes: the nodes attributes feature matrix.
    node_to_index: dictionary that maps from node name to its index.
    clf: the classifier to use for training.
    n_folds: number of cross validation folds.
    tests_names: a list that contains the names of each test with respect
                 to each dataset in Xs (i.e. the auc curve name).
    params: a dictionary that holds needed parameters.
            Here we need the following: params['rwr_alpha'], 
            params['lrw_nSteps']. (you can omit this parameter if you don't need the global features)
    plot_file_name: the name of the figure to be saved. (you can prepend the path as well).
    """
    
    print '##############Stratiefied cross-validation############'
    
    rocs = []
    prs = []
    t = timer.Timerx()
    
    for set, test_name in zip(enabled_features,tests_names):
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
    
        all_aucs = []
        all_aucs_pr = []
        
        all_y_test = []
        all_props = []
        all_prec = {}
        all_rec = {}
        all_aucs_pr_d = {}
        i = 0
        
        t.start()
        for i in xrange(n_folds):
            random_state = random.randint(0,1000)
            Gx, U, Y, pp_list, np_list, nn_list = prepare_graph_for_training(G, edge_removal_perc, undersample, random_state)
            Y_pp = np.ones( len(pp_list) )
            Y_np = np.ones( len(np_list) )
            Y_nn = np.zeros( len(nn_list) )
            
            if extend_graph:
                Gx, original_degrees = get_extended_graph(Gx, X_nodes)
                original_degrees_list = gsim.get_degrees_list(Gx, original_degrees)
            else:
                original_degrees_list = None
        
            degrees = gsim.get_degrees_list(Gx)
            A = nx.adj_matrix(Gx)
            A = np.asarray(A)
            
            X_pp = None
            X_np = None
            X_nn = None
            
            for feature_flag in set:
                if feature_flag == 1:#local topo features 
                    X_pp = add_local_topo_features(X_pp, A, pp_list, degrees, G.nodes(), original_degrees_list)
                    X_np = add_local_topo_features(X_np, A, np_list, degrees, G.nodes(), original_degrees_list)
                    X_nn = add_local_topo_features(X_nn, A, nn_list, degrees, G.nodes(), original_degrees_list)
                elif feature_flag == 2:# global topo features
                    X_pp = add_global_features(Gx, X_pp, pp_list, node_to_index, params)
                    X_np = add_global_features(Gx, X_np, np_list, node_to_index, params)
                    X_nn = add_global_features(Gx, X_nn, nn_list, node_to_index, params)
                elif feature_flag == 3:#node attributes
                    X_pp = add_raw_feature(X_pp, pp_list, X_nodes, G.nodes(), node_to_index=node_to_index)
                    X_np = add_raw_feature(X_np, np_list, X_nodes, G.nodes(), node_to_index=node_to_index)
                    X_nn = add_raw_feature(X_nn, nn_list, X_nodes, G.nodes(), node_to_index=node_to_index)
                elif feature_flag == 4:#raw features
                    X_pp = add_raw_feature(X_pp, pp_list, A, G.nodes(), node_to_index=node_to_index)
                    X_np = add_raw_feature(X_np, np_list, A, G.nodes(), node_to_index=node_to_index)
                    X_nn = add_raw_feature(X_nn, nn_list, A, G.nodes(), node_to_index=node_to_index)
    
            X_pp, y_pp = shuffle(X_pp, Y_pp, random_state=random_state)
            X_np, y_np = shuffle(X_np, Y_np, random_state=random_state)
            X_nn, y_nn = shuffle(X_nn, Y_nn, random_state=random_state)

            X_pp, X_np, X_nn = scale_three_sets(X_pp, X_np, X_nn)
            
            half_nn = int( len(X_nn) / 2 )
            X_train = np.vstack( (X_pp, X_nn[0:half_nn]) )
            y_train = np.concatenate( (y_pp, y_nn[0:half_nn]) )
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
            
            X_test = np.vstack( (X_np, X_nn[half_nn:]) )
            y_test = np.concatenate( (y_np, y_nn[half_nn:]) )
            X_test, y_test = shuffle(X_test, y_test, random_state=random_state)

            probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            all_aucs.append( auc(fpr, tpr) )
            mean_tpr[0] = 0.0
            
            precision, recall, average_precision = get_PR_curve_values(y_test, probas_[:, 1])
            all_prec[i] =  precision
            all_rec[i] = recall
            all_aucs_pr_d[i] = average_precision
            all_aucs_pr.append( average_precision )
            
            all_y_test.extend(y_test)
            all_props.extend(probas_[:, 1])
            
            i += 1
            
        
        all_aucs = np.array(all_aucs)
        all_aucs_pr = np.array(all_aucs_pr)
        
        mean_tpr /= n_folds
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        mean_prec, mean_recall, _ =  get_PR_curve_values(all_y_test, all_props)
        mean_pr_auc = np.mean(all_aucs_pr)
        all_prec['mean'] = mean_prec
        all_rec['mean'] = mean_recall
        all_aucs_pr_d['mean'] = mean_pr_auc
        all_y_test = np.array(all_y_test)
        random_prec = all_y_test[all_y_test.nonzero()].size / all_y_test.size
        
        print "Stratified Cross-validation ROC auc stats: Testname: %s,\
         STD: %f, Variance: %f.\n" % (test_name, np.std(all_aucs), np.var(all_aucs))
        
        print "Stratified Cross-validation PR auc stats: Testname: %s,\
         STD: %f, Variance: %f.\n" % (test_name, np.std(all_aucs_pr), np.var(all_aucs_pr))
         
        the_roc_curve = (test_name, mean_fpr, mean_tpr, mean_auc)
        the_pr_curve = (test_name, mean_prec, mean_recall, mean_pr_auc)
        all_curves = {"roc":the_roc_curve, "pr": the_pr_curve, "all_prec": all_prec, "all_rec": all_rec, "all_auc_pr": all_aucs_pr_d}
        print "[Stratified] Test name: %s, time: %s, ROC auc: %f\n" % (test_name, t.stop(), the_roc_curve[3])
        print "[Stratified] PR auc: %f\n" % (the_pr_curve[3])
        rocs.append(the_roc_curve)
        prs.append(the_pr_curve)
        draw_pr_curves_n_folds(n_folds, test_name, all_prec, all_rec, all_aucs_pr_d, random_prec, plot_file_name[:-4]+"_folds_"+test_name+"_PR_.png")
    draw_rocs(rocs, plot_file_name)
    draw_pr_curves(prs, plot_file_name[:-4]+"_PR_.png")
            
            
            
def get_stratified_training_sets(G, X_nodes, node_to_index, params, edge_removal_perc=0.5, enabled_features=[1], 
                                 undersample=False, extend_graph=False, random_state=0):            
    Gx, U, Y, pp_list, np_list, nn_list = prepare_graph_for_training(G, edge_removal_perc, undersample, random_state)
    Y_pp = np.ones( len(pp_list) )
    Y_np = np.ones( len(np_list) )
    Y_nn = np.zeros( len(nn_list) )
    
    if extend_graph:
        Gx, original_degrees = get_extended_graph(Gx, X_nodes)
        original_degrees_list = gsim.get_degrees_list(Gx, original_degrees)
    else:
        original_degrees_list = None
            
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    
    X_pp = None
    X_np = None
    X_nn = None
    
    for feature_flag in enabled_features:
        if feature_flag == 1:#local topo features 
            X_pp = add_local_topo_features(X_pp, A, pp_list, degrees, G.nodes(), original_degrees_list)
            X_np = add_local_topo_features(X_np, A, np_list, degrees, G.nodes(), original_degrees_list)
            X_nn = add_local_topo_features(X_nn, A, nn_list, degrees, G.nodes(), original_degrees_list)
        elif feature_flag == 2:# global topo features
            X_pp = add_global_features(Gx, X_pp, pp_list, node_to_index, params)
            X_np = add_global_features(Gx, X_np, np_list, node_to_index, params)
            X_nn = add_global_features(Gx, X_nn, nn_list, node_to_index, params)
        elif feature_flag == 3:#node attributes
            X_pp = add_raw_feature(X_pp, pp_list, X_nodes, G.nodes(), node_to_index=node_to_index)
            X_np = add_raw_feature(X_np, np_list, X_nodes, G.nodes(), node_to_index=node_to_index)
            X_nn = add_raw_feature(X_nn, nn_list, X_nodes, G.nodes(), node_to_index=node_to_index)
        elif feature_flag == 4:#raw features
            X_pp = add_raw_feature(X_pp, pp_list, A, G.nodes(), node_to_index=node_to_index)
            X_np = add_raw_feature(X_np, np_list, A, G.nodes(), node_to_index=node_to_index)
            X_nn = add_raw_feature(X_nn, nn_list, A, G.nodes(), node_to_index=node_to_index)

    X_pp, y_pp = shuffle(X_pp, Y_pp, random_state=random_state)
    X_np, y_np = shuffle(X_np, Y_np, random_state=random_state)
    X_nn, y_nn = shuffle(X_nn, Y_nn, random_state=random_state)

    X_pp, X_np, X_nn = scale_three_sets(X_pp, X_np, X_nn)
    
    half_nn = int( len(X_nn) / 2 )
    X_train = np.vstack( (X_pp, X_nn[0:half_nn]) )
    y_train = np.concatenate( (y_pp, y_nn[0:half_nn]) )
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    
    X_test = np.vstack( (X_np, X_nn[half_nn:]) )
    y_test = np.concatenate( (y_np, y_nn[half_nn:]) )
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)
    
    return X_train, X_test, y_train, y_test   
            
            







