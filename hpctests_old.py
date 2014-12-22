from rakg.graph_utils import *
from rakg.timer import *
import rakg.graphsim as gsim
import numpy as np
import networkx as nx
from itertools import izip
import sys
from memory_profiler import profile
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import cross_validation
import rakg.facebook100_parser as fb_parser
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


def main():
    args = sys.argv
    choice = int(args[1])
    if choice == 1:
        test_randomForest_with_toplogy_only()
    elif choice == 2:
        test_randomForest_with_atts_only()
    elif choice == 3:
        test_randomForest_with_toplogy_and_atts()
    elif choice == 4:
        get_n_neihoods_topo_only()
    elif choice == 5:
        draw_combined_roc()
    elif choice == 6:
        rbm_logisticreg_test_topology()
    elif choice == 7:
        rbm_logisticreg_test_topology_atts()
    elif choice == 8:
        rbm_logisticreg_test_topology_cross()
        
def rbm_logisticreg_test_topology_cross():
    print 'Started rbm_logisticreg_test_topology_cross'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    print 'U length:', len(U)
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    print 'Data shape w/o atts:', data.shape
    #add nodes attributes
#     data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    print 'Data shape w atts:', data.shape
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = (data - np.min(data, 0)) / (np.max(data, 0) + 0.0001)  # 0-1 scaling
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=random_state, verbose=True)
    clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000.0
    kfold = cross_validation.KFold(len(X), n_folds=10)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
        
    t.start()
    for train, test in kfold:
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    t.stop()
    print("Area under the ROC curve : %f" % (mean_auc))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(mean_fpr, mean_tpr, label='ROC curve (area = %0.2f)' % mean_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
    pl.savefig("rbm_toplogy_cross.png",dpi=72)
    return mean_fpr, mean_tpr, mean_auc
        
def rbm_logisticreg_test_topology_atts():
    print 'Started rbm_logisticreg_test_topology_atts'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    print 'U length:', len(U)
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    print 'Data shape w/o atts:', data.shape
    #add nodes attributes
    data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    print 'Data shape w atts:', data.shape
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = (data - np.min(data, 0)) / (np.max(data, 0) + 0.0001)  # 0-1 scaling
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    t.start()
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=random_state, verbose=True)
    clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000.0
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    t.stop()
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % (roc_auc))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
    pl.savefig("rbm_toplogy_atts.png",dpi=72)


    return fpr, tpr, roc_auc
    
def rbm_logisticreg_test_topology():
    print 'Started rbm_logisticreg_test_topology'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    print 'U length:', len(U)
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    print 'Data shape w/o atts:', data.shape
    #add nodes attributes
#     data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    print 'Data shape w atts:', data.shape
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = (data - np.min(data, 0)) / (np.max(data, 0) + 0.0001)  # 0-1 scaling
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    t.start()
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=random_state, verbose=True)
    clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    logistic.C = 6000.0
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    t.stop()
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % (roc_auc))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
    pl.savefig("rbm_toplogy.png",dpi=72)


    return fpr, tpr, roc_auc
        
def draw_combined_roc():
    fpr1, tpr1, roc_auc1 = test_crossvalidation_randomForest_with_toplogy_only()
    fpr2, tpr2, roc_auc2 = test_crossvalidation_randomForest_with_atts_only()
    fpr3, tpr3, roc_auc3 = test_crossvalidation_randomForest_with_toplogy_and_atts()
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr1, tpr1, lw=1, label='Topo ROC curve (area = %0.2f)' % roc_auc1)
    pl.plot(fpr2, tpr2, lw=1, label='Node atts ROC curve (area = %0.2f)' % roc_auc2)
    pl.plot(fpr3, tpr3, lw=1, label='Combined ROC curve (area = %0.2f)' % roc_auc3)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.savefig("roc_combined.png",dpi=72)

def test_crossvalidation_randomForest_with_toplogy_and_atts():
    """
    This tests a random forest classifier with a dataset of links
    that has both topology and node atts features. 
    It uses cross-validation for testing.
    
    Returns
    -------
    It returns the FPR, TPR, AUC for plotting
    """
    
    print 'Started test_randomForest_with_toplogy_and_atts...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    print 'U length:', len(U)
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    print 'Data shape w/o atts:', data.shape
    #add nodes attributes
    data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    print 'Data shape w atts:', data.shape
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    kfold = cross_validation.KFold(len(X), n_folds=10)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in kfold:
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, mean_auc

def test_crossvalidation_randomForest_with_atts_only():
    """
    This tests a random forest classifier with a dataset of links
    that has only node atts features only. It uses cross-validation for testing.
    
    Returns
    -------
    It returns the FPR, TPR, AUC for plotting
    """
    
    print 'Started test_randomForest_with_atts_only...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    #add nodes attributes
    data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    kfold = cross_validation.KFold(len(X), n_folds=10)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in kfold:
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, mean_auc
    

def test_crossvalidation_randomForest_with_toplogy_only():
    """
    This tests a random forest classifier with a dataset of links
    that has only topology features. It uses cross-validation for testing.
    
    Returns
    -------
    It returns the FPR, TPR, AUC for plotting
    """
    print 'Started test_randomForest_with_toplogy_only...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    kfold = cross_validation.KFold(len(X), n_folds=10)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in kfold:
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_fpr, mean_tpr, mean_auc

def get_n_neihoods_topo_only():
    print 'Started... n_neighoods'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    print G.number_of_nodes()
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    U_n = build_Un_neighborhoods(Gx, U, [2,3,4])
    num_subsets = len(U_n)
    Y_2 = build_Y(G, U_n[2])
    Y_3 = build_Y(G, U_n[3])
    Y_4 = build_Y(G, U_n[4])
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    datas = {}
    
    for i in sorted(U_n):
        data = None
#         data = add_feature(data, U_n[i], gsim.cn(A), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.lp(A), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.salton(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.jacard(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.sorensen(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.hpi(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.hdi(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.lhn1(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.pa(A, degrees), Gx.nodes())
#         data = add_feature(data, U_n[i], gsim.ra(A, degrees), Gx.nodes())
        #add nodes attributes features 
        data = fb_parser.add_nodes_features(Nodes_X, data, U_n[i], Gx.nodes())
        datas[i] = data
        
    print 'Scaling...'    
    for data in datas.values():
        min_max_scaler = preprocessing.MinMaxScaler()
        if data.shape[0] != 0:
            data = min_max_scaler.fit_transform(data)
         
    print 'Training...'
    X2, y2 = shuffle(datas[2], Y_2, random_state=random_state)
    X3, y3 = shuffle(datas[3], Y_3, random_state=random_state)
    X4, y4 = shuffle(datas[4], Y_4, random_state=random_state)
    num_folds = 10.0
    kfold2 = cross_validation.KFold(len(X2), n_folds=num_folds)
    kfold3 = cross_validation.KFold(len(X3), n_folds=num_folds)
    kfold4 = cross_validation.KFold(len(X4), n_folds=num_folds)
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    pl.clf()
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for train, test in kfold2:
        probas_ = clf.fit(X2[train], y2[train]).predict_proba(X2[test])
        fpr, tpr, thresholds = roc_curve(y2[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
    mean_tpr /= num_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr, lw=1, label='n=2 ROC curve (area = %0.2f)' % mean_auc)
    
    ####
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    for train, test in kfold3:
        probas_ = clf.fit(X3[train], y3[train]).predict_proba(X3[test])
        fpr, tpr, thresholds = roc_curve(y3[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
    mean_tpr /= num_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr, lw=1, label='n=3 ROC curve (area = %0.2f)' % mean_auc)
    
    ####
#     mean_tpr = 0.0
#     mean_fpr = np.linspace(0, 1, 100)
#     
#     for train, test in kfold4:
#         probas_ = clf.fit(X4[train], y4[train]).predict_proba(X4[test])
#         fpr, tpr, thresholds = roc_curve(y4[test], probas_[:, 1])
#         mean_tpr += interp(mean_fpr, fpr, tpr)
#         mean_tpr[0] = 0.0
#         
#     mean_tpr /= num_folds
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     pl.plot(mean_fpr, mean_tpr, lw=1, label='n=4 ROC curve (area = %0.2f)' % mean_auc)
    
     # Plot ROC curve
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.savefig("geod-atts.png",dpi=72)
        
def test_randomForest_with_atts_only():
    print 'Started test_randomForest_with_atts_only...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    #add nodes attributes
    data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    t.start()
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    t.stop()
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % (roc_auc))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
    pl.savefig("roc_atts_only.png",dpi=72)


def test_randomForest_with_toplogy_and_atts():
    print 'Started test_randomForest_with_toplogy_and_atts...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    print 'U length:', len(U)
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    print 'Data shape w/o atts:', data.shape
    #add nodes attributes
    data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())
    print 'Data shape w atts:', data.shape
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    t.start()
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    t.stop()
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % (roc_auc))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
    pl.savefig("roc_toplogy_and_atts.png",dpi=72)

def test_randomForest_with_toplogy_only():
    print 'Started test_randomForest_with_toplogy_only...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    
    Gx = G.copy()
    removal_perc = 0.5
    num_edges_to_remove = int( removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    degrees = gsim.get_degrees_list(Gx)
    A = nx.adj_matrix(Gx)
    A = np.asarray(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), Gx.nodes())
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    data = add_feature(data, U, gsim.ra(A, degrees), G.nodes())
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.33, random_state=random_state)
    
    t.start()
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    t.stop()
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % (roc_auc))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.legend(loc="lower right")
    pl.savefig("roc_toplogy.png",dpi=72)

def hpc_test2():
    print 'Started...'
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/all.txt"
    G = read_graph(path) #Reads a links-list file
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    degrees = gsim.get_degrees_list(G)
    A = nx.adj_matrix(G)
    A = np.asarray(A)
    data = None
    t.start()
    data = add_feature(data, U, gsim.lp(A), G.nodes())
    print 'LP'
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    print 'Salton'
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    print 'Jacard'
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    print 'Sorensen'
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    print 'HPI'
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    print 'HDI'
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    print 'LHN1'
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    print "PA"
    t.stop()
    
    t.start()
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    t.stop()
    
    t.start()
    print 'Training...'
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=20)
    clf.fit(data, Y)
    t.stop()
    
    t.start()
    print 'Persisting'
    joblib.dump(clf, 'clf_links2.pkl')
    t.stop()
    
def hpc_test():
    print 'Started...'
    t = Timerx(True)
    path = "/home/rdirbas/Graph/Graph/data/all.txt"
    G = read_graph(path) #Reads a links-list file
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    degrees = gsim.get_degrees_list(G)
    A = nx.adj_matrix(G)
    A = np.asarray(A)
    data = None
    t.start()
    data = add_feature(data, U, gsim.lp(A), G.nodes())
    print 'LP'
    data = add_feature(data, U, gsim.salton(A, degrees), G.nodes())
    print 'Salton'
    data = add_feature(data, U, gsim.jacard(A, degrees), G.nodes())
    print 'Jacard'
    data = add_feature(data, U, gsim.sorensen(A, degrees), G.nodes())
    print 'Sorensen'
    data = add_feature(data, U, gsim.hpi(A, degrees), G.nodes())
    print 'HPI'
    data = add_feature(data, U, gsim.hdi(A, degrees), G.nodes())
    print 'HDI'
    data = add_feature(data, U, gsim.lhn1(A, degrees), G.nodes())
    print 'LHN1'
    data = add_feature(data, U, gsim.pa(A, degrees), G.nodes())
    print "PA"
    t.stop()
    
    t.start()
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    t.stop()
    
    t.start()
    print 'Saving data....'
    np.savetxt("X.csv", data, delimiter=',')
    t.stop()
    
    t.start()
    print 'Training...'
    clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    kfold = cross_validation.KFold(len(data), n_folds=3)
    scores = [clf.fit(data[train], Y[train]).score(data[hpc_test], Y[hpc_test]) for train, hpc_test in kfold]
    print scores
    clf.fit(data, Y)
    t.stop()
     
    t.start()
    print 'Persisting'
    joblib.dump(clf, 'clf_links.pkl')
#     joblib.dump(data, 'data_per/data.pkl')
    t.stop()    
    
    
main()