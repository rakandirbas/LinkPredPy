"""
This is very very old!!!! Dont look at it!
"""


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
import pylab as pl
import random

def main():
    pass
#     train_evolv_net()
    get_n_neihoods()


def get_n_neihoods():
    print 'Started...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    G = read_graph('/Users/rockyrock/Documents/workspace/Graph/data/edges.txt')
#     G, Nodes_X = fb_parser.read_mat_file("/Users/rockyrock/Documents/workspace/Graph/data/Caltech36.mat")
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
        data = add_feature(data, U_n[i], gsim.cn(A), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.lp(A), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.salton(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.jacard(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.sorensen(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.hpi(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.hdi(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.lhn1(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.pa(A, degrees), Gx.nodes())
        data = add_feature(data, U_n[i], gsim.ra(A, degrees), Gx.nodes())
        datas[i] = data
        
    print 'Scaling...'    
    for data in datas.values():
        min_max_scaler = preprocessing.MinMaxScaler()
        if data.shape[0] != 0:
            data = min_max_scaler.fit_transform(data)
         
    print 'Training...'
    folds_auc = []
    folds_fpr = []
    folds_tpr = []
    num_folds = 3.0
    for fold_k in xrange(int(num_folds)):
        X2, y2 = shuffle(datas[2], Y_2, random_state=random_state)
        X3, y3 = shuffle(datas[3], Y_3, random_state=random_state)
        X4, y4 = shuffle(datas[4], Y_4, random_state=random_state)
        
        X2_train, X2_test, y2_train, y2_test = \
            cross_validation.train_test_split(X2, y2, test_size=0.33, random_state=random_state)
          
        X3_train, X3_test, y3_train, y3_test = \
            cross_validation.train_test_split(X3, y3, test_size=0.33, random_state=random_state)
               
        X4_train, X4_test, y4_train, y4_test = \
            cross_validation.train_test_split(X4, y4, test_size=0.33, random_state=random_state)
              
        X_test = np.vstack((X2_test, X3_test, X4_test))
        y_test = np.vstack((y2_test, y3_test, y4_test))
        
        estimators = {}
        for i in sorted(U_n):
            estimators[i] = RandomForestClassifier(n_estimators=10, n_jobs=-1)
            
        estimators[2].fit(X2_train, y2_train)
        estimators[3].fit(X3_train, y3_train)
        estimators[4].fit(X4_train, y4_train)
        
        probas2 = estimators[2].predict_proba(X_test)
        probas3 = estimators[3].predict_proba(X_test)
        probas4 = estimators[4].predict_proba(X_test)
        
        probas = probas2[:,1] + probas3[:,1] + probas4[:,1]
        probas = probas / 3.0
        
        fpr, tpr, thresholds = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        folds_auc.append(roc_auc)
        folds_fpr.append(fpr)
        folds_tpr.append(tpr)
        
    fpr = sum(folds_fpr) / num_folds
    tpr = sum(folds_tpr) / num_folds
    roc_auc = sum(folds_auc) / num_folds
    
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
    pl.savefig("geod_roc.png",dpi=72)
    
def train_evolv_net():
    print 'Started...'
    SEED = 0
    random_state = np.random.RandomState(SEED)
    random.seed(SEED)
    t = Timerx(True)
    path = "/Users/rockyrock/Desktop/edges.txt"
    G = read_graph(path)
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
    data = add_feature(data, U, gsim.lp(A), Gx.nodes())
    
    print 'Scaling...'
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    
    print 'Training...'
    X, y = shuffle(data, Y, random_state=random_state)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=0.33, random_state=random_state)
        
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    
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
#     pl.title('ROC Curve')
    pl.legend(loc="lower right")
    pl.savefig("roc.png",dpi=72)
#     pl.show()
    
def test_AUC():
    random_state = np.random.RandomState(0)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    #make it a binary classification problem
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape
    
    # Add noisy features to make the problem harder
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    
    half = int(n_samples/2)
    X, y = shuffle(X, y, random_state=random_state)
    X_train, X_test = X[:half], X[half:]
    y_train, y_test = y[:half], y[half:]
    
    clf = RandomForestClassifier(n_estimators=3)
    clf.fit(X_train, y_train)
    probas = clf.predict_proba(X_test)
    print probas.shape
    
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_auc2 = roc_auc_score(y_test, probas[:, 1])
    print fpr
    print tpr
    print thresholds
    print("Area under the ROC curve : %f, %f" % (roc_auc, roc_auc2))
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
    
    
def test_facebook_data():
    t = Timerx(True)
    path = "/Users/rockyrock/Documents/workspace/Graph/data/Caltech36.mat"
    G, Nodes_X = fb_parser.read_mat_file(path)
    print G.number_of_nodes()
#     U = build_U(G.nodes())
#     Y = build_Y(G, U)
#     degrees = gsim.get_degrees_list(G)
#     A = nx.adj_matrix(G)
#     A = np.asarray(A)
#     data = None
    
    #Adding features
#     data = add_feature(data, U, gsim.lp(A), G.nodes())
#     data = fb_parser.add_nodes_features(Nodes_X, data, U, G.nodes())

def test_aa():
    t = Timerx(True)
    path = "/Users/rockyrock/Desktop/all.txt"
    G = read_graph(path)
    degrees = gsim.get_degrees_list(G)
    A = nx.adj_matrix(G)
    A = np.asarray(A)
    print 'computing...'
    t.start()
    S = gsim.aa(A, degrees)
#     print S.shape
#     i = G.nodes().index(44)
#     j = G.nodes().index(1)
#     print S[i,j]
    t.stop()
    print 'done.'
    
def test3():
    print 'Started...'
    t = Timerx(True)
    path = "/Users/rockyrock/Desktop/all.txt"
    G = read_graph(path)
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
    clf = RandomForestClassifier(n_estimators=1000)
    kfold = cross_validation.KFold(len(data), n_folds=3)
    scores = [clf.fit(data[train], Y[train]).score(data[hpc_test], Y[hpc_test]) for train, hpc_test in kfold]
    print scores
    t.stop()
    
    t.start()
    print 'Persisting'
    joblib.dump(clf, 'model_per/clf_links.pkl')
    joblib.dump(data, 'data_per/data.pkl')
    t.stop()

# @profile
def test2():
    t = Timerx(True)
    path = "/Users/rockyrock/Desktop/all.txt"
    G = read_graph(path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    degrees = gsim.get_degrees_list(G)
    A = nx.adj_matrix(G)
    A = np.asarray(A)
#     S = gsim.cn(A)
    data = None
    data = add_feature(data, U, gsim.cn(A), G.nodes())
      
def test1():
    t = Timerx(True)
    path = "/Users/rockyrock/Desktop/all.txt"
    G = read_graph(path)
    U = build_U(G.nodes())
    Y = build_Y(G, U)
    print G.number_of_nodes()
    print len(U), len(Y)
    
    degrees = gsim.get_degrees_list(G)
    A = nx.adj_matrix(G)
    A = np.asarray(A)
    
    print 'aa'
    t.start()
    S = gsim.aa(A,degrees)
    t.stop()
#     i = G.nodes().index('1')
#     j = G.nodes().index('52')
#     print S[i,j]
#     print S


main()