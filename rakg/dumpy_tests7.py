"""
Saves a dataset into .mat
"""

import numpy as np
import scipy.io
import datasets
import testing_utils
import networkx as nx
import matrix_fact

print 'Started'

dataset_path = "/Users/rockyrock/Desktop/facebook/3980.edges"
# dataset_path = "/Users/rockyrock/Desktop/facebook/0.edges"
G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
G = testing_utils.filter_unlinked_nodes(G)

A = nx.adj_matrix(G)
A = np.array(A)


scipy.io.savemat('/Users/rockyrock/Desktop/dataset.mat', mdict={'A': A, 'X_nodes': X_nodes})


"Little test to test the MF code"
# MF = matrix_fact.Matrix_Factorization(k = 2, G=G, X=X_nodes )
# 
# fpr, tpr, auc = MF.train_test_normal_model(n_folds = 2, 
#             alpha=0.1, n_iter=25)
# 
# print "The auc: ", auc







