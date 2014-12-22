"""
Testing the memory usage of training with normal classifiers.
"""

import numpy as np
import networkx as nx
from memory_profiler import profile
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import datasets, testing_utils, graph_utils, graphsim, datasets_stats
from sklearn.utils import shuffle
from sklearn import linear_model
from runs import _0

# @profile
def main():
    print 'Started...'
#     X = np.random.randn(10**6,600) #it's taking 1.3GB
#     print X.shape


    dataset_path = "/Users/rockyrock/Desktop/facebook/3980.edges"
#     dataset_path = "/Users/rockyrock/Desktop/facebook/0.edges"
    path = "/Users/rockyrock/Desktop/tests/"
     
    G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
    G = testing_utils.filter_unlinked_nodes(G)
    node_to_index = testing_utils.get_node_to_index(G)
    removal_perc = 0.20
    
    print G.number_of_nodes()

    clfs = []
     
    logistic = linear_model.LogisticRegression()
    clfs.append(logistic) #1
 
    X_pp_flags = [0, 1, 2]
     
    options = {}
    options['rwr_alpha'] = 0.7
    options['lrw_nSteps'] = 10
     
    enabled_features = [[1], [2], [3], [4]]
     
    tests_names = ["local", "global", 'atts', 'raw']
     
    Xs, Ys, X_pps, Y_pps = \
        testing_utils.build_training_datasets(G, X_nodes, node_to_index, params=options, 
                    edge_removal_perc=removal_perc, enabled_features=enabled_features)
        
    for i, clf in enumerate(clfs):
        plot_file_name = path + str(1) + '/' + "_clfid_" + str(i)
        print 'test with clf id: ', i, "\n\n"
        _0.test_supervised_methods_general(clf, Xs, Ys, X_pps, Y_pps, X_pp_flags, tests_names, plot_file_name)


    print 'End'
main()