"""
To test the stratified cross-validation version
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

def main():
    print 'Started...'


#     dataset_path = "/Users/rockyrock/Desktop/facebook/3980.edges"
    dataset_path = "/Users/rockyrock/Desktop/facebook/0.edges"
    path = "/Users/rockyrock/Desktop/tests/"
     
    G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
    G = testing_utils.filter_unlinked_nodes(G)
    node_to_index = testing_utils.get_node_to_index(G)
    removal_perc = 0.30 
    

    clfs = []
     
    logistic = linear_model.LogisticRegression()
    clfs.append(logistic) #1
 
     
    options = {}
    options['rwr_alpha'] = 0.7
    options['lrw_nSteps'] = 10
     
    enabled_features = [[1], [2], [3], [4]]
     
    tests_names = ["local", "global", 'atts', 'raw']
     
        
    for i, clf in enumerate(clfs):
        plot_file_name = path + str(1) + '/' + "_clfid_" + str(i) + ".png"
        print 'test with clf id: ', i, "\n\n"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)

    print 'End'
main()