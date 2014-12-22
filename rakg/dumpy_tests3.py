"""
To test generating the plots
"""
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datasets, testing_utils, graph_utils, graphsim, datasets_stats
from sklearn.utils import shuffle

dataset_path = "/Users/rockyrock/Desktop/facebook/3980.edges"
path = "/Users/rockyrock/Desktop/tests/"

# G=nx.Graph()
# G.add_edges_from([(1,2), (1,6),(2,3), (3,4), (4,5), (5,6), (3,6), (3,5)])
# X_nodes = np.array( [[1,0,0], [1,0,0], [1,0,0], [0,0,1], [0,1,0], [0,0,1]] )

G, X_nodes = datasets.load_SNAP_dataset(dataset_path)


G = testing_utils.filter_unlinked_nodes(G)
node_to_index = testing_utils.get_node_to_index(G)
removal_perc = 0.20

print datasets_stats.get_statistic(G, X_nodes)
        
enabled_features = [[1], [2], [1,2]]
tests_names = ["local", "global", 'loc_glob']

options = {}
options['rwr_alpha'] = 0.7
options['lrw_nSteps'] = 10

Xs, Ys, X_pps, Y_pps = \
    testing_utils.build_training_datasets(G, X_nodes, node_to_index, params=options, 
                edge_removal_perc=removal_perc, enabled_features=enabled_features)


for X, Y, X_pp, Y_pp, test_name in zip(Xs, Ys, X_pps, Y_pps, tests_names):
    plot_file_name = path + str(1) + '/' + test_name + "_"
    datasets_stats.save_covariance_plot(X, plot_file_name + 'cov_' + 'X')
    datasets_stats.save_covariance_plot(X_pp, plot_file_name + 'cov_' + 'X_pp')
    
    datasets_stats.save_scatterplot_matix(X, plot_file_name + 'scatter_' + 'X')
    datasets_stats.save_scatterplot_matix(X_pp, plot_file_name + 'scatter_' + 'X_pp')
    
    X_train = np.vstack(( X_pp, X ))
    y_train = np.concatenate(( Y_pp, Y ))
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    
    importances, feature_order = datasets_stats.get_important_features(X_train, y_train)
    print '\n\nPrinting the features and their importances\n'
    print 'The first array is the features_order. The second is the importance of each feature\
    sorted from the most important to lowest.\n\n'
    
    print "features order:\n\n"
    print feature_order 
    print '\n\n'
    print "Importances:\n\n"
    print importances[feature_order]
    print '\n\n'
    
