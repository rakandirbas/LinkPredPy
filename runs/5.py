import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from rakg import graphsim, graph_utils,\
     light_srw, matrix_fact, supervised_methods, \
     testing_utils, timer, srw, facebook100_parser, datasets_stats,\
     datasets

import _0

# dataset_path = "/home/rdirbas/datasets/SNAP/gplus/101263615503715477581.edges"
dataset_path = "/Users/rockyrock/Desktop/gplus/101263615503715477581.edges"
G, X_nodes = datasets.load_SNAP_dataset(dataset_path, directed=True)
# G = G.to_undirected(reciprocal=True)
G, X_nodes = testing_utils.filter_unlinked_nodes(G, X_nodes)
node_to_index = testing_utils.get_node_to_index(G)

removal_perc = 0.5
options = {}
options['katz_h'] = 0.005



_0.cross_validation_for_unsupervised_local_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10)


print 'Done...'