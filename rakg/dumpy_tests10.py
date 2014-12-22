import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datasets, testing_utils, graph_utils, graphsim

print 'Started ...'

dataset_path = "/Users/rockyrock/Desktop/gplus/101263615503715477581.edges"
# dataset_path = "/Users/rockyrock/Desktop/facebook/414.edges"

G, X_nodes = datasets.load_SNAP_dataset(dataset_path, directed=True)

if G.is_directed():
    print 'directed'

print "#Nodes: ", G.number_of_nodes(), "#Edges: ", G.number_of_edges()

G = G.to_undirected(reciprocal=True)

print "#Nodes: ", G.number_of_nodes(), "#Edges: ", G.number_of_edges()

G, X_nodes = testing_utils.filter_unlinked_nodes(G, X_nodes)

print "#Nodes: ", G.number_of_nodes(), "#Edges: ", G.number_of_edges()

# node_to_index = testing_utils.get_node_to_index(G)
# 
# print X_nodes.shape
# 
# removal_perc = 0.4
#  
# Gx, U, Y, pp_list, np_list, nn_list = \
#             graph_utils.prepare_graph_for_training(G, removal_perc) 
#  
# A = np.array( nx.adj_matrix(G) )
#  
# katz_h = graphsim.katz_h(A)
# katz_h = katz_h * 0.1
# katz_p = graphsim.predict_scores(U, graphsim.katz(A, katz_h), node_to_index)
# print katz_p
# katz_auc = testing_utils.compute_AUC(Y, katz_p)
# print "Katz: ", katz_auc, "\n"

print 'Done.'