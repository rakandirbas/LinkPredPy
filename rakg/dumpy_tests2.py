import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datasets, testing_utils, graph_utils, graphsim
from runs import _0 as exp

dataset_path = "/Users/rockyrock/Desktop/facebook/0.edges"

G=nx.Graph()

G.add_edges_from([(1,2), (1,6),(2,3), (3,4), (4,5), (5,6), (3,6), (3,5)])

print G.nodes()

print G.degree(G.nodes())
 
X_nodes = np.array( [[1,0,0], [1,0,0], [1,0,0], [0,0,1], [0,1,0], [0,0,1]] )

# G, X_nodes = datasets.load_SNAP_dataset(dataset_path)



G = testing_utils.filter_unlinked_nodes(G)
node_to_index = testing_utils.get_node_to_index(G)
removal_perc = 0.20

print G.number_of_nodes(), G.number_of_edges()
 
plt.figure(1)
nx.draw_shell(G)

Gx, U, Y, pp_list, np_list, nn_list = \
            graph_utils.prepare_graph_for_training(G, removal_perc)
            
print 'U', U 
            
print Gx.number_of_nodes(), Gx.number_of_edges()
            
plt.figure(2)
nx.draw_shell(Gx)
 
# Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes)
# original_degrees_list = graphsim.get_degrees_list(Gx, original_degrees)
# print Gx.number_of_nodes(), Gx.number_of_edges()
# plt.figure(3)
# nx.draw_shell(Gx)


options = {}            
options['katz_h'] = 0.005

# exp.test_local_unsupervised_methods(Gx, U, Y, node_to_index, options, original_degrees_list)

exp.test_local_unsupervised_methods(Gx, U, Y, node_to_index, options)
plt.show()

print 'done!'