"""
Testing the extended graphs
"""

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

#add the extra edges to be removed
G.add_edges_from([(1,3), (2,4), (2,5), (2,6), (4,6)])

# print G.nodes()

# print G.degree(G.nodes())
 
X_nodes = np.array( [[1,0], [0,1], [1,0], [0,1], [0,1], [1,0]] )

# G, X_nodes = datasets.load_SNAP_dataset(dataset_path)



G = testing_utils.filter_unlinked_nodes(G)
node_to_index = testing_utils.get_node_to_index(G)
removal_perc = 0.20

# print G.number_of_nodes(), G.number_of_edges()
 
plt.figure(1)
# 
# pos = nx.shell_layout(G)
# nx.draw_networkx_nodes(G, pos, nodelist=[1,2,3,4,5,6], node_color='r', node_size=500)
# p = [(1,2), (1,6),(2,3), (3,4), (5,6), (3,5), (1,3), (2,5), (2,6), (4,6)]
# nx.draw_networkx_edges(G, pos, edgelist=p, 
#                        edge_color='k')
# nx.draw_networkx_edges(G, pos, 
#                        edgelist=[(4, 5),(3, 6),(2, 4)], 
#                        edge_color='r')
# 
# labels={1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
# 
# nx.draw_networkx_labels(G,pos,labels)
# plt.axis('off')

# plt.savefig('/Users/rockyrock/Desktop/1.eps', bbox_inches='tight')
# plt.savefig('/Users/rockyrock/Desktop/1.png', bbox_inches='tight')

nx.draw_shell(G)

Gx, U, Y, pp_list, np_list, nn_list = \
            graph_utils.prepare_graph_for_training(G, removal_perc)
            
print 'U', U 

print 'Y', Y
            
# print Gx.number_of_nodes(), Gx.number_of_edges()
            
plt.figure(2)
nx.draw_shell(Gx)
# plt.savefig('/Users/rockyrock/Desktop/2.eps', bbox_inches='tight')
# plt.savefig('/Users/rockyrock/Desktop/2.png', bbox_inches='tight')

####To extend the graph####
Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes)
original_degrees_list = graphsim.get_degrees_list(Gx, original_degrees)


# print Gx.number_of_nodes(), Gx.number_of_edges()
plt.figure(3)
nx.draw_shell(Gx)
# G_fancy = nx.relabel_nodes(Gx, {'col_0_1': 'a1', 'col_1_1': 'a2'})
# 
# pos = nx.shell_layout(G_fancy)
# nx.draw_networkx_nodes(G_fancy, pos, nodelist=['a1', 'a2'], node_color='b', node_size=500)
# nx.draw_networkx_nodes(G_fancy, pos, nodelist=[1,2,3,4,5,6], node_color='r', node_size=500)
# 
# nx.draw_networkx_edges(G_fancy, pos, 
#                        edgelist=G_fancy.edges(), 
#                        edge_color='k')
# nx.draw_networkx_edges(G_fancy, pos, 
#                        edgelist=[('a1', 1), ('a1', 3), ('a1', 6), ('a2', 2), ('a2', 4), ('a2', 5)], 
#                        edge_color='b')
# 
# labels={'a1': 'a1', 'a2': 'a2', 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
# 
# nx.draw_networkx_labels(G_fancy,pos,labels)
 

##########################


options = {}            
options['katz_h'] = 0.005

exp.test_local_unsupervised_methods(Gx, U, Y, node_to_index, options, original_degrees_list)

# exp.test_local_unsupervised_methods(Gx, U, Y, node_to_index, options)
plt.axis('off')
# plt.savefig('/Users/rockyrock/Desktop/3.eps', bbox_inches='tight')
# plt.savefig('/Users/rockyrock/Desktop/3.png', bbox_inches='tight')

plt.show()










