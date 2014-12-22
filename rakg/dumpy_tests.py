import graph_utils, graphsim
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

print 'hi'

G=nx.Graph()

G.add_edges_from([(1,2), (1,6),(2,3), (3,4), (4,5), (5,6), (3,6), (3,5)])

print G.nodes()

print G.degree(G.nodes())
 
X = np.array( [[1,0,0], [1,0,0], [1,0,0], [0,0,1], [0,1,0], [0,0,1]] )
 
G_ext, original_degrees = graph_utils.get_extended_graph(G, X)
 
# print G_ext.nodes()
# print original_degrees
# print G_ext.degree(G_ext.nodes())

plt.figure(1)
nx.draw_shell(G)
plt.figure(2)
nx.draw_shell(G_ext)
plt.show()
#  
# print graphsim.jacard( nx.adj_matrix(G_ext), graphsim.get_degrees_list(G_ext) )[0,2]
#  
# print graphsim.aa( nx.adj_matrix(G_ext), graphsim.get_degrees_list(G_ext, original_degrees) )[5,3]
# 
# print "test: ", 1.0/np.log10(2) + 1.0/np.log10(4) + 1.0/np.log10(3)






