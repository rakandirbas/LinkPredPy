"""
Just to test filtering the Graph and its node attribute matrix from nodes that are not linked to anything.
"""

import testing_utils as tu
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_node(77)
G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(3,4)
G.add_edge(5,6)
G.add_node(7)
G.add_node(8)

plt.figure()

nx.draw_shell(G)

print G.number_of_nodes()

plt.savefig('/Users/rockyrock/Desktop/3.png', bbox_inches='tight')

print G.nodes()

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 77])

G, x = tu.filter_unlinked_nodes(G, x)

print "X", x

plt.clf()
nx.draw_shell(G)
plt.savefig('/Users/rockyrock/Desktop/4.png', bbox_inches='tight')

print G.nodes()

print 'done'