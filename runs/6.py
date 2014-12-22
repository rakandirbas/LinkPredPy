import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import networkx as nx
import numpy as np
from rakg import graphsim, graph_utils,\
     light_srw, matrix_fact, supervised_methods, \
     testing_utils, timer, srw, facebook100_parser, datasets_stats,\
     datasets
     
     
print 'Started ...'
file_path = '/Users/rockyrock/Desktop/matlab-format/Rochester38.csv'
file_path2 = '/Users/rockyrock/Desktop/Bucknell39.csv'
# G = facebook100_parser.parse(file_path, with_attributes=False)
G = graph_utils.read_graph(file_path2)

# G = nx.read_graphml('/Users/rockyrock/Desktop/Rochester38.graphml')

print G.number_of_nodes(), G.number_of_edges()

print nx.diameter(G)


# nx.write_graphml(G, '/Users/rockyrock/Desktop/Rochester38.graphml')