import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from rakg import graph_utils
import os
from rakg import datasets_stats
from rakg import datasets

# path = "/Users/rockyrock/Desktop/facebook/"
# path = "/Users/rockyrock/Desktop/gplus/"

# 
# edges_files = []
#     
# for file in os.listdir(path):
#     fileName, fileExtension = os.path.splitext(path+file)
#     if fileExtension == '.edges':
#         edges_files.append(fileName+fileExtension)
#              
# for file in edges_files:
#     G = graph_utils.read_graph(file)
#     if G.number_of_nodes() > 200 and G.number_of_nodes() < 1000:
#         print file, G.number_of_nodes(), G.number_of_edges()
        
dataset_path = "/Users/rockyrock/Desktop/facebook/1912.edges"
# dataset_path = "/Users/rockyrock/Desktop/gplus/101263615503715477581.edges"
G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
#   
print datasets_stats.srw_stats(G, X_nodes, k=50, delta=50)