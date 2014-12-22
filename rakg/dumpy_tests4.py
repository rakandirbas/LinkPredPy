"""
This test to check the NaN errors with the deep learning experiments
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
 
G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
G = testing_utils.filter_unlinked_nodes(G)
node_to_index = testing_utils.get_node_to_index(G)
removal_perc = 0.20

