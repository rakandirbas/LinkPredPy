"""
This module contains functions that loads datasets contained in the framework.
"""

import numpy as np
import networkx as nx
import graph_utils
import os

on_server = False

def main():
    print 'hi'
    
#     G, Nodes_X = load_prot_prot()
#     
#     print G.number_of_nodes(), G.number_of_edges()
#     print Nodes_X.shape

#     path = "/Users/rockyrock/Desktop/SNAP/facebook/facebook/0.edges"
#     G, Nodes_X = load_SNAP_dataset(path)
#     print G.number_of_nodes(), Nodes_X.shape
     
    print 'done :)'

def load_prot_prot(on_server = False):
    """
    Parses the protein protein interaction dataset
    
    Returns
    -------
    :type G: networkx
    :param G: networkx G graph object
    
    :type F: numpy 2D array
    :param F: nodes features matrix
    """
    
    local_path = '/Users/rockyrock/Documents/University/Konstanz/Lectures/'+\
        'Master thesis/Datasets/MatrixFact/Protein-protein interaction data/ext/'
    server_path = '/home/rdirbas/datasets/prot-prot/'#always append with a /
    path = ''
    
    if on_server:
        path = server_path
    else:
        path = local_path
        
    # read the first line to determine the number of columns    
    with open(path+'Adj', 'rb') as f:
        ncols = len(next(f).split('\t'))
        
    x = np.genfromtxt(path+'Adj', delimiter='\t', dtype=None, names=True,
                  usecols=range(1,ncols) # skip the first column
                  )
    labels = x.dtype.names
    
    y = x.view(dtype=('int', len(x.dtype)))# y is a view of x
    G = nx.from_numpy_matrix(y, create_using=nx.DiGraph())
    G = nx.relabel_nodes(G, dict(zip(range(ncols-1), labels)))
    
    #Now read the node features matrix
    with open(path+'Atts', 'rb') as f:
        ncols = len(next(f).split('\t'))
    
    x = np.genfromtxt(path+'Atts', delimiter='\t', dtype=None, names=True,
                  usecols=range(1,ncols) # skip the first column
                  )
    Nodes_X = x.view(dtype=('int', len(x.dtype)))
    
    return G, Nodes_X
    

def load_SNAP_dataset(file_path, directed=False):
    """
    Loads a network from SNAP. Note that the file of the node features
    must exist in the same folder. So if the file name is '0.edges', then there must
    be a file for the node features called '0.feat'
    
    Parameters:
    ------------
    file_path: the path to a '.edges' file
    directed: is the graph directed
    
    Returns:
    --------
    G: a networkx graph.
    Nodes_X: the nodes attributes features matrix.
    """
    G = graph_utils.read_graph(file_path, directed=directed)
    base_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_name)[0]
    
    feats_file_name = file_name + '.feat'
    feats_file_path = os.path.dirname(file_path) + "/" + feats_file_name
    
    if os.path.exists(feats_file_path):
        X = np.loadtxt(feats_file_path, delimiter=' ' )
        node_name_to_index = {}
        
        
        ####
#         X = np.array( X[:, 1:], dtype=np.int )
        X = X[:, 1:]
        
        
        nid = []
           
        with open(feats_file_path) as f_file:
            for line in f_file:
                row = line.rstrip().split(' ')
                nid.append( int(row[0]) )
        
        nid = np.array(nid)
        nid = nid.reshape(-1,1)
        X = np.hstack( (nid, X) )
        ####
        
        for i in xrange( len(X) ):
            if node_name_to_index.has_key(X[i,0]):
                print X[i,0]
                raise Exception("There is a duplicate feature vector for a node!")
            else:
                node_name_to_index[ X[i,0] ] = i
        
        Nodes_X = []
        for node_name in G.nodes():
            index = node_name_to_index[node_name]
            node_feats = X[index, 1:]  
            Nodes_X.append(node_feats)
        Nodes_X = np.array(Nodes_X)
        
    else:
        raise Exception("The features file for the snap network doesn't exist")
    
    if G.number_of_nodes() != Nodes_X.shape[0]:
        raise Exception("The node feature matrix have less or more rows than \
         the number of nodes in the network")
    
    return G, Nodes_X
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    