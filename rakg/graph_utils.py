import networkx as nx
from timer import Timerx
from itertools import izip
import numpy as np
from sklearn.utils import shuffle
import itertools as IT
import random
import graphsim as gsim


def read_graph(file_path, delimiter=' ', convert_string_to_int = True, directed=False):
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    with open(file_path) as edges_file:
        for line in edges_file:
            source, target = line.rstrip().split(delimiter)
            if (convert_string_to_int):
                G.add_edge(int(source), int(target))
            else:
                G.add_edge(source, target)
    
    return G

def build_fast_U(nodes_list):
    """
    Using a fast routine.
    Returns a list U which is the universal edges set that contains 
    every possible edge that can be in the graph. 
    The size of this set N*(N-1)/2 where N is the number of nodes/vertices.
    """
    U = [pair for pair in IT.combinations(nodes_list, 2)]
    return np.array(U)

def build_U(nodes_list):
    """
    Returns a list U which is the universal edges set that contains 
    every possible edge that can be in the graph. 
    The size of this set N*(N-1)/2 where N is the number of nodes/vertices.
    """
    
#     N = len(nodes_list)
#     U_size = (N*(N-1))/2
#     U = np.zeros((U_size,2))
#     
#     j = 0
#     for i, node1 in enumerate(nodes_list):
#         for node2 in nodes_list[i+1:]:
#             edge = (node1,node2)
#             U[j] = edge
#             j += 1
            
    
    return build_fast_U(nodes_list)

def build_small_U(G):
    """
    Returns a down sampled list U, where U is a list that contains tuples of edges.
    For example U[0] = (node_i, node_j). Down sampled means here is that U contains 
    the same number of (edges that truly exists) AND (edges that don't exist) ... (i.e. positive
    and negative examples for training).
    
    Parameters:
    -----------
    G: a networkx graph object.
    
    Returns:
    --------
    U: a list of edges.
    """
    U = G.edges()
    sample_size = len(U)
    
    A = nx.adj_matrix(G)
    A = np.array(A)
    
    nonedges = np.transpose(np.nonzero(A == 0))
    nonedges = shuffle(nonedges, random_state=0)
    
    index_to_node = {}
    for i, node in enumerate(G.nodes()):
        index_to_node[i] = node
    
    for x, (i, j) in enumerate(nonedges):
        if x < sample_size:
            node1 = index_to_node[i]
            node2 = index_to_node[j]
            U.append( (node1, node2) )
        else:
            break
        
    return U
        
    

def build_Y(G, U):
    """
    Builds the Y ground-truth list that contains a label
    for each edge in the universal to indicate if the edge
    really exists in the graph or not.
    
    Parameters:
    G: the Networkx graph.
    U: the universal edges set that contains every possible edge in the graph.
    """
    N = len(U)
    Y = np.zeros(N)
    
    for i, edge in enumerate(U):
        node1 = edge[0]
        node2 = edge[1]
        if G.has_edge(node1,node2):
            Y[i] = 1
            
#     Y = list()
#     
#     for edge in U:
#         node1 = edge[0]
#         node2 = edge[1]
#         if G.has_edge(node1,node2):
#             Y.append(1)
#         else:
#             Y.append(0)
    
    return Y

def add_feature(data, U, S, nodes_list):
    """
    Extends the data matrix with a column that represents 
    a new feature.
    
    Parameters:
    data: the data matrix (usually called X in sci-kit).
    U: the universal edges list (set).
    S: the similarity matrix that holds the similarities between each
       node in the graph.
    nodes_list: the list of nodes from a networkx Graph (returned by the G.nodes() method). 
       
    Returns the data extended with a new column.
    """
    
    nodes = {}
    for i, node in enumerate(nodes_list):
        nodes[node] = i
    
    N = len(U)
    feature = np.zeros(N)
    
    for i, edge in enumerate(U):
        node1 = edge[0]
        node2 = edge[1]
        node1_index = nodes[node1]
        node2_index = nodes[node2]
        feature[i] = S[node1_index, node2_index]

    if data == None:
        data = np.reshape(feature, (-1,1))
    else:
        data = np.column_stack([data, feature])

    return data

def add_raw_feature(data, U, M, nodes_list, node_to_index=None):
    """
    Extends the 'data' matrix (X in scikit) with raw features from M.
    So each dyad (i,j) features is just the concatination of M[i] and M[j]
    
    Parameters
    ----------
    data: features numpy matrix.
    U: the universal edges set.
    M: the raw features set (this is most likely to be the adjacency matrix)
    nodes_list: a list of all nodes in the graph

    Returns
    -------
    data: the features numpy matrix extended with columns
    """
    if node_to_index == None:
        nodes = {}
        for i, node in enumerate(nodes_list):
            nodes[node] = i
    else:
        nodes = node_to_index
    
    N = len(U)
    C = M.shape[1] #num of columns
    feature = np.zeros((N, C*2)) #number dyads times the number of double coulmns
    
    for i, edge in enumerate(U):
        node1 = edge[0]
        node2 = edge[1]
        node1_index = nodes[node1]
        node2_index = nodes[node2]
        feature[i] = np.concatenate(( M[node1_index], M[node2_index] ))
        
    if data == None:
        data = feature
    else:
        data = np.column_stack([data, feature])
        
    return data


def add_local_topo_features(X, A, U, degrees, nodes, original_degrees_list=None):
    """
    Computes and adds the local topological features to the design matrix X.
    
    Parameters:
    X: the design matrix (it can be None).
    A: the adjacency matrix of the graph.
    U: the list of the training dyads.
    degrees: a list that holds the degree of each node, so degrees[0] 
        is the degree of the node that has id 0.
    nodes: the list of all nodes in the graph.
    original_degrees_list: *list* of the degrees of the nodes before the graph gets extended.
                    This is to be used if the features to be computed while using an extended
                    graph. Of course it should include the attribute nodes degrees as well.
                    If omited, then the features will be computed for the un-extended graph.
    
    """
    X = add_feature(X, U, gsim.cn(A), nodes)
    X = add_feature(X, U, gsim.lp(A), nodes)
    X = add_feature(X, U, gsim.salton(A, degrees), nodes)
    X = add_feature(X, U, gsim.jacard(A, degrees), nodes)
    X = add_feature(X, U, gsim.sorensen(A, degrees), nodes)
    X = add_feature(X, U, gsim.hpi(A, degrees), nodes)
    X = add_feature(X, U, gsim.hdi(A, degrees), nodes)
    X = add_feature(X, U, gsim.lhn1(A, degrees), nodes)
    X = add_feature(X, U, gsim.pa(A, degrees), nodes)
    
    if original_degrees_list == None:
        X = add_feature(X, U, gsim.aa(A, degrees), nodes)
        X = add_feature(X, U, gsim.ra(A, degrees), nodes)
    else:
        X = add_feature(X, U, gsim.aa(A, original_degrees_list), nodes)
        X = add_feature(X, U, gsim.ra(A, original_degrees_list), nodes)
    return X

def add_global_features(G, X, U, node_to_index, params):
    """
    Adds the global features to the design matrix.
    
    Parameters:
    -----------
    G: networkx graph.
    X: the design matrix.
    U: a list of dyads.
    node_to_index: a dictionary that maps from node name to node index.
    params: needed parameters. Here we need the following: params['rwr_alpha'], 
                                params['lrw_nSteps'].
                                
    Returns:
    ---------
    X_new: the design matrix with added columns.
    """
    A = np.array( nx.adj_matrix(G) )
    rwr_alpha = params['rwr_alpha']
    lrw_nSteps = params['lrw_nSteps']
    katz_h = gsim.katz_h(A)
    katz_h = katz_h * 0.1
    katz_p = gsim.predict_scores(U, gsim.katz(A, katz_h), node_to_index)
    
    rwr_p = gsim.RWR_Clf(A, rwr_alpha).score(U, node_to_index) 
    lrw_p = gsim.LRW_Clf(A, lrw_nSteps, G.number_of_edges()).score(U, node_to_index)
    srw_p = gsim.SRW_Clf(A, lrw_nSteps, G.number_of_edges()).score(U, node_to_index)
    
    
    if X == None:
        X = np.reshape(katz_p, (-1,1))
    else:
        X = np.column_stack([X, katz_p])
    
    X = np.column_stack([X, rwr_p])
    X = np.column_stack([X, lrw_p])
    X = np.column_stack([X, srw_p])
    
#     btwness = nx.betweenness_centrality(G)
#     N = len(U)
#     feature = np.zeros(N)
#     
#     for i, (node1, node2) in enumerate(U):
#         feature[i] = btwness[node1] + btwness[node2]
#         
#         
#     X = np.column_stack([X, feature])
    
    
    return X

def node_neighborhood(G, node, n):
    """
    Returns a list of nodes which are the n-neighborhood of the input node.
    
    Parameters
    ----------
    G: networkx graph object.
    node: the node to get the neighborhood for.
    n: the neighborhood degree.
    """
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.iteritems()
                    if length == n]

def build_Un_neighborhoods(G, U, lengths_list):
    """
    Builds U_n the set of all possible links for nodes that are within 
    a specified neighborhood length n.
    
    Parameters
    ----------
    G: networkx graph object.
    U: the set of all possible links.
    lengths_list: a list that holds the neighborhoods degrees to compute.
    
    Returns
    -------
    a dictionary mapped by neighborhood degree. The values are numpy 2D arrays
     where each vector is of the form [v_i, v_j]
    to indicate that v_i has a shortest path of length n to v_j
    """
    
    U_n = {}
    for n in lengths_list:
        U_n[n] = []
        
    paths_lengths = nx.shortest_path_length(G)
    
    for edge in U:
        node1 = edge[0]
        node2 = edge[1]
        if paths_lengths[node1].has_key(node2):
            plength = paths_lengths[node1][node2]
        else:
            plength = 0
        if  U_n.has_key(plength):
            U_n[plength].append(edge)
    
    for k, v in U_n.iteritems():
        U_n[k] = np.array(v)
    
    return U_n
    
    
class DyadFeaturesIndexLocator:
    """
    Returns the index of a given dyad in the 
    dyads features matrix.
    """
    
    def __init__(self, U, nodes_list):
        """
        U: the set of all possible edges/dyads
        node_list: the list of all nodes in the graph
        """
        
        self.nodes = {}
        for i, node in enumerate(nodes_list):
            self.nodes[node] = i
        
        self.M = {}
        for i, edge in enumerate(U):
            node1 = edge[0]
            node2 = edge[1]
            node1_index = self.nodes[node1]
            node2_index = self.nodes[node2]
            self.M[ (node1_index, node2_index) ] = i
    
    def get_feature_index(self, node1_index, node2_index):
        """
        Returns the index of a given dyad in the
        dyads features matrix.
        
        Parameters:
        -----------
        node1_index: the index of the first node 
        node2_index: the index of the secod node
        """
        if (node1_index, node2_index) in self.M:
            index = self.M[ (node1_index, node2_index) ]
        elif (node2_index, node1_index) in self.M:
            index = self.M[ (node2_index, node1_index) ]
        else:
            raise Exception("I can't find an index for the given\
             dyad in the dyad feature matrix!!!")
        return index
    
    
def ttt():
    U = [(1,2), (1,3), (2,3)]
    nodes_list = [1,2,3]
    X = [12,13,23]
    locator = DyadFeaturesIndexLocator(U, nodes_list)
    index =  locator.get_feature_index(0,1)
    print X[index]
    
def get_extended_graph(G, X, bins=None):
    """
    Extends the networkx graph G with nodes that represent attributes and 
    links between the attribute nodes and normal nodes if the normal nodes have
    the attribute values. A Real-valued attribute is divided into bins where each bin is a node, and 
    a link between a normal node and a bin node is created if the normal node has an attribute value 
    that falls within the bin node range.

    Similarly each binary attribute is represented by two nodes, a node that represents a value 1 and 
    another that represents a value 0.
    
    Parameters:
    -----------
    G: a networkx graph object.
    X: a design matrix where columns are attributes and each row represents a node.
       So each row is the feature vector of a node.
       
    Returns:
    ----------
    extended_G: returns an extended version of G with attributes nodes and the 
               links between them and normal nodes. (the original graph object is untouched)
    original_degrees: a dictionary of the degrees of the nodes before extending the graph, but 
                    this dictionary also holds the degrees of the attributes nodes after
                    extending the graph. This is basically to be used to calcualted Adamic Adar 
                    and Resource Allocation indices.
    """
    G = G.copy()
    if bins == None:
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bins = np.array(bins)
    
    index_to_node = {} #to map from index to node name
    for i, node in enumerate(G.nodes()):
        index_to_node[i] = node

    n_rows = X.shape[0]
    n_cols = X.shape[1]

    added_nodes = []
    original_degrees = G.degree(G.nodes())
    
    for col in xrange(n_cols):
        #check if X[:, col] is binary or real
        v = (X[:,col] == 1)

        num_ones = np.where(v)[0].size
        
        v = (X[:,col] == 0)
        
        num_zeros = np.where(v)[0].size
        
        total_size = num_ones + num_zeros

        is_binary = False
        
        if n_rows == total_size:
            is_binary = True
            
        if is_binary:
#             node0 = "col_" + str(col) + "_0" 
            node1 = "col_" + str(col) + "_1" 
            
#             zeroval_nodes_inds =  np.where((X[:,col] == 0))[0] #returns the indices of the nodes that has the attribute value as 0
            onesval_nodes_inds =  np.where((X[:,col] == 1))[0] #returns the indices of the nodes that has the attribute value as 1

#             nodes_linked_to_zero = [ index_to_node[node_index] for node_index in  zeroval_nodes_inds ]
            nodes_linked_to_one = [ index_to_node[node_index] for node_index in  onesval_nodes_inds ]
            
#             zeros_edges = [ (node0, x) for x in nodes_linked_to_zero ]
            ones_edges = [ (node1, x) for x in nodes_linked_to_one ]

#             if len(zeros_edges) > 0:
#                 G.add_edges_from(zeros_edges)
                
            if len(ones_edges) > 0:
                G.add_edges_from(ones_edges)
                added_nodes.append(node1)
        else:
            column = X[:,col]
            binplace = np.digitize(column, bins)#each element in binplace is the index of the bin that this element falls into.
            for bin in xrange(bins.size):
                elements_inds = np.where(binplace==bin)[0]
                nodes_linked_to_bin = [ index_to_node[node_index] for node_index in  elements_inds ]
                node_bin = "col_" + str(col) + "_" + str(bin)
                edges =  [ (node_bin, x) for x in nodes_linked_to_bin ]
                if len(edges) > 0:
                    G.add_edges_from(edges)


    added_nodes_degrees = G.degree(added_nodes)
    original_degrees.update(added_nodes_degrees)

    return G, original_degrees

def prepare_graph_for_training(G, removal_perc, undersample=False, random_state = 0):
    """
    Returns the graph used for learning, testing set and the corresponding labels
    used to check the performance of un-supervised methods.
    It also returns the lists of dyads that are postive-to-positive,
    negative-to-positive, negative-to-negative. Those lists represent 
    the training and testing scenario.
     
    Basically this methods removes a percentage of the edges in the graph.
    Then this graph will be used to compute the similarity matrix. 
    The removed dyads will be the set of positive examples. 
    The dyads that don't represent links in the new graph and not part of the removed dyads
    are considered negative examples.
    
    Parameters:
    ------------
    G: networkx graph.
    removal_perc: the percentage of edges to remove as testing set. (highest is 1.0 and lowest 0.0)
    undersample: to either reduce the number of negative examples and make them equal to the number 
            of positive examples or not.
    
    Returns:
    ------------
    Gx: networkx graph where a percentage of its edges were removed.
    U: a list of dyads that are used as a test set.
    Y: a list of labels for the dyads.
    pp_list: list of dyads that were positive and remained positive
    np_list: list of dyads that were negative and become positive
    nn_list: list of dyads that were negative and remained negative
    """
    random.seed(random_state)
    
    G = G.copy()
    pp_list = [] # list of dyads that were positive and remained positive
    np_list = [] # list of dyads that were negative and become positive
    nn_list = [] # list of dyads that were negative and remained negative
    
    n_edges = G.number_of_edges()
    n_nodes = G.number_of_nodes()
    n_dyads = (n_nodes*(n_nodes-1))/2

    np_size = np.ceil(removal_perc * n_edges)
    pp_size = n_edges - np_size
    nn_size = n_dyads - n_edges
    
    if not undersample:
        nn_list = [pair for pair in IT.combinations(G.nodes(), 2)
               if not G.has_edge(*pair)]
    else:
        nn_list = get_undersampled_nn(G, pp_size)
    
    removed_edges = random.sample( G.edges(), int(np_size) )
    G.remove_edges_from(removed_edges)
    
    np_list = np.array(removed_edges)
    pp_list = np.array(G.edges())
    nn_list = np.array(nn_list)
    
    U = np.vstack((np_list, nn_list))
    Y = np.zeros( len(np_list) + len(nn_list) )
    Y[0:len(np_list)] = 1
    
    return G, U, Y, pp_list, np_list, nn_list


def get_node_to_index(G):
    """
    Returns a dictionary that maps from node name to node index
    
    Parameters:
    -----------
    G: networkx graph.
    """
    node_to_index = {}
    for i, node in enumerate(G.nodes()):
        node_to_index[node] = i
    return node_to_index


def get_undersampled_nn(G, pp_size):
    """
    Returns a the list nn that contains an undersampled list of negative dyads.
    """
    
    nn_list = []
    sample_size = pp_size * 2
    
    A = nx.adj_matrix(G)
    A = np.array(A)
    
    nonedges = np.transpose(np.nonzero(A == 0))
    nonedges = shuffle(nonedges, random_state=0)
    
    index_to_node = {}
    for i, node in enumerate(G.nodes()):
        index_to_node[i] = node
    
    for x, (i, j) in enumerate(nonedges):
        if x < sample_size:
            node1 = index_to_node[i]
            node2 = index_to_node[j]
            nn_list.append( (node1, node2) )
        else:
            break
        
    return nn_list



