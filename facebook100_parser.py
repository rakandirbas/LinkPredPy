import numpy as np
import scipy as sp
import scipy.io
from pprint import pprint
from sklearn.feature_extraction import DictVectorizer
import networkx as nx
import graph_utils as gut

node_attributes = ["student_fac", "gender", "major_index", "second_major", "dorm", "year", "high_school"]

attribute_dict = {
    "student_fac" : 0,
    "gender" : 1,
    "major_index" : 2,
    "second_major" : 3,
    "dorm" : 4,
    "year" : 5,
    "high_school" : 6,
    }

def main():
    path = "/Volumes/BigHD/Datastore/Networks/FB100/facebook100/Caltech36.mat"
    read_mat_file(path)

def parse(file_name, with_attributes=True):
    """
    Parses a dataset file.
    
    Returns
    -------
    :type G: networkx
    :param G: networkx G graph object
    
    :type F: numpy 2D array
    :param F: nodes features matrix 
    """
    return read_mat_file(file_name, with_attributes)

def read_mat_file(matlab_filename, with_attributes=True):
    network_name = matlab_filename.strip(".").strip("/").split("/")[-1].split(".")[0]
    print "Now parsing " + network_name
    matlab_object = scipy.io.loadmat(matlab_filename)
    scipy_sparse_graph = matlab_object["A"]
    attributes = matlab_object["local_info"] #A.shape[0] * num_feat
    G = nx.from_scipy_sparse_matrix(scipy_sparse_graph)
    
    if with_attributes == False:
        return G
    
    samples = []
    for row in attributes:
        row = row.astype('str')
        d = dict(zip(node_attributes, row))
        samples.append(d)
        
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(samples)
    print 'done'
#     print X[0]
    return G, X


def add_nodes_features(Nodes_X, data, U, nodes_list, raw_features):
    
    if not raw_features:
        nodes = {}
        for i, node in enumerate(nodes_list):
            nodes[node] = i
        
        N = len(U)
        
        F = []
        
        for i, edge in enumerate(U):
            node1 = edge[0]
            node2 = edge[1]
            node1_index = nodes[node1]
            node2_index = nodes[node2]
            node1_feats = Nodes_X[node1_index].astype('int')
            node2_feats = Nodes_X[node2_index].astype('int')
            f = np.bitwise_and(node1_feats, node2_feats)
            F.append(f)
        
    #     F = np.array(F)
        
        if data != None:
            data = np.column_stack([data, F])
        else:
            data = np.array(F)
     
        return data
    else:
        return gut.add_raw_feature(data, U, Nodes_X, nodes_list)
    

    
# main()