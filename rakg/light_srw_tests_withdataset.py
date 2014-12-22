import numpy as np
import networkx as nx
from collections import defaultdict
import graph_utils
from light_srw import SRW
import srw
import testing_utils

def main():
    print 'hi'
    X = np.random.uniform(size=(8,3))
    G = None #networkx graph
    psi = GeneralPSI(G=G, X=X, k=10, delta=5)
    alpha=0.5
    srw_obj = SRW(psi=psi, alpha=alpha)
    srw_obj.optimize(iter=1)
    
    aucs = []
    for s in psi.get_testingS():
        P = srw_obj.get_P(s)
        s_index = psi.get_s_index(s)
        probs = srw.rwr(P, s_index, alpha)
        y_test = get_y_test(s, psi)
        auc = testing_utils.compute_AUC(y_test, probs)
        aucs.append(auc)
    
    aucs = np.array(aucs)
    print "\n\nAverage AUC:", np.mean(aucs)
    
def get_y_test(s, psi):
    """
    Parameters:
    s: the source id
    psi: the psi
    """
    D = psi.get_D(s)
    G = psi.source_nodes_data[s]['G']
    N = G.number_of_nodes()
    y_test = np.zeros((N))
    y_test[D] = 1
    
    return y_test
    

class GeneralPSI:
    def __init__(self, G, X, k=10, delta=5):
        """
        Parameters:
        -----------
        G: a networkx graph object.
        X: the nodes features matrix. Each row is the feature vector of a node.
        k: the number of neighbours a node must have in order to be considered a candidate source node.
        delta: the number of edges the candidate source node made in the future that close a triangle. 
               (i.e. the future/destination node is a friend of a friend, so delta is a threshold that sees how many
                 of these future/destination nodes are friends of current friends). A candidate source node
                 that has a degree above k, and made future friends above delta, then becomes a source node
                 for training.
        """
        
        self.G = G
        self.X = X
        self.k = k
        self.delta = delta
        self.num_features = self.X.shape[1]
        self.Global_node_to_index = {}
        
        for i, n in enumerate(self.G.nodes()):
            self.Global_node_to_index[n] = i
        
        degrees = self.G.degree( self.G.nodes() )
        degree_nodes = set()
        for node in degrees:
            d = degrees[node]
            if d >= k:
                degree_nodes.add(node)
                
        #degree_nodes are the candidate source nodes
        
        source_nodes = set()
        self.source_nodes_data = defaultdict(dict)
        print 'Finding the source nodes...'
        
        for i, node in enumerate(degree_nodes):
            G = self.G.copy()
            neighbours = G.neighbors(node)
            kth = len(neighbours)/2
            future_nodes = set( neighbours[kth+1:] )
            for fnode in future_nodes:
                G.remove_edge(node, fnode)
            
            node_1_neiborhood = set(G.neighbors(node))
            node_2_neiborhood = graph_utils.node_neighborhood(G, node, 2)
            node_2_neiborhood = set(node_2_neiborhood)
            
            counter = 0
            D_nodes = set()
            for fnode in future_nodes:
                if fnode in node_2_neiborhood:
                    counter += 1
                    D_nodes.add(fnode)
            
            node_to_index = {}
            index_to_node = {}
            L_nodes = set()
            
            if counter >= delta:
                source_nodes.add(node)
                for n in G.nodes():
                    if (n not in node_1_neiborhood) and (n not in node_2_neiborhood) and (n != node):
                        G.remove_node(n)
                    else:
                        if G.degree(n) == 0:
                            raise Exception("a node has zero degree!!!")
                
                for i, n in enumerate(G.nodes()):
                    node_to_index[n] = i
                    index_to_node[i] = n
                    
                for n in G.nodes():
                    if n not in D_nodes:
                        L_nodes.add(n)
                        
                s_index = node_to_index[node]  
                self.source_nodes_data[node] = \
                    {'G': G, 'D': D_nodes, 'L': L_nodes, 'index': s_index,
                     'node_to_index': node_to_index, 'index_to_node': index_to_node}
                    
        split = len(self.source_nodes_data.keys())
        split = split/2    
        self.training_S = self.source_nodes_data.keys()[0:split]
        self.testing_S = self.source_nodes_data.keys()[split:]
        
    def get_feature_vec(self, s, i, j):
        """
        Returns the feature vector between two nodes i, j with 
        respect to a source node s.
        
        Parameters:
        -----------
        s: the source node ID or something.
        i: the node i index in the adjacency matrix of the source node graph.
        j: the node j index in the adjacency matrix of the source node graph.
        """
        
        index_to_node = self.source_nodes_data[s]['index_to_node']
        node1 = index_to_node[i]
        node2 = index_to_node[j]
        node1_global_index = self.Global_node_to_index[node1]
        node2_global_index = self.Global_node_to_index[node2]
        v = np.concatenate( (self.X[node1_global_index], self.X[node2_global_index]) )
        return v
        
    def num_feats(self):
        """
        Returns the number of features in-use.
        """
        
        return self.num_features
        
    def get_D(self, s):
        """
        Returns a list of nodes indices that are the destinations for the source node.
        
        Parameters:
        -----------
        s: the source node ID.
        
        Returns:
        D: a list that represents the nodes indices (with respect to the source node graph)
         that are the destinations for the source node
        """
        
        D = self.source_nodes_data[s]['D']
        node_to_index = self.source_nodes_data[s]['node_to_index']
        D_ind = []
        
        for d in D:
            D_ind.append( node_to_index[d] )
        
        return D_ind
        
    def get_L(self, s):
        """
        Returns a list of nodes indices that are not destinations for the source node.
        
        Parameters:
        -----------
        s: the source node ID.
        
        Returns:
        D: a list that represents the nodes indices (with respect to the source node graph)
         that are the destinations for the source node
        """
        L = self.source_nodes_data[s]['L']
        node_to_index = self.source_nodes_data[s]['node_to_index']
        L_ind = []
        
        for l in L:
            L_ind.append( node_to_index[l] )
        
        return L_ind
        
    def get_S(self):
        """
        Returns a list of all source nodes IDs to use as training source nodes.
        """
        return self.training_S
    
    def get_testingS(self):
        """
        Returns a list of all source nodes IDs to use as testing source nodes.
        """
        return self.testing_S
        
    def get_A(self, s):
        """
        Returns the adjacency matrix of a source node s.
        It must always return a numpy array and NOOOT a matrix!!!
        Parameters:
        -----------
        s: the source node ID.
        """
        G = self.source_nodes_data[s]['G']
        return np.array(nx.adj_matrix(G))
        
        
    def get_s_index(self, s):
        """
        Returns the index of the source node with respect to its adjacency matrix.
        
        Parameters:
        -----------
        s: the source node ID
        
        Returns:
        s_index: the source node index in the adjacency matrix.
        """
        
        return self.source_nodes_data[s]['index']
    
    
if __name__ == '__main__':
    main()
    
    