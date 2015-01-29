"""
This module parses the Hep-th dataset and prepares it for use.
"""

import numpy as np
from collections import defaultdict
import networkx as nx
import datetime
import math
import text_utils
import graph_utils
import graphsim

path = "/home/rdirbas/datasets/hepth/"

# path = "/Users/rockyrock/Documents/workspace/Graph/data/hepth/"

def main():
    print 'Started...'
    
    parser = HepthParser()
    
    print 'Getting degrees...'
    degrees = parser.G.degree( parser.G.nodes() )
    
    k = 10
    delta = 5
    degree_nodes = set()

    for node in degrees:
        d = degrees[node]
        if d >= k:
            degree_nodes.add(node)
    
    print len(degree_nodes)
    
    print 'daaa'
    
    source_nodes = set()
    for i, node in enumerate(degree_nodes):
        G = parser.G.copy()
        tu, edges_list = parser.get_tu(node)
        kth = len(edges_list)/2
        future_nodes = set( edges_list[kth+1:] )

        for fnode in future_nodes:
            G.remove_edge(node, fnode[0])
            
        node_2_neiborhood = graph_utils.node_neighborhood(G, node, 2)
        
        counter = 0
        
        for fnode in future_nodes:
            if fnode[0] in node_2_neiborhood:
                counter += 1
        
        if counter >= delta:
            source_nodes.add(node)
            print 'node#: ', i, 'Degree: ', degrees[node] 
    
    print len(source_nodes)
        

class HepthParser:
    
    def __init__(self):
        authors, papers = get_authors_papers()
        self.coauthors = get_coauthors_links_and_coauthors()
        authored_links = set()
        
        self.G = nx.Graph() #The graph object
        self.authors_atts = defaultdict(dict) #a dict that holds the attributes for each author
        self.papers_atts = defaultdict(dict) #a dict that holds the attributes for each paper
        self.authors_papers = defaultdict(set) #a dict that holds the papers that each author wrote
        self.nodes = {} # dic that holds a mapping from a node index in adj_M to a node name
        
        for source in self.coauthors:
            self.G.add_node(source)
            for target in self.coauthors[source]:
                self.G.add_edge(source, target['coauthor'])
                
#         self.G.add_nodes_from(authors)#TODO to be removed
#                 
#         print self.G.number_of_nodes(), self.G.number_of_edges()
#         print len(authors), len(papers)
        
        with open(path + "O_attr_title_str.data", 'r') as f:
            for line in f:
                obj_id, obj_att = line.strip().split('\t')
                self.papers_atts[obj_id]['title'] = obj_att
                
        with open(path + "O_attr_authors_str.data", 'r') as f:
            for line in f:
                obj_id, obj_att = line.strip().split('\t')
                self.papers_atts[obj_id]['authors'] = obj_att        
        
        with open(path + "O_attr_slac_date.data", 'r') as f:
            for line in f:
                obj_id, obj_att = line.strip().split('\t')
                year, month, day = obj_att.split('-')
                d = datetime.date(int(year), int(month), int(day))
                self.papers_atts[obj_id]['date'] = d
        
        with open(path + "L_attr_linktype.data", 'r') as f:
            for line in f:
                id, val = line.strip().split('\t')
                if val == '\"Authored\"':
                    authored_links.add(id)
    
        with open(path + "links.data", 'r') as f:
            for line in f:
                link_id, source, target = line.strip().split('\t')
                if link_id in authored_links:
                    self.authors_papers[source].add(target)
                    
        with open(path + "O_attr_preferred_name.data", 'r') as f:
            for line in f:
                id, val = line.strip().split('\t')
                self.authors_atts[id]['name'] = val
                
        for i, node in enumerate(self.G.nodes()):
            self.nodes[i] = node
            
    def get_tu(self, node):
        """
        Returns the time tu which is the time when the node
        created its (ku/2)th edge
        
        Parameters:
        ------------
        node: the id of the node/author in the dataset
        
        Returns:
        ---------
        tu: a datetime object
        edges_list: a list of tuples for all the edges and their creation date that this node has
                of the form (coauthor_id, creation_date)
        """
        
        collabs = self.coauthors[node]
        node_papers = self.authors_papers[node]
        edges = {}
        
        for work in collabs:
            coauthor = work['coauthor']
            coauthor_papers = self.authors_papers[coauthor]
            for paper in node_papers:
                if paper in coauthor_papers:
                    paper_date = self.papers_atts[paper]['date']
                    if coauthor in edges:
                        old_paper_date = edges[coauthor]
                        if paper_date < old_paper_date:
                            edges[coauthor] = paper_date
                    else:
                        edges[coauthor] = paper_date
        
        
        sorted_edges = sorted(edges, key=edges.__getitem__)
        
        edges_list = []
        
        for author in sorted_edges:
            edges_list.append( ( author, edges[author] ) )
        
        kth = len(edges_list)/2
        tu = edges_list[kth][1]
        return tu, edges_list
            
    def get_feature_vec(self, source, source_tu, cn, node1, node2):
        """
        Returns a feature vector for two nodes, given the source
        node, s, that we are calculating the features values with respect to.
        
        Parameters:
        ------------
        source: the id of the source node
        source_tu: the time when the source created its k/2th edge
        cn: the number of common neighbors between the source node and node2.
        node1: the id of the first node in the dataset.
        node2: the id of the second node in the dataset.
        
        Returns:
        --------
        vector: a numpy array that has 6 elements/[features values].
        """
#         node1 = self.nodes[node1_index]
#         node2 = self.nodes[node2_index]
        feats_num = 6 #number of features
        vector = np.zeros((feats_num))
        normalizer = 10000.0
        
        node1_papers = self.authors_papers[node1]
        node2_papers = self.authors_papers[node2]
        
        count = 0
        for paper in node1_papers:
            paper_date = self.papers_atts[paper]['date']
            if paper_date < source_tu:
                count += 1
        
        vector[0] = count/normalizer #adding the first feature value
        
        count = 0
        for paper in node2_papers:
            paper_date = self.papers_atts[paper]['date']
            if paper_date < source_tu:
                count += 1   
                
        vector[1] = count/normalizer #adding the second feature value
        
        common_papers = set()
        if len(node1_papers) < len(node2_papers):
            for paper in node1_papers:
                if paper in node2_papers:
                    common_papers.add(paper)
        else:
            for paper in node2_papers:
                if paper in node1_papers:
                    common_papers.add(paper)
                    
        vector[2] = len(common_papers)/normalizer #adding the third feature value
        
        text1 = ""
        text2 = ""
        
        for paper in node1_papers:
            paper_title = self.papers_atts[paper]['title']
            text1 += paper_title + " "
            
        for paper in node2_papers:
            paper_title = self.papers_atts[paper]['title']
            text2 += paper_title + " "
            
        vector[3] = text_utils.get_cosine(text1, text2) #adding the fourth feature value
        
        tt = datetime.date(2003,1,1)
        
        if len(common_papers) > 0:
            paper = common_papers.pop()
            t = self.papers_atts[paper]['date']
            
            for paper in common_papers:
                t2 = self.papers_atts[paper]['date']
                if t2 > t:
                    t = t2
        else:
            t = datetime.date(1992,1,1)
        
                
        
        diff = tt - t
        
        vector[4] = diff.days/normalizer #adding the fifth feature value
        
        vector[5] = cn/normalizer #adding the sixth feature value
                
        return vector
    
def get_authors_papers():
    authors = set()
    papers = set()
    f = open(path + "O_attr_objecttype.data", 'r')
    
    for line in f:
        obj_id, obj_type = line.split('\t')
        obj_type = obj_type.strip()
        if obj_type == "\"Author\"":
            authors.add(obj_id)
        elif obj_type == "\"Paper\"":
            papers.add(obj_id)
            
    f.close()
    
    return authors, papers

def get_coauthors_links_and_coauthors():
    coauthors_links = set()
    coauthors = defaultdict(list)
    coauthred_papers = {}
    
    f = open(path + "L_attr_linktype.data", 'r')
    
    for line in f:
        link_id, link_type = line.split('\t')
        link_type = link_type.strip()
        if link_type == "\"Co-Authored\"":
            coauthors_links.add(link_id)
    
    f.close()
    
    #####
    with open(path + "L_attr_paper.data", 'r') as f:
            for line in f:
                link_id, val = line.strip().split('\t')
                coauthred_papers[link_id] = val
    #####
    
    f = open(path + "links.data", 'r')
    
    for line in f:
        link_id, source, target = line.strip().split('\t')
        if link_id in coauthors_links:
            work = {}
            work['coauthor'] = target
            work['paper'] = coauthred_papers[link_id]
            coauthors[source].append(work)
    f.close()
    
    
    return coauthors


class HepthPSI:
    def __init__(self, k=10, delta=5):
        self.num_features = 6
        
        self.parser = HepthParser()
        degrees = self.parser.G.degree( self.parser.G.nodes() )
        degree_nodes = set()
        for node in degrees:
            d = degrees[node]
            if d >= k:
                degree_nodes.add(node)
                
        source_nodes = set()
        self.source_nodes_data = defaultdict(dict)
        print 'Finding the source nodes...'
        
        for i, node in enumerate(degree_nodes):
            G = self.parser.G.copy()
            tu, edges_list = self.parser.get_tu(node)
            kth = len(edges_list)/2
            future_nodes = set( edges_list[kth+1:] )
            
            for fnode in future_nodes:
                G.remove_edge(node, fnode[0])
                
            node_1_neiborhood = set(G.neighbors(node))
            node_2_neiborhood = graph_utils.node_neighborhood(G, node, 2)
            node_2_neiborhood = set(node_2_neiborhood)
            
            counter = 0
            D_nodes = set()
            for fnode in future_nodes:
                if fnode[0] in node_2_neiborhood:
                    counter += 1
                    D_nodes.add(fnode[0])
            
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
                A = nx.adj_matrix(G)
                CN = graphsim.cn(A)      
                self.source_nodes_data[node] = \
                    {'G': G, 'D': D_nodes, 'L': L_nodes, 'index': s_index, 'tu': tu,
                        'node_to_index': node_to_index, 'index_to_node': index_to_node,
                        'CN': CN}
                
                    
        split = len(self.source_nodes_data.keys())
        split = split/2    
        self.training_S = self.source_nodes_data.keys()[0:split]
        self.testing_S = self.source_nodes_data.keys()[split:]
                    
                
            #if this node is a source node then create:
                #1-its graph
                #2-its D nodes
                #3-its L nodes
                #4-its features matrix
        
    def num_feats(self):
        return self.num_features
    
    def get_S(self):
        return self.training_S
    
    def get_testingS(self):
        return self.testing_S
    
    def get_D(self, s):
        D = self.source_nodes_data[s]['D']
        node_to_index = self.source_nodes_data[s]['node_to_index']
        D_ind = []
        
        for d in D:
            D_ind.append( node_to_index[d] )
        
        return D_ind
    
    def get_L(self, s):
        L = self.source_nodes_data[s]['L']
        node_to_index = self.source_nodes_data[s]['node_to_index']
        L_ind = []
        
        for l in L:
            L_ind.append( node_to_index[l] )
        
        return L_ind
    
    def get_A(self, s):
        """
        it must always return a numpy array and NOOOT a matrix!!!
        """
        G = self.source_nodes_data[s]['G']
        return np.array(nx.adj_matrix(G))
    
    def get_s_index(self, s):
        return self.source_nodes_data[s]['index']
    
    def get_feature_vec(self, s, i, j):
        index_to_node = self.source_nodes_data[s]['index_to_node']
        CN = self.source_nodes_data[s]['CN']
        
        node1 = index_to_node[i]
        node2 = index_to_node[j]
        tu = self.source_nodes_data[s]['tu']
        vec = self.parser.get_feature_vec(s, tu, CN[i,j], node1, node2)
        
        return vec

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    

























