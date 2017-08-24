import networkx as nx
import numpy as np
import scipy
import itertools as IT


def build_missing_dyads(graph):
    """
    :param graph: the graph whose missing dyads will be returned.
    :type graph: networkx graph.
    :return: a list containing the missing dyads in the graph.
    """
    return [pair for pair in IT.combinations(graph.nodes(), 2)
               if not graph.has_edge(*pair)]

def cn(graph):
    """
    Common neighbours similarity index.
    :param graph: a graph to apply this similarity index on.
    :type graph: networkx graph.
    :return:
    """
    adjacency_matrix = nx.adj_matrix(graph)
    cn_matrix = adjacency_matrix**2

