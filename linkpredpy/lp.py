import networkx as nx


class Link:
    def __init__(self, node_a, node_b, score):
        """
        :param node_a: the first node.
        :param node_b: the second node.
        :param score: the confidence score that a link exists between the two nodes.
        """
        self.node_a = node_a
        self.node_b = node_b
        self.score = score

    def get_node_a(self):
        """
        :return: the first node.
        """
        return self.node_a

    def get_node_b(self):
        """
        :return: the second node.
        """
        return self.node_b

    def get_score(self):
        """
        :return: the confidence score
        """
        return self.score

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "(%s, %s, %f)" % (self.node_a, self.node_b, self.score)


def cn(graph):
    """
    Common neighbours similarity index.
    :param graph: a graph to apply this similarity index on.
    :return: similarity matrix.
    """
    adjacency_matrix = nx.adj_matrix(graph)
    return adjacency_matrix**2


def get_missing_links(graph, similarity_index):
    """
    Returns the missing links in the graph and their confidence scores according to the specified similarity index.
    :param graph: a networkx graph whose missing links will be predicted.
    :param similarity_index: a function that return a similarity matrix for the graph.
    :return: a list of Links objects.
    """
    for index, name in enumerate(graph.nodes()):
        graph.node[name]['index'] = index

    similarity_matrix = similarity_index(graph)

    links = []
    for node1, node2 in nx.non_edges(graph):
        node1_index = graph.node[node1]['index']
        node2_index = graph.node[node2]['index']
        score = similarity_matrix[node1_index, node2_index]
        link = Link(node1, node2, score)
        links.append(link)

    links.sort(reverse=True)

    return links


def extend_graph(graph, missing_links, threshold):
    """
    Returns a graph composed by copying the original graph and adding the missing links to it.
    :param graph: the graph whose missing links to be added.
    :param missing_links: a list of missing links.
    :param threshold: the links whose scores above the specified threshold will be added.
    :return:
    """

    graph_copy = graph.copy()
    for link in missing_links:
        if link.score > threshold:
            graph_copy.add_edge(link.node_a, link.node_b)

    return graph_copy
