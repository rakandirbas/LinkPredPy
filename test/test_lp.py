import unittest
from linkpredpy import lp
import networkx as nx

class TestBuildMissingDyads(unittest.TestCase):

    def test_missing_dyads_are_calculated_correctly(self):
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edges_from([(1, 3), (2, 3), (2, 5)])

        missing_dyads = lp.build_missing_dyads(G)
        self.assertEquals(3, len(missing_dyads))
        self.assertTrue((1, 2) in missing_dyads)
        self.assertTrue((1, 5) in missing_dyads)
        self.assertTrue((3, 5) in missing_dyads)

if __name__ == '__main__':
    unittest.main()