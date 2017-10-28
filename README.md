LinkPredPy
==========
A gentle link prediction toolkit in python.


# Example

```python
import networkx as nx
from linkpredpy import lp

G = nx.fast_gnp_random_graph(5, 0.5) # create a random graph.
links = lp.get_missing_links(G, lp.cn) # use common neighbours as a similarity index.

for link in links:
    print("node1: %s, node2: %s, score: %f" % (link.node_a, link.node_b, link.score))
```

# Supported Similarity Indices
* Common neighbours

# Installation

Requires: networkx & python3.

`python setup.py install`