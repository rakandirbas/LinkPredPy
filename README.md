LinkPredPy
==========
A gentle link prediction library in python.


#Example

```python
import networkx as nx
from linkpredpy import lp

G = nx.fast_gnp_random_graph(5, 0.5) # create a random graph.
links = lp.missing_links(G, lp.cn) # use common neighbours as a similarity index.

for link in links:
    print("node1: %s, node2: %s, score: %f" % (link.node_a, link.node_b, link.score))
```

#Requirements
networkx & python3

#Installation

`python setup.py install`