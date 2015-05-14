LinkPredPy
==========

Just import the `LinkPredPy.py` script and use one of its methods.

This script has the following five methods:

- `unsupservised_method(method_name, G, X_nodes, options, removal_perc=0.3,`
                         `n_folds=10, undersample=False, seed=0)`

- `classifier_method(clf, G, X_nodes, options, plot_file_path, `
                      `enabled_features = [[1], [2], [1,2]], tests_names = ["local", "global", 'loc+glob'],`
                      `removal_perc=0.3, n_folds=10, undersample=False, seed=0)`

- `matrix_fact_traditional(G, X, test_name, options, plot_file_name, edge_removal_perc=0.3, seed=0)`
- `matrix_fact_auc_opti(G, X, test_name, options, plot_file_name, edge_removal_perc, seed=0)`
- `supervised_random_walk(G, X, test_name, plot_file_name, k=10, delta=5, alpha=0.5, iter=1, psiClass=None)`
- `get_train_test_split(G, fold_number, removal_perc=0.3, undersample=False, seed = 0)`
- `generate_correlation_plot(X, labels, path)`
- `calculate_AUPR(y_test, probs)`
- `calculate_AUROC(y_test, probs)`
- `get_dbn(n_components, n_iter, n_RBMs)`


Each method contains a Docstring. Just use LinkPredPy.classifier_method? with IPython for example to get the documentation.


Code examples (check examples.py):
```
import LinkPredPy as lp
import networkx as nx
import numpy as np
from sklearn import linear_model


G = nx.fast_gnp_random_graph(50, 0.5) #generate a random graph to play with.
 
# Example 1) unsupervised method:
#
# options = {}
# roc_aucs, pr_aucs = lp.unsupservised_method("CN", G, None, options, 
#                                             removal_perc=0.3, n_folds=2, undersample=False, seed=0)
# 
# print roc_aucs, pr_aucs

# Example 2) classifier method:
# clf = linear_model.LogisticRegression()
# options = {}
# options['lp_katz_h'] = 0.05
# options['rwr_alpha'] = 0.3
# options['lrw_nSteps'] = 5
# path = "/Users/x/Desktop/plot.pdf"
# lp.classifier_method(clf, G, X_nodes=None, options = options, plot_file_path=path, 
#                       enabled_features = [[1]], tests_names = ["local"],
#                       removal_perc=0.3, n_folds=1, undersample=False, seed=0)

# Example 3) matrix factorization methods:
# path = "/Users/x/Desktop/plot.pdf"
# 
# options = {}
# options['mf_n_latent_feats'] = 2 #(number of latent features), 
# options['mf_n_folds'] = 2 #(number of folds), 
# options['mf_alpha'] = 0.1 #(gradient descent learning rate), 
# options['mf_n_iter'] = 1 #(number of gradient descent epochs), 
# options['mf_with_sampling'] = False

# lp.matrix_fact_traditional(G, X=None, test_name="MF", options=options, plot_file_name=path, edge_removal_perc=0.3, seed=0)

# lp.matrix_fact_auc_opti(G, X=None, test_name="MF", options=options, plot_file_name=path, edge_removal_perc=0.3, seed=0)


# Example 3) supervised random walk:
# path = "/Users/x/Desktop/plot.pdf"
# X = np.random.randn(50,2) #two features for each node
# lp.supervised_random_walk(G, X, test_name="SRW", plot_file_name=path, k=10, delta=5, alpha=0.5, iter=1, psiClass=None)

# Gx, test_set, Y = lp.get_train_test_split(G, fold_number=1, removal_perc=0.3, undersample=False, seed = 0)
#Gx is the graph that should be used for training
# print G.number_of_edges(), Gx.number_of_edges()

```










