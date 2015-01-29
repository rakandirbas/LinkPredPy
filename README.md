LinkPredPy
==========

Just import LinkPredPy.py script and use one of its methods.

This script has the following five methods:

- unupservised_method(method_name, G, X_nodes, options, removal_perc=0.3,
                         n_folds=10, undersample=False)

- classifier_method(clf, G, X_nodes, options, 
                      enabled_features = [[1], [2], [1,2]], tests_names = ["local", "global", 'loc+glob'],
                       plot_file_path, removal_perc=0.3, n_folds=10, undersample=False)

- matrix_fact_traditional(G, X, test_name, options, plot_file_name)
- matrix_fact_auc_opti(G, X, test_name, options, plot_file_name)
- supervised_random_walk(G, X, k=10, delta=5, alpha=0.5, iter=1, test_name, plot_file_name)


Each method contains a Docstring. Just use LinkPredPy.classifier_method? with IPython for example to get the documentation.
