import networkx as nx
import numpy as np
import testing_utils
import unsupervised_methods as unmethods
import supervised_methods as smethods

    
def unupservised_method(method_name, G, X_nodes, options, removal_perc=0.3,
                         n_folds=10, undersample=False):
    """
    Returns the the AU-ROCs and AU-PRs of the n_folds for the specified unsupervised method.
    Method names: CN, Salton, Jacard, Sorensen, HP, HD, LHN1, PA, AA, RA, LP, Katz, RWR, LRW, SPRW
    The methods that require parameters:
        LP: options['lp_katz_h'] damping factor.
        RWR: options['rwr_alpha'] the probability of continuing the random walk.
        LRW: options['lrw_nSteps'] number of steps to run the walk.
        SPRW: options['lrw_nSteps'] number of steps to run the walk.
    
    e.g.:
    
    unupservised_method(method_name, G, X_nodes, options, removal_perc=0.3,
                         n_folds=10, undersample=False)    
    
    Parameters:
    -----------
    :type method_name: string
    :param method_name: name of the method to be used.
    
    :type G: networkx graph.
    :param G: the graph.
    
    :type X_nodes: 2D numpy array.
    :param X_nodes: each row of this matrix is the feature vector of a node. 
                    The nodes have the same index as in the networkx graph G.
                    
    :type options: python dictionary.
    :param options: the options needed for some methods.
    
    :type removal_perc: float.
    :param removal_perc: the percentage of edges to be removed to be used in test.
    
    :type n_folds: int.
    :param n_folds: number of folds.
    
    :type undersample: boolean.
    :param undersample: should it use undersampling too.
    
    """
    node_to_index = testing_utils.get_node_to_index(G)
    extended_graphs = None
    
    if X_nodes == None:
        extended_graphs = False
    else:
        extended_graphs = True
        
    roc_aucs, pr_aucs = unmethods.calculate_method(method_name, G, X_nodes,
                        n_folds, node_to_index, removal_perc, options, undersample, extended_graphs)
    
    return roc_aucs, pr_aucs

def classifier_method(clf, G, X_nodes, options, 
                      enabled_features = [[1], [2], [1,2]], tests_names = ["local", "global", 'loc+glob'],
                       plot_file_path, removal_perc=0.3, n_folds=10, undersample=False):
    """
    Use a classifier (e.g. Logistic Regression) from scikit-learn to perform supervised link prediction.
    The methods that require parameters:
        LP: options['lp_katz_h'] damping factor.
        RWR: options['rwr_alpha'] the probability of continuing the random walk.
        LRW: options['lrw_nSteps'] number of steps to run the walk.
        SPRW: options['lrw_nSteps'] number of steps to run the walk.
        
    e.g:
    
    classifier_method(clf, G, X_nodes, options, 
                      enabled_features = [[1], [2], [1,2]], tests_names = ["local", "global", 'loc+glob'],
                       plot_file_path, removal_perc=0.3, n_folds=10, undersample=False)
        
    Parameters:
    -----------
    :type clf: scikit-learn classifier
    :param clf: the classifier to be used.
    
    :type G: networkx graph.
    :param G: the graph.
    
    :type X_nodes: 2D numpy array.
    :param X_nodes: each row of this matrix is the feature vector of a node. 
                    The nodes have the same index as in the networkx graph G.
                    
    :type options: python dictionary.
    :param options: the options needed for some methods.
    
    :type enabled_features: python list.
    :param enabled_features: a list of list that contains which features to be used in creating the dataset.
                            e.g.: enabled_features = [[1], [2], [1,2]] will first create a traininig dataset
                            using the local topological features only. Then another dataset with the global features only.
                            Then another dataset with local & global featuers. The output plot will contain 
                            the ROC plots for all of the tests.
                            
                            Flags: 1 for local, 2 for global, 3 for node attributes, 4 for raw features.
    :type tests_names: python list of strings.
    :param tests_names: the names of the ROC curves to be put on the plot (see enabled_features).
                        e.g tests_names = ["local", "global", 'loc+glob']
    
    :type removal_perc: float.
    :param removal_perc: the percentage of edges to be removed to be used in test.
    
    :type plot_file_path: python string.
    :param plot_file_path: the path to save the output plot for ROC and PR.
    
    :type n_folds: int.
    :param n_folds: number of folds.
    
    :type undersample: boolean.
    :param undersample: should it use undersampling too.
        
    
    """
    
    node_to_index = testing_utils.get_node_to_index(G)
    testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=n_folds, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_path, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=undersample, extend_graph=False)

def matrix_fact_traditional(G, X, test_name, options, plot_file_name):
    """
    Use traditional matrix factorization model.
    
    :type G: networkx graph.
    :param G: the graph.
    
    :type test_name: python string.
    :param test_name: the name of ROC curve on the plot.
    
    :type X: 2D numpy array [None if not available].
    :param X: each row of this matrix is the feature vector of a node. 
                    The nodes have the same index as in the networkx graph G.
    
    :type options: python dictionary.
    :param options: a dictionary that holds the following parameters:
            options['mf_n_latent_feats'] (number of latent features), 
            options['mf_n_folds'] (number of folds), 
            options['mf_alpha'] (gradient descent learning rate), 
            options['mf_n_iter'] (number of gradient descent epochs), 
            options['mf_with_sampling'] (use undersampling or not)
    :type plot_file_name: python string.
    :param plot_file_name: where to save the ROC/PR plots.
    
    
    """
    rocs = []
    prs = []

    all_curves = smethods.train_matrix_fact_normal(G, X, test_name, options)
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    rocs.append(roc)
    prs.append(pr_cur)
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR.pdf")


def matrix_fact_auc_opti(G, X, test_name, options, plot_file_name):
    """
    Use auc-optimized matrix factorization model.
    
    :type G: networkx graph.
    :param G: the graph.
    
    :type test_name: python string.
    :param test_name: the name of ROC curve on the plot.
    
    :type X: 2D numpy array [None if not available].
    :param X: each row of this matrix is the feature vector of a node. 
                    The nodes have the same index as in the networkx graph G.
    
    :type options: python dictionary.
    :param options: a dictionary that holds the following parameters:
            options['mf_n_latent_feats'] (number of latent features), 
            options['mf_n_folds'] (number of folds), 
            options['mf_alpha'] (gradient descent learning rate), 
            options['mf_n_iter'] (number of gradient descent epochs), 
            options['mf_with_sampling'] (use undersampling or not)
    
    :type plot_file_name: python string.
    :param plot_file_name: where to save the ROC/PR plots.
    
    
    """
    rocs = []
    prs = []

    all_curves = smethods.train_matrix_fact_ranking(G, X, test_name, options)
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    rocs.append(roc)
    prs.append(pr_cur)
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR.pdf")




def supervised_random_walk(G, X, k=10, delta=5, alpha=0.5, iter=1, test_name, plot_file_name):
    """
    Link prediction with supervised random walk.
    
    :type G: networkx graph.
    :param G: the graph.
    
    :type X: 2D numpy array [None if not available].
    :param X: each row of this matrix is the feature vector of a node. 
                    The nodes have the same index as in the networkx graph G.
                    
    :type test_name: python string.
    :param test_name: the name of ROC curve on the plot.
                    
    :type plot_file_name: python string.
    :param plot_file_name: where to save the ROC/PR plots.
    
    :type k: int
    :param k: the number of neighbours a node must have in order to be considered a candidate source node.
    
    :type delta: int
    :param delta: the number of edges the candidate source node made in the future that close a triangle. 
           (i.e. the future/destination node is a friend of a friend, so delta is a threshold that measures how many
             of these future/destination nodes are friends of current friends). A candidate source node
             that has a degree above k, and made future friends above delta, then becomes a source node
             for training.
    
    :type alpha: float
    :param alpha: restart probability.
    
    :type iter: int
    :param iter: gradient desecent epochs.
    """
    
    rocs = []
    prs = []

    all_curves, _, _ = smethods.train_with_srw(G, X, test_name, k, delta, alpha, iter)
    roc = all_curves["roc"]
    pr_curv = all_curves['pr']
    rocs.append(roc)
    prs.append(pr_curv)
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR_"+".pdf")

    











