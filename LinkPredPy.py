import networkx as nx
import numpy as np
import testing_utils
import unsupervised_methods as unmethods
import supervised_methods as smethods
import random
import graph_utils
import datasets_stats
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
    
def unsupservised_method(method_name, G, X_nodes, options, removal_perc=0.3,
                         n_folds=10, undersample=False, seed=0):
    """
    Returns the the AU-ROCs and AU-PRs of the n_folds for the specified unsupervised method.
    Method names: CN, Salton, Jacard, Sorensen, HP, HD, LHN1, PA, AA, RA, LP, Katz, RWR, LRW, SPRW
    The abbrevations of follow those mentioned in the Experiementation chapter in the thesis.
    
    The methods that require parameters (see Parameters below):
        LP: options['lp_katz_h'] damping factor.
        RWR: options['rwr_alpha'] the probability of continuing the random walk.
        LRW: options['lrw_nSteps'] number of steps to run the walk.
        SPRW: options['lrw_nSteps'] number of steps to run the walk.
    
    usage:
    unsupservised_method(method_name, G, X_nodes, options, removal_perc=0.3,
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
                    Add the node attributes if you want to run the unsupervised methods on
                    Social-Attribute Networks.
                    
    :type options: python dictionary.
    :param options: the options needed for some methods.
    
    :type removal_perc: float.
    :param removal_perc: the percentage of edges to be removed to be used in test.
    
    :type n_folds: int.
    :param n_folds: number of folds.
    
    :type undersample: boolean.
    :param undersample: if it should only use a part of the negative edges to testing and not all of them.
    
    :type seed: int.
    :param seed: the see for the traning and testing edges.
    
    """
    node_to_index = testing_utils.get_node_to_index(G)
    extended_graphs = None
    
    if X_nodes == None:
        extended_graphs = False
    else:
        extended_graphs = True
        
    roc_aucs, pr_aucs = unmethods.calculate_method(method_name, G, X_nodes,
                        n_folds, node_to_index, removal_perc, options, undersample, extended_graphs, random_state=seed)
    
    return roc_aucs, pr_aucs

def classifier_method(clf, G, X_nodes, options, plot_file_path, 
                      enabled_features = [[1], [2], [1,2]], tests_names = ["local", "global", 'loc+glob'],
                      removal_perc=0.3, n_folds=10, undersample=False, seed=0):
    """
    Use a classifier (e.g. Logistic Regression) from scikit-learn to perform supervised link prediction.
    
    The methods that require parameters:
        LP: options['lp_katz_h'] damping factor.
        RWR: options['rwr_alpha'] the probability of continuing the random walk.
        LRW: options['lrw_nSteps'] number of steps to run the walk.
        SPRW: options['lrw_nSteps'] number of steps to run the walk.
        
    Usage:
    classifier_method(clf, G, X_nodes, options, 
                      enabled_features = [[1], [2], [1,2]], tests_names = ["local", "global", 'loc+glob'],
                       plot_file_path, removal_perc=0.3, n_folds=10, undersample=False, seed=0)
        
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
    
    :type seed: int.
    :param seed: the seed for the traning and testing edges.
    
    Returns:
    --------
    It outputs a plot for the ROC and PR curves. It also outputs the TPR, FPR, Precision, Recall values as numpy arrays, 
    so you can plot the ROC and PR curves as you want.
    
    """
    
    node_to_index = testing_utils.get_node_to_index(G)
    testing_utils.train_with_stratified_cross_validation_new_protocol(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=n_folds, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_path, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=undersample, extend_graph=False, random_state=seed)

def matrix_fact_traditional(G, X, test_name, options, plot_file_name, edge_removal_perc=0.3, seed=0):
    """
    Use traditional matrix factorization model.
    
    usage: 
    
    matrix_fact_traditional(G, X, test_name, options, plot_file_name, edge_removal_perc=0.3, seed=0)
    
    Parameters:
    -------------
    
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
    
    :type edge_removal_perc: float.
    :param edge_removal_perc: the percentage of edges to be removed to be used in test.
    
    :type seed: int.
    :param seed: the see for the traning and testing edges.
    
    Returns:
    --------
    It outputs a plot for the ROC and PR curves. It also outputs the TPR, FPR, Precision, Recall values as numpy arrays, 
    so you can plot the ROC and PR curves as you want.
    """
    rocs = []
    prs = []

    all_curves = smethods.train_matrix_fact_normal(G, X, test_name, options, edge_removal_perc, seed)
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    rocs.append(roc)
    prs.append(pr_cur)
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR.pdf")


def matrix_fact_auc_opti(G, X, test_name, options, plot_file_name, edge_removal_perc, seed=0):
    """
    Use auc-optimized matrix factorization model.
    
    usage: matrix_fact_auc_opti(G, X, test_name, options, plot_file_name, edge_removal_perc, seed=0)
    
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
    
    :type edge_removal_perc: float.
    :param edge_removal_perc: the percentage of edges to be removed to be used in test.
    
    :type seed: int.
    :param seed: the see for the traning and testing edges.
    
    Returns:
    --------
    It outputs a plot for the ROC and PR curves. It also outputs the TPR, FPR, Precision, Recall values as numpy arrays, 
    so you can plot the ROC and PR curves as you want.
    """
    rocs = []
    prs = []

    all_curves = smethods.train_matrix_fact_ranking(G, X, test_name, options, edge_removal_perc, seed)
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    roc = all_curves["roc"]
    pr_cur = all_curves["pr"]
    rocs.append(roc)
    prs.append(pr_cur)
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR.pdf")




def supervised_random_walk(G, X, test_name, plot_file_name, k=10, delta=5, alpha=0.5, iter=1, psiClass=None):
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
    
    Returns:
    --------
    It outputs a plot for the ROC and PR curves. It also outputs the TPR, FPR, Precision, Recall values as numpy arrays, 
    so you can plot the ROC and PR curves as you want.
    """
    
    rocs = []
    prs = []

    all_curves, _, _ = smethods.train_with_srw(G, X, test_name, k, delta, alpha, iter, psiClass)
    roc = all_curves["roc"]
    pr_curv = all_curves['pr']
    rocs.append(roc)
    prs.append(pr_curv)
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR_"+".pdf")

    

def get_train_test_split(G, fold_number, removal_perc=0.3, undersample=False, seed = 0):
    """
    Gives the train test split that is used for the other methods, for a specific fold and specific seed.
    
    Usage:
    get_train_test_split(G, fold_number, removal_perc=0.3, undersample=False, seed = 0)
    
    Parameters:
    -----------
    :type G: networkx graph.
    :param G: the graph. 
    
    :type removal_perc: float.
    :param removal_perc: the percentage of edges to be removed to be used in test.
    
    :type fold_number: int.
    :param fold_number: the fold number invovled in the train/test split.
    
    :type undersample: boolean.
    :param undersample: should it use undersampling too.
    
    :type seed: int.
    :param seed: the seed for the traning and testing edges.
    
    Returns:
    ---------
    G: networkx object which represents the graph used for training.
    test_set: a list of edges which represent the test set.
    Y: the ground truth of the test edges.
    """
    random.seed(seed)
    random_state = 0
    for i in xrange(fold_number):
        random_state = random.randint(0,1000)
    Gx, test_set, Y, pp_list, np_list, nn_list = graph_utils.prepare_graph_for_training_new_protocl(G, removal_perc, undersample, random_state = random_state)

    return Gx, test_set, Y


def generate_correlation_plot(X, labels, path):
    """
    Generates a correlation plot.
    
    Usage:
    
    generate_correlation_plot(X, labels, path)
    
    Parameters:
    -----------
    X: the matrix that contains the random variables and their values.
    labels: a python list that contains names (strings) for each random variable, eg ['R1', 'R2'] if X's shape is (2,2).
    path: the path where to save the plot.
    """
    datasets_stats.save_correlation_plot(X, labels, path)

def calculate_AUPR(y_test, probs):
    """
    Returns AUPR
    
    usage:
    calculate_AUPR(y_test, probs)
    
    Parameters:
    -----------
    y_test: a numpy array that contains the ground truth for the test samples.
    probs: a numpy array that contains the predicated confidence scores for the test samples.
    
    Returns:
    ---------
    the AUPR as a float.
    """
    
    return testing_utils.compute_PR_auc(y_test, probs)

def calculate_AUROC(y_test, probs):
    """
    Returns AUROC
    
    usage:
    calculate_AUROC(y_test, probs)
    
    Parameters:
    -----------
    y_test: a numpy array that contains the ground truth for the test samples.
    probs: a numpy array that contains the predicated confidence scores for the test samples.
    
    Returns:
    ---------
    the AUROC as a float.
    """
    return testing_utils.compute_AUC(y_test, probs)

def get_dbn(n_components, n_iter, n_RBMs):
    """
    Returns a Deep Belief Network.
    
    Usage:
    get_dbn(n_components, n_iter, n_RBMs)
    
    Parameters:
    -----------
    n_components: number of hidden units in each RBM.
    n_iter: numb of iterations
    n_RBMs: numb of RBMs to use.
    
    
    Returns:
    --------
    a scikit learn classifier object.
    """
    
    steps = []
    
    logistic = linear_model.LogisticRegression()
    
    for i in xrange(n_RBMs):
        rbm = BernoulliRBM(n_components=n_components, n_iter=n_iter)
        name = "rbm" + str(i)
        steps.append( (name, rbm) )
    
    steps.append( ('logistic', logistic) )
    rbm_logistic_clf = Pipeline(steps=steps) 
    return rbm_logistic_clf



