"""
This module includes the old experiements (especially for exp1 and exp2 after chaning to stratified cross validation)
"""

from __future__ import division
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import networkx as nx
import numpy as np
from rakg import graphsim, graph_utils,\
     light_srw, matrix_fact, supervised_methods, \
     testing_utils, timer, srw, facebook100_parser, datasets_stats,\
     datasets
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from memory_profiler import profile

# @profile
def run_tests():
    args = sys.argv
    dataset_id = int(args[1])
    exp_id = int(args[2])
    large_data_set = False
    
    if len(args) == 4:
        yes_no = args[3]
        if yes_no == "y":
            large_data_set = True
        else:
            large_data_set = False
    
    if len(args) == 5:
        k = int(args[3])
        delta = int(args[4])
    
    path = "/home/rdirbas/final_tests/"
#     path = "/Users/rockyrock/Desktop/final_tests/"
    random_state = 0
    removal_perc = 0.4
    
    options = {}
    
    

    if dataset_id == 1: #load Facebook100 Caltech36
        dataset_path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
        G, X_nodes = facebook100_parser.parse(dataset_path)
        path = path + str(dataset_id) + '/' #appends the dataset directory
        
        #adding the parameters for SRW
        options_list_srw = []
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=5,del=5,itr=10"
        options_srw["srw_k"] = 5
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 10
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=30"
        options_srw["srw_k"] = 10
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 30
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=50"
        options_srw["srw_k"] = 10
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 50
        options_list_srw.append(options_srw)
        
        ###############################
        
    elif dataset_id == 2: #load a small facebook snap net (300 nodes).
        dataset_path = "/home/rdirbas/datasets/SNAP/facebook/facebook/0.edges"
#         dataset_path = "/Users/rockyrock/Desktop/facebook/0.edges"
        G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
        path = path + str(dataset_id) + '/'
        
        #adding the parameters for SRW
        options_list_srw = []
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=5,del=5,itr=10"
        options_srw["srw_k"] = 5
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 10
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=30"
        options_srw["srw_k"] = 10
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 30
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=50"
        options_srw["srw_k"] = 10
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 50
        options_list_srw.append(options_srw)
        
        ###############################
        
    elif dataset_id == 3: #load a small facebook snap net (150 nodes).
        dataset_path = "/home/rdirbas/datasets/SNAP/facebook/facebook/414.edges"
#         dataset_path = "/Users/rockyrock/Desktop/facebook/414.edges"
        G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
        path = path + str(dataset_id) + '/'
        
        #adding the parameters for SRW
        options_list_srw = []
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=5,del=5,itr=10"
        options_srw["srw_k"] = 20
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 10
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=30"
        options_srw["srw_k"] = 20
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 30
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=50"
        options_srw["srw_k"] = 20
        options_srw["srw_delta"] = 5
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 50
        options_list_srw.append(options_srw)
        
        ###############################
        
    elif dataset_id == 4: #loads a big fb dataset (1034 nodes)
        dataset_path = "/home/rdirbas/datasets/SNAP/facebook/facebook/107.edges"
#         dataset_path = "/Users/rockyrock/Desktop/facebook/107.edges"
        G, X_nodes = datasets.load_SNAP_dataset(dataset_path)
        path = path + str(dataset_id) + '/'
        
        #adding the parameters for SRW
        options_list_srw = []
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=5,del=5,itr=10"
        options_srw["srw_k"] = 60
        options_srw["srw_delta"] = 60
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 10
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=30"
        options_srw["srw_k"] = 60
        options_srw["srw_delta"] = 60
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 30
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=10,del=5,itr=50"
        options_srw["srw_k"] = 60
        options_srw["srw_delta"] = 60
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 50
        options_list_srw.append(options_srw)
        
        ###############################
        
        
    elif dataset_id == 5: #loads a big g+ dataset
        dataset_path = "/home/rdirbas/datasets/SNAP/gplus/101263615503715477581.edges"
        G, X_nodes = datasets.load_SNAP_dataset(dataset_path, directed=True)
        G = G.to_undirected(reciprocal=True)
        path = path + str(dataset_id) + '/'
        
        
        #adding the parameters for SRW
        options_list_srw = []
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=90,del=90,itr=10"
        options_srw["srw_k"] = 90
        options_srw["srw_delta"] = 90
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 10
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=90,del=90,itr=30"
        options_srw["srw_k"] = 90
        options_srw["srw_delta"] = 90
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 30
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=90,del=90,itr=50"
        options_srw["srw_k"] = 90
        options_srw["srw_delta"] = 90
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 50
        options_list_srw.append(options_srw)
        
        ###############################
        
    elif dataset_id == 6: #loads a big g+ dataset
        dataset_path = "/home/rdirbas/datasets/SNAP/gplus/100329698645326486178.edges"
        G, X_nodes = datasets.load_SNAP_dataset(dataset_path, directed=True)
        G = G.to_undirected(reciprocal=True)
        path = path + str(dataset_id) + '/'
        
        #adding the parameters for SRW
        options_list_srw = []
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=30,del=30,itr=10"
        options_srw["srw_k"] = 30
        options_srw["srw_delta"] = 30
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 10
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=30,del=30,itr=30"
        options_srw["srw_k"] = 30
        options_srw["srw_delta"] = 30
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 30
        options_list_srw.append(options_srw)
        
        options_srw = {}
        options_srw["srw_test_name"] = "k=30,del=30,itr=50"
        options_srw["srw_k"] = 30
        options_srw["srw_delta"] = 30
        options_srw["srw_alpha"] = 0.1
        options_srw["srw_iter"] = 50
        options_list_srw.append(options_srw)
        
        ###############################
        
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0 or len(X_nodes) == 0:
        raise Exception("The graph has zero nodes or zero edges!") 
        
    G, X_nodes = testing_utils.filter_unlinked_nodes(G, X_nodes)
    node_to_index = testing_utils.get_node_to_index(G)
    
    
    if exp_id == 1:
        """
        Computes statistics for the normal graph.
        """
        
        print datasets_stats.get_statistic(G, X_nodes)
        
        enabled_features = [[1], [2], [1,2]]
        tests_names = ["local", "global", 'loc_glob']
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        Xs, Ys, X_pps, Y_pps = \
            testing_utils.build_training_datasets(G, X_nodes, node_to_index, params=options, 
                        edge_removal_perc=removal_perc, enabled_features=enabled_features)
        

        for X, Y, X_pp, Y_pp, test_name in zip(Xs, Ys, X_pps, Y_pps, tests_names):
            plot_file_name = path + str(exp_id) + '/' + test_name + "_"
            datasets_stats.save_covariance_plot(X, plot_file_name + 'cov_' + 'X')
            datasets_stats.save_covariance_plot(X_pp, plot_file_name + 'cov_' + 'X_pp')
            
            datasets_stats.save_scatterplot_matix(X, plot_file_name + 'scatter_' + 'X')
            datasets_stats.save_scatterplot_matix(X_pp, plot_file_name + 'scatter_' + 'X_pp')
            
            X_train = np.vstack(( X_pp, X ))
            y_train = np.concatenate(( Y_pp, Y ))
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
            
            importances, feature_order = datasets_stats.get_important_features(X_train, y_train)
            print '\n\nPrinting the features and their importances\n'
            print 'The first array is the features_order. The second is the importance of each feature\
            sorted from the most important to lowest.\n\n'
            
            print "features order:\n\n" 
            print feature_order 
            print '\n\n'
            print "Importances:\n\n" 
            print importances[feature_order]
            print '\n\n'
            
        print '\nPrinting statistics for the neighbourhood datsets:\n'
        print datasets_stats.neighbourhood_datasets_stats(G, removal_perc)
        print '\n\n'
        
        
    elif exp_id == 2:
        """
        Print statistics for the extended graph.
        """
        
        print '\nPrinting statistics for the extended graphs datsets:\n'
        Gx, original_degrees = graph_utils.get_extended_graph(G, X_nodes)
        print datasets_stats.get_statistic(Gx, X_nodes)
        
        enabled_features = [[1], [2], [1,2]]
        tests_names = ["local", "global", 'loc_glob']
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        Xs, Ys, X_pps, Y_pps = \
            testing_utils.build_training_datasets(G, X_nodes, node_to_index, params=options, 
                        edge_removal_perc=removal_perc, enabled_features=enabled_features, extend_graph=True)
            
        for X, Y, X_pp, Y_pp, test_name in zip(Xs, Ys, X_pps, Y_pps, tests_names):
            plot_file_name = path + str(exp_id) + '/' + test_name + "_"
            datasets_stats.save_covariance_plot(X, plot_file_name + 'cov_' + 'X')
            datasets_stats.save_covariance_plot(X_pp, plot_file_name + 'cov_' + 'X_pp')
            
            datasets_stats.save_scatterplot_matix(X, plot_file_name + 'scatter_' + 'X')
            datasets_stats.save_scatterplot_matix(X_pp, plot_file_name + 'scatter_' + 'X_pp')
            
            X_train = np.vstack(( X_pp, X ))
            y_train = np.concatenate(( Y_pp, Y ))
            X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
            
            importances, feature_order = datasets_stats.get_important_features(X_train, y_train)
            print '\n\nPrinting the features and their importances\n'
            print 'The first array is the features_order. The second is the importance of each feature\
            sorted from the most important to lowest.\n\n'
            
            print "features order:\n\n" 
            print feature_order 
            print '\n\n'
            print "Importances:\n\n" 
            print importances[feature_order]
            print '\n\n'
        
    elif exp_id == 3:
        options['katz_h'] = 0.005
        
        Gx, U, Y, pp_list, np_list, nn_list = \
            graph_utils.prepare_graph_for_training(G, removal_perc) 
            
        test_local_unsupervised_methods(Gx, U, Y, node_to_index, options)
        
        cross_validation_for_unsupervised_local_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10)
        
    elif exp_id == 4:
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        Gx, U, Y, pp_list, np_list, nn_list = \
        graph_utils.prepare_graph_for_training(G, removal_perc) 
    
        test_global_unsupervised_methods(Gx, U, Y, node_to_index, options)
        
        cross_validation_for_unsupervised_global_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10,
                                                      extended_graphs = False)
    elif exp_id == 5:
        options['rwr_alpha'] = 0.5
        options['lrw_nSteps'] = 15
        
        Gx, U, Y, pp_list, np_list, nn_list = \
        graph_utils.prepare_graph_for_training(G, removal_perc) 
    
        test_global_unsupervised_methods(Gx, U, Y, node_to_index, options)
        
        cross_validation_for_unsupervised_global_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10,
                                                      extended_graphs = False)
        
    elif exp_id == 6:
        """
        Normal classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and raw features.
        """
        clfs = get_normal_clfs()
        X_pp_flags = [0, 1]
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
            
        for i, clf in enumerate(clfs):
            plot_file_name = path + str(exp_id) + '/' + "_clfid_" + str(i) + "_stratified.png"
            print 'test with clf id: ', i, "\n\n"
            testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 7:
        """
        Normal classifiers, all kinds of datasets creation+testing,
        local and global features, local and attributes, global and attributes, 
        local global and attributes features.
        """
        clfs = get_normal_clfs()
        X_pp_flags = [0, 1]
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        enabled_features = [[1,2], [1,3], [2,3], [1,2,3]]
        
        tests_names = ['loc+glob', 'loc+att', 'glob+att', 'loc+glob+att']
        
        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        for i, clf in enumerate(clfs):
            plot_file_name = path + str(exp_id) + '/' + "_clfid_" + str(i) + "_stratified.png"
            print 'test with clf id: ', i, "\n\n"
            testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
            
    elif exp_id == 8:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: default
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 256
        options['RBM_n_iter'] = 10
        options['n_RBMs'] = 1
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        
        clf = get_deep_clf(options)
        clfs = [clf]
        
        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=2, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
        
    elif exp_id == 9:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 100
        options['RBM_n_iter'] = 10
        options['n_RBMs'] = 1
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]
        
        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 10:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 400
        options['RBM_n_iter'] = 10
        options['n_RBMs'] = 1
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]

        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 11:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 256
        options['RBM_n_iter'] = 40
        options['n_RBMs'] = 1
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]

        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 12:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 256
        options['RBM_n_iter'] = 10
        options['n_RBMs'] = 2
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]
        
        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
                
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 13:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 256
        options['RBM_n_iter'] = 10
        options['n_RBMs'] = 3
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]

        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 14:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 256
        options['RBM_n_iter'] = 20
        options['n_RBMs'] = 2
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]

        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 15:
        """
        Deep classifiers, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and RAW features.
        
        Different RBM parameters: 
        """
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        options['RBM_n_components'] = 400
        options['RBM_n_iter'] = 40
        options['n_RBMs'] = 1
        
        X_pp_flags = [0, 1]
        
        if large_data_set:
            enabled_features = [[1], [2], [3]]
            tests_names = ["local", "global", 'atts']
        else:
            enabled_features = [[1], [2], [3], [4]]
            tests_names = ["local", "global", 'atts', 'raw']
        
        clf = get_deep_clf(options)
        clfs = [clf]

        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path)
        
        plot_file_name = path + str(exp_id) + '/' + "stratified.png"
        testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=False)
    elif exp_id == 16:
        """
        Normal classifiers with UNDER-SAMPLING, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and raw features.
        """
        clfs = get_normal_clfs()
        X_pp_flags = [0, 1]
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        
        if large_data_set:
            enabled_features = [[1], [2], [3], [1,2,3]]
            tests_names = ["local", "global", 'atts','loc+glob+atts']
        else:
            enabled_features = [[1], [2], [3], [4], [1,2,3]]
            tests_names = ["local", "global", 'atts', 'raw','loc+glob+atts']
            
        
        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path, undersample=True)
            
        for i, clf in enumerate(clfs):
            plot_file_name = path + str(exp_id) + '/' + "_clfid_" + str(i) + "_stratified.png"
            print 'test with clf id: ', i, "\n\n"
            testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=True, extend_graph=False)
    elif exp_id == 17:
        """
        Normal classifiers with Neighbourhood-SAMPLING, all kinds of datasets creation+testing, local features only,
        global features only, attributes only and raw features.
        """
        clfs = get_normal_clfs()
        X_pp_flags = [0]
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        enabled_features = [1,2] #NOT a list of lists!!
        
        tests_names = ["local", "global", 'atts', 'raw', 'loc+glob+atts']
        
        Xs, Ys = testing_utils.build_neighborhood_sets(G, X_nodes, node_to_index, params=options, 
                            edge_removal_perc=removal_perc, enabled_features=enabled_features)
        
        X_pps = [None, None, None]
        Y_pps = [None, None, None]
        
        for i, clf in enumerate(clfs):
            plot_file_name = path + str(exp_id) + '/' + "_clfid_" + str(i)
            print 'test with clf id: ', i, "\n\n"
            test_supervised_methods_general(clf, Xs, Ys, X_pps, Y_pps, X_pp_flags, tests_names, plot_file_name)
    elif exp_id == 18:
        """
        Unsupervised local methods with extended graph
        """
        Gx, U, Y, pp_list, np_list, nn_list = \
            graph_utils.prepare_graph_for_training(G, removal_perc) 
            
        Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes)
        original_degrees_list = graphsim.get_degrees_list(Gx, original_degrees)
            
        options['katz_h'] = 0.005
        
        test_local_unsupervised_methods(Gx, U, Y, node_to_index, options, original_degrees_list)
        
        cross_validation_for_unsupervised_local_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10,
                                                    extended_graphs = True)
        
    elif exp_id == 19:
        """
        Unsupervised global methods with extended graph
        """
        Gx, U, Y, pp_list, np_list, nn_list = \
            graph_utils.prepare_graph_for_training(G, removal_perc)
        
        Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes) 
    
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        test_global_unsupervised_methods(Gx, U, Y, node_to_index, options)
        
        cross_validation_for_unsupervised_global_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10,
                                                      extended_graphs = True)
        
    elif exp_id == 20:
        """
        Normal classifiers with Extended graphs, all kinds of datasets creation+testing, local features only,
        global features only, local and global features.
        """
        clfs = get_normal_clfs()
        X_pp_flags = [0, 1]
        
        options['rwr_alpha'] = 0.7
        options['lrw_nSteps'] = 10
        
        enabled_features = [[1], [2], [1,2]]
        
        tests_names = ["local", "global", 'loc+glob']

        plot_path = path + str(exp_id) + '/'
        exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path, extended_graphs=True)
            
        for i, clf in enumerate(clfs):
            plot_file_name = path + str(exp_id) + '/' + "_clfid_" + str(i) + "stratified.png"
            print 'test with clf id: ', i, "\n\n"
            testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                            clf=clf, n_folds=10, tests_names=tests_names,
                                            params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                            enabled_features=enabled_features, undersample=False, extend_graph=True)
    elif exp_id == 21:
            """
            Deep classifiers with Extended Graphs, all kinds of datasets creation+testing, local features only,
            global features only, local and global features.
            
            Different RBM parameters: default
            """
            options['rwr_alpha'] = 0.7
            options['lrw_nSteps'] = 10
            options['RBM_n_components'] = 256
            options['RBM_n_iter'] = 10
            options['n_RBMs'] = 1
            
            
            X_pp_flags = [0, 1]
            enabled_features = [[1], [2], [1,2]]
        
            tests_names = ["local", "global", 'loc+glob']
            clf = get_deep_clf(options)
            clfs = [clf]

            plot_path = path + str(exp_id) + '/'
            exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path, extended_graphs=True)
            
            plot_file_name = path + str(exp_id) + '/' + "stratified.png"
            testing_utils.train_with_stratified_cross_validation(G=G, X_nodes=X_nodes, node_to_index=node_to_index,
                                                clf=clf, n_folds=10, tests_names=tests_names,
                                                params=options, plot_file_name=plot_file_name, edge_removal_perc=removal_perc, 
                                                enabled_features=enabled_features, undersample=False, extend_graph=True)
    elif exp_id == 22:
        """
        Supervised random walk.
        
        """
        for options_srw in options_list_srw:
            k = options_srw["srw_k"]
            delta = options_srw["srw_delta"]
            print datasets_stats.srw_stats(G, X_nodes, k=k, delta=delta)
        
        plot_file_name = path + str(exp_id) + '/' + "srw.png"
        test_supervised_random_walk(G, X_nodes, plot_file_name, options_list_srw)
        
    elif exp_id == 23:
        """
        Matrix factorization normal model without node attributes.
        
        """   
        options_list_mfn = []   
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=2'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=2'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=2'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=3'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=3'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=3'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        plot_file_name = path + str(exp_id) + '/' + 'mf_normal.png'
        test_matrix_fact_normal(G, X=None, plot_file_name=plot_file_name, options_list=options_list_mfn)
    elif exp_id == 24:
        """
        Matrix factorization normal model with node attributes.
        
        """   
        options_list_mfn = []   
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=2'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=2'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=2'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=3'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=3'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=3'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        plot_file_name = path + str(exp_id) + '/' + 'mf_normal.png'
        test_matrix_fact_normal(G, X_nodes, plot_file_name, options_list=options_list_mfn)
    elif exp_id == 25:
        """
        Matrix factorization normal model in sampling mode with node attributes.
        
        """   
        options_list_mfn = []   
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=2'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = True
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=2'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = True
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=2'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 2
        options_mfn['mf_with_sampling'] = True
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=3'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = True
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=3'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = True
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=3'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 3
        options_mfn['mf_with_sampling'] = True
        options_list_mfn.append(options_mfn)
        
        plot_file_name = path + str(exp_id) + '/' + 'mf_normal.png'
        test_matrix_fact_normal(G, X_nodes, plot_file_name, options_list=options_list_mfn)
    elif exp_id == 26:
        """
        Matrix factorization ranking model without node attributes.
        
        """   
        options_list_mfn = []   
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=1'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 3
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 1
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=1'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 3
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 1
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=1'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 3
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 1
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
#         options_mfn = {}
#         options_mfn['mf_test_name'] = 'LF=10,itr=3'
#         options_mfn['mf_n_latent_feats'] = 10
#         options_mfn['mf_n_folds'] = 5
#         options_mfn['mf_alpha'] = 0.1
#         options_mfn['mf_n_iter'] = 3
#         options_mfn['mf_with_sampling'] = False
#         options_list_mfn.append(options_mfn)
#         
#         options_mfn = {}
#         options_mfn['mf_test_name'] = 'LF=30,itr=3'
#         options_mfn['mf_n_latent_feats'] = 30
#         options_mfn['mf_n_folds'] = 5
#         options_mfn['mf_alpha'] = 0.1
#         options_mfn['mf_n_iter'] = 3
#         options_mfn['mf_with_sampling'] = False
#         options_list_mfn.append(options_mfn)
#         
#         options_mfn = {}
#         options_mfn['mf_test_name'] = 'LF=50,itr=3'
#         options_mfn['mf_n_latent_feats'] = 50
#         options_mfn['mf_n_folds'] = 5
#         options_mfn['mf_alpha'] = 0.1
#         options_mfn['mf_n_iter'] = 3
#         options_mfn['mf_with_sampling'] = False
#         options_list_mfn.append(options_mfn)
        
        plot_file_name = path + str(exp_id) + '/' + 'mf_ranking.png'
        test_matrix_fact_ranking(G, X=None, plot_file_name=plot_file_name, options_list=options_list_mfn)
    elif exp_id == 27:
        """
        Matrix factorization ranking model with node attributes.
        
        """   
        options_list_mfn = []   
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=10,itr=1'
        options_mfn['mf_n_latent_feats'] = 10
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 1
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=30,itr=1'
        options_mfn['mf_n_latent_feats'] = 30
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 1
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
        options_mfn = {}
        options_mfn['mf_test_name'] = 'LF=50,itr=1'
        options_mfn['mf_n_latent_feats'] = 50
        options_mfn['mf_n_folds'] = 5
        options_mfn['mf_alpha'] = 0.1
        options_mfn['mf_n_iter'] = 1
        options_mfn['mf_with_sampling'] = False
        options_list_mfn.append(options_mfn)
        
#         options_mfn = {}
#         options_mfn['mf_test_name'] = 'LF=10,itr=3'
#         options_mfn['mf_n_latent_feats'] = 10
#         options_mfn['mf_n_folds'] = 5
#         options_mfn['mf_alpha'] = 0.1
#         options_mfn['mf_n_iter'] = 3
#         options_mfn['mf_with_sampling'] = False
#         options_list_mfn.append(options_mfn)
#         
#         options_mfn = {}
#         options_mfn['mf_test_name'] = 'LF=30,itr=3'
#         options_mfn['mf_n_latent_feats'] = 30
#         options_mfn['mf_n_folds'] = 5
#         options_mfn['mf_alpha'] = 0.1
#         options_mfn['mf_n_iter'] = 3
#         options_mfn['mf_with_sampling'] = False
#         options_list_mfn.append(options_mfn)
#         
#         options_mfn = {}
#         options_mfn['mf_test_name'] = 'LF=50,itr=3'
#         options_mfn['mf_n_latent_feats'] = 50
#         options_mfn['mf_n_folds'] = 5
#         options_mfn['mf_alpha'] = 0.1
#         options_mfn['mf_n_iter'] = 3
#         options_mfn['mf_with_sampling'] = False
#         options_list_mfn.append(options_mfn)
        
        plot_file_name = path + str(exp_id) + '/' + 'mf_ranking.png'
        test_matrix_fact_ranking(G, X_nodes, plot_file_name, options_list=options_list_mfn)
#     elif exp_id == 28:
#         """
#         Finds the srw parameters for the selected dataset
#         """
#         print datasets_stats.srw_stats(G, X_nodes, k=90, delta=90)

        
def test_local_unsupervised_methods(G, U, Y, node_to_index, options, original_degrees_list=None):
    """
    
    Do link prediction with local un-supervised methods and print the auc for each
    un-supervised method.
    
    Parameters:
    -----------
    G: networkx graph object.
    U: a list that holds dyads for testing.
    Y: the true labels for the dyads.
    node_to_index: a dictionary that maps from a node name to a node index.
    options: is a dictionary that holds parameters values. 
            Here we only need: options['katz_h'].
    original_degrees_list: to be provided if using extended graphs.
    
    Returns:
    --------
    all_aucs = {"roc": all_aucs_roc, "pr": all_aucs_pr}
    """
    t = timer.Timerx()
    
    degrees = graphsim.get_degrees_list(G)
    katz_h = options['katz_h']
    A = np.array( nx.adj_matrix(G) )
    
    print "\n\nComputing the auc of the simple un-supervised methods...\n"
    
    t.start()
    cn_p = graphsim.predict_scores(U, graphsim.cn(A), node_to_index)
    print "CN: ", t.stop(), "\n"
#     print graphsim.cn(A)
#     print cn_p

    t.start()
    salton_p = graphsim.predict_scores(U, graphsim.salton(A, degrees), node_to_index)
    print "Salton: ", t.stop(), "\n"
    
    t.start()
    jacard_p = graphsim.predict_scores(U, graphsim.jacard(A, degrees), node_to_index)
    print "Jacard: ", t.stop(), "\n"
    
    t.start()
    sorensen_p = graphsim.predict_scores(U, graphsim.sorensen(A, degrees), node_to_index)
    print "Sorensen: ", t.stop(), "\n"
    
    t.start()
    hpi_p = graphsim.predict_scores(U, graphsim.hpi(A, degrees), node_to_index)
    print "HPI: ", t.stop(), "\n"
    
    t.start()
    hdi_p = graphsim.predict_scores(U, graphsim.hdi(A, degrees), node_to_index)
    print "HDI: ", t.stop(), "\n"
    
    t.start()
    lhn1_p = graphsim.predict_scores(U, graphsim.lhn1(A, degrees), node_to_index)
    print "LHN1: ", t.stop(), "\n"
    
    t.start()
    pa_p = graphsim.predict_scores(U, graphsim.pa(A, degrees), node_to_index)
    print "PA: ", t.stop(), "\n"
    
    t.start()
    if original_degrees_list == None:    
        aa_p = graphsim.predict_scores(U, graphsim.aa(A, degrees), node_to_index)
    else:
        aa_p = graphsim.predict_scores(U, graphsim.aa(A, original_degrees_list), node_to_index)
    print "AA: ", t.stop(), "\n"
    
    t.start()
    if original_degrees_list == None: 
        ra_p = graphsim.predict_scores(U, graphsim.ra(A, degrees), node_to_index)
    else:
        ra_p = graphsim.predict_scores(U, graphsim.ra(A, original_degrees_list), node_to_index)
    print "RA: ", t.stop(), "\n"
    
    t.start()
    lp_p = graphsim.predict_scores(U, graphsim.lp(A, h=katz_h), node_to_index)
    print "LP: ", t.stop(), "\n"
    
    print 'Done.\n\n'
    
    print "\nAUCs:\n"
    
    cn_auc = testing_utils.compute_AUC(Y, cn_p)
    cn_auc_pr = testing_utils.compute_PR_auc(Y, cn_p)
    print "CN: ", cn_auc, "\n"
    print "CN PR: ", cn_auc_pr, "\n"
    
    salton_auc =  testing_utils.compute_AUC(Y, salton_p)
    salton_auc_pr = testing_utils.compute_PR_auc(Y, salton_p)
    print "Salton: ", salton_auc, "\n"
    print "Salton PR: ", salton_auc_pr, "\n"
    
    jacard_auc = testing_utils.compute_AUC(Y, jacard_p)
    jacard_auc_pr = testing_utils.compute_PR_auc(Y, jacard_p)
    print "Jacard: ", jacard_auc, "\n"
    print "Jacard PR: ", jacard_auc_pr, "\n"
    
    sorensen_auc = testing_utils.compute_AUC(Y, sorensen_p)
    sorensen_auc_pr = testing_utils.compute_PR_auc(Y, sorensen_p)
    print "Sorensen: ", sorensen_auc, "\n"
    print "Sorensen PR: ", sorensen_auc_pr, "\n"
    
    hpi_auc = testing_utils.compute_AUC(Y, hpi_p)
    hpi_auc_pr = testing_utils.compute_PR_auc(Y, hpi_p)
    print "HPI: ", hpi_auc, "\n"
    print "HPI PR: ", hpi_auc_pr, "\n"
    
    hdi_auc = testing_utils.compute_AUC(Y, hdi_p)
    hdi_auc_pr = testing_utils.compute_PR_auc(Y, hdi_p)
    print "HDI: ", hdi_auc, "\n"
    print "HDI PR: ", hdi_auc_pr, "\n"
    
    lhn1_auc = testing_utils.compute_AUC(Y, lhn1_p)
    lhn1_auc_pr = testing_utils.compute_PR_auc(Y, lhn1_p)
    print "LHN1: ", lhn1_auc, "\n"
    print "LHN1 PR: ", lhn1_auc_pr, "\n"
    
    pa_auc = testing_utils.compute_AUC(Y, pa_p)
    pa_auc_pr = testing_utils.compute_PR_auc(Y, pa_p)
    print "PA: ", pa_auc, "\n"
    print "PA PR: ", pa_auc_pr, "\n"
    
    aa_auc = testing_utils.compute_AUC(Y, aa_p)
    aa_auc_pr = testing_utils.compute_PR_auc(Y, aa_p)
    print "AA: ", aa_auc, "\n"
    print "AA PR: ", aa_auc_pr, "\n"
    
    ra_auc = testing_utils.compute_AUC(Y, ra_p)
    ra_auc_pr = testing_utils.compute_PR_auc(Y, ra_p)
    print "RA: ", ra_auc, "\n"
    print "RA PR: ", ra_auc_pr, "\n"
    
    lp_auc = testing_utils.compute_AUC(Y, lp_p)
    lp_auc_pr = testing_utils.compute_PR_auc(Y, lp_p)
    print "LP: ", lp_auc, "\n"
    print "LP PR: ", lp_auc_pr, "\n"
    
    print 'The End.'
    
    Y = np.array(Y)
    random_prec = Y[Y.nonzero()].size / Y.size
    
    all_aucs_roc = {"CN": cn_auc, "salton": salton_auc, "jacard": jacard_auc, "sorensen": sorensen_auc,
                "hpi": hpi_auc, "hdi": hdi_auc, "lhn1": lhn1_auc, "pa": pa_auc, "aa": aa_auc,
                "ra": ra_auc, "lp": lp_auc}
    
    all_aucs_pr = {"CN": cn_auc_pr, "salton": salton_auc_pr, "jacard": jacard_auc_pr, "sorensen": sorensen_auc_pr,
                "hpi": hpi_auc_pr, "hdi": hdi_auc_pr, "lhn1": lhn1_auc_pr, "pa": pa_auc_pr, "aa": aa_auc_pr,
                "ra": ra_auc_pr, "lp": lp_auc_pr}
    all_aucs = {"roc": all_aucs_roc, "pr": all_aucs_pr, 'random': random_prec}
    return all_aucs
    
    
def test_global_unsupervised_methods(G, U, Y, node_to_index, options):
    """
    
    Do link prediction with global un-supervised methods and print the auc for each
    un-supervised method.
    
    Parameters:
    -----------
    G: networkx graph object.
    U: a list that holds dyads for testing.
    Y: the true labels for the dyads.
    node_to_index: a dictionary that maps from a node name to a node index.
    options: is a dictionary that holds parameters values. 
            Here we only need: options['rwr_alpha'],
            options['lrw_nSteps'].
    
    Returns:
    --------
    It doesn't return anything, but only prints the auc scores for each method
    to the output.
    """
    t = timer.Timerx()
    rwr_alpha = options['rwr_alpha']
    lrw_nSteps = options['lrw_nSteps']
    A = np.array( nx.adj_matrix(G) )
    
    print "\n\nComputing the auc of the global un-supervised methods...\n"
    
    t.start()
    katz_h = graphsim.katz_h(A)
    katz_h = katz_h * 0.1
    katz_p = graphsim.predict_scores(U, graphsim.katz(A, katz_h), node_to_index)
    print "Katz: ", t.stop(), "\n"
    
    t.start()
    rwr_p = graphsim.RWR_Clf(A, rwr_alpha).score(U, node_to_index) 
    print "RWR: ", t.stop(), "\n"
    
    t.start()
    lrw_p = graphsim.LRW_Clf(A, lrw_nSteps, G.number_of_edges()).score(U, node_to_index)
    print "LRW: ", t.stop(), "\n"
     
    t.start()
    srw_p = graphsim.SRW_Clf(A, lrw_nSteps, G.number_of_edges()).score(U, node_to_index)
    print "SRW: ", t.stop(), "\n"
    
    print 'Done.\n\n'
    
    print "\nAUCs:\n"
    
    katz_auc = testing_utils.compute_AUC(Y, katz_p)
    katz_auc_pr = testing_utils.compute_PR_auc(Y, katz_p)
    print "Katz: ", katz_auc, "\n"
    print "Katz PR: ", katz_auc_pr, "\n"
    
    rwr_auc = testing_utils.compute_AUC(Y, rwr_p)
    rwr_auc_pr = testing_utils.compute_PR_auc(Y, rwr_p)
    print "RWR: ", rwr_auc, "\n"
    print "RWR PR: ", rwr_auc_pr, "\n"
    
    lrw_auc = testing_utils.compute_AUC(Y, lrw_p)
    lrw_auc_pr = testing_utils.compute_PR_auc(Y, lrw_p)
    print "LRW: ", lrw_auc, "\n"
    print "LRW PR: ", lrw_auc_pr, "\n"
    
    srw_auc = testing_utils.compute_AUC(Y, srw_p)
    srw_auc_pr = testing_utils.compute_PR_auc(Y, srw_p)
    print "SRW: ", srw_auc, "\n"
    print "SRW PR: ", srw_auc_pr, "\n"
    
    print "The end.\n"
    
    Y = np.array(Y)
    random_prec = Y[Y.nonzero()].size / Y.size
    
    all_aucs_roc = { "katz": katz_auc, "rwr": rwr_auc, "lrw": lrw_auc, "srw": srw_auc }
    all_aucs_pr = { "katz": katz_auc_pr, "rwr": rwr_auc_pr, "lrw": lrw_auc_pr, "srw": srw_auc_pr, 'random': random_prec }
    
    all_aucs = {"roc": all_aucs_roc, "pr": all_aucs_pr}
    
    return all_aucs


def test_supervised_methods_general(clf, Xs, Ys, X_pps, Y_pps, X_pp_flags, tests_names, plot_file_name):
    """
    
    Train with a specified classifier using the set of datasets.
    It outputs a plot for the aucs.
    
    Parameters:
    -----------
    clf: a sci-kit classifier object.
    Xs: a list of training datasets.
    Ys: a list of labeling vectors.
    X_pps: a list of training datasets of positive dyads.
    Y_pps: a list of labeling vector for the X_pps.
    X_pp_flags: a list of flags that indicate the usage of X_pp for training and testing. 
                Usually it's like this [1,2,3], ie to test with all modes.
    tests_names: a list that contains the names of each test with respect
                 to each dataset in Xs (i.e. the auc curve name).
    plot_file_name: the name of the figure to be saved. (you can prepend the path as well).
    """
    rocs = []
    prs = []
    t = timer.Timerx()
    
    """
    X_pp_flag: a flag to see how to use the X_pp dataset when doing cross validation.
            X_pp_flag: 0==don't use at all, 1==use for training only, 2-use for training and testing
    """
    
    for X_pp_flag in X_pp_flags:
        print '\nTesting with X_pp flag: ', X_pp_flag, "\n"
        for X, Y, X_pp, Y_pp, test_name in zip(Xs, Ys, X_pps, Y_pps, tests_names):
            t.start()
            all_curves = supervised_methods.general_learner(clf, X, Y, X_pp, Y_pp, X_pp_flag, test_name)
            roc = all_curves["roc"]
            pr_cur = all_curves["pr"]
            print "Test name: %s, time: %s, ROC auc: %f\n" % (test_name, t.stop(), roc[3])
            print "PR auc: %f\n" % (pr_cur[3])
            rocs.append(roc)
            prs.append(pr_cur)
        temp_plot_file_name_roc = plot_file_name + '_X_pp_flag_' + str(X_pp_flag) + '.png'
        temp_plot_file_name_pr = plot_file_name + 'PR_X_pp_flag_' + str(X_pp_flag) + '.png'
        testing_utils.draw_rocs(rocs, temp_plot_file_name_roc)
        testing_utils.draw_pr_curves(prs, temp_plot_file_name_pr)
        rocs = []
        prs = []
    
    
def test_supervised_random_walk(G, X, plot_file_name, options_list):
    """
    Applies the Supervised Random Walk method and outputs a plot for the aucs.
    
    Parameters:
    ------------
    G: networkx graph.
    X: node attributes matrix. Each row i is the features vector for node i.
    plot_file_name: the name of the figure to be saved. (you can prepend the path as well).
    
    options_list: a list that contains dictionaries that holds parameters values. 
                  Each dictionary represents a complete test case. All the curves
                  of the test cases will be joined in one plot.
            Example:
            --------
            options1["srw_test_name"], options1["srw_k"], options1["srw_delta"],
            options1["srw_alpha"], options1["srw_iter"].
            options_list = [options1, options2, ..etc]
            
            The parameters in each dictionary means:
            ----------------------------------------
            k: the number of neighbours a node must have in order to be considered a candidate source node.
            delta: the number of edges the candidate source node made in the future that close a triangle. 
                   (i.e. the future/destination node is a friend of a friend, so delta is a threshold that sees how many
                     of these future/destination nodes are friends of current friends). A candidate source node
                     that has a degree above k, and made future friends above delta, then becomes a source node
                     for training.
            alpha: restart probability.
            iter: gradient desecent epochs.
    """
    
    
    rocs = []
    prs = []
    t = timer.Timerx()
    
    for options in options_list:
        test_name = options["srw_test_name"]
        k = options["srw_k"]
        delta = options["srw_delta"]
        alpha = options["srw_alpha"]
        iter = options["srw_iter"]
        t.start()
        all_curves, typical_auc, my_auc = supervised_methods.train_with_srw(G, X, test_name, k, delta, alpha, iter)
        roc = all_curves["roc"]
        pr_curv = all_curves['pr']
        all_prec = all_curves['all_prec']
        all_rec = all_curves['all_rec']
        all_aucs_pr_d = all_curves['all_auc_pr']
        print "Time: ", t.stop(), "\n\n"
        print "typical ROC auc: ", typical_auc, "\n"
        print "my ROC auc: ", my_auc, "\n"
        print "PR AUC: ", pr_curv[3], "\n"
        print "============"
        rocs.append(roc)
        prs.append(pr_curv)
        temp_rocs = [roc]
        temp_prs = [pr_curv]
        temp_file_name_roc = plot_file_name[0:-4] + "_" + test_name + ".png"
        temp_file_name_pr = plot_file_name[0:-4] + "_" + test_name + "_PR.png"
        temp_file_name_pr_folds = plot_file_name[0:-4] + "_" + test_name + "_folds_PR.png"
        testing_utils.draw_rocs(temp_rocs, temp_file_name_roc)
        testing_utils.draw_pr_curves(temp_prs, temp_file_name_pr)
        testing_utils.draw_pr_curves_n_folds(all_curves['n_folds'], test_name, all_prec, all_rec, all_aucs_pr_d, all_curves['random'], temp_file_name_pr_folds)
    
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR_"+".png")
    
    
def test_matrix_fact_normal(G, X, plot_file_name, options_list):  
    """
    Applies the normal factorization model and outputs a plot for the aucs.
    
    Parameters:
    -----------
    G: networkx graph.
    X: node attributes matrix. Each row i is the features vector for node i.
    plot_file_name: the name of the figure to be saved. (you can prepend the path as well).
    options_list: a list that contains dictionaries that holds parameters values. 
                  Each dictionary represents a complete test case. All the curves
                  of the test cases will be joined in one plot. Each dictionary must 
                  have the following parameters: options['mf_test_name']
                  options['mf_n_latent_feats'], options['mf_n_folds'], 
                  options['mf_alpha'], options['mf_n_iter'], options['mf_with_sampling']
    """
    t = timer.Timerx()
    rocs = []
    prs = []
    
    for options in options_list:
        test_name = options['mf_test_name']
        t.start()
        all_curves = supervised_methods.train_matrix_fact_normal(G, X, test_name, options)
        roc = all_curves["roc"]
        pr_cur = all_curves["pr"]
        n_folds_curves = all_curves['n_folds_curves']
        all_prec = n_folds_curves['all_prec']
        all_rec = n_folds_curves['all_rec']
        all_aucs_pr_d = n_folds_curves['all_auc_pr']
        random_prec = n_folds_curves['random']
        print "Test name: %s, time: %s, ROC auc: %f\n" % (test_name, t.stop(), roc[3])
        print "PR auc: %f\n" % (pr_cur[3])
        rocs.append(roc)
        prs.append(pr_cur)
        temp_rocs = [roc]
        temp_prs = [pr_cur]
        temp_file_name_roc = plot_file_name[0:-4] + "_" + test_name + ".png"
        temp_file_name_pr = plot_file_name[0:-4] + "_" + test_name + "_PR.png"
        temp_file_name_pr_folds = plot_file_name[0:-4] + "_" + test_name + "_folds_PR.png"
        testing_utils.draw_rocs(temp_rocs, temp_file_name_roc)
        testing_utils.draw_pr_curves(temp_prs, temp_file_name_pr)
        testing_utils.draw_pr_curves_n_folds(options['mf_n_folds'], test_name, all_prec, all_rec, all_aucs_pr_d, random_prec, temp_file_name_pr_folds)
    
    testing_utils.draw_rocs(rocs, plot_file_name) 
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR.png")
    
    
def test_matrix_fact_ranking(G, X, plot_file_name, options_list):  
    """
    Applies the ranking factorization model and outputs a plot for the aucs.
    
    Parameters:
    -----------
    G: networkx graph.
    X: node attributes matrix. Each row i is the features vector for node i.
    plot_file_name: the name of the figure to be saved. (you can prepend the path as well).
    options_list: a list that contains dictionaries that holds parameters values. 
                  Each dictionary represents a complete test case. All the curves
                  of the test cases will be joined in one plot. Each dictionary must 
                  have the following parameters: options['mf_test_name']
                  options['mf_n_latent_feats'], options['mf_n_folds'], 
                  options['mf_alpha'], options['mf_n_iter']
    """
    t = timer.Timerx()
    rocs = []
    prs = []
    
    for options in options_list:
        test_name = options['mf_test_name']
        t.start()
        all_curves = supervised_methods.train_matrix_fact_ranking(G, X, test_name, options)
        roc = all_curves["roc"]
        pr_cur = all_curves["pr"]
        n_folds_curves = all_curves['n_folds_curves']
        all_prec = n_folds_curves['all_prec']
        all_rec = n_folds_curves['all_rec']
        all_aucs_pr_d = n_folds_curves['all_auc_pr']
        random_prec = n_folds_curves['random']
        print "Test name: %s, time: %s, ROC auc: %f\n" % (test_name, t.stop(), roc[3])
        print "PR auc: %f\n" % (pr_cur[3])
        rocs.append(roc)
        prs.append(pr_cur)
        temp_rocs = [roc]
        temp_prs = [pr_cur]
        temp_file_name_roc = plot_file_name[0:-4] + "_" + test_name + ".png"
        temp_file_name_pr = plot_file_name[0:-4] + "_" + test_name + "_PR.png"
        temp_file_name_pr_folds = plot_file_name[0:-4] + "_" + test_name + "_folds_PR.png"
        testing_utils.draw_rocs(temp_rocs, temp_file_name_roc) 
        testing_utils.draw_pr_curves(temp_prs, temp_file_name_pr)
        testing_utils.draw_pr_curves_n_folds(options['mf_n_folds'], test_name, all_prec, all_rec, all_aucs_pr_d, random_prec, temp_file_name_pr_folds)
    
    testing_utils.draw_rocs(rocs, plot_file_name)
    testing_utils.draw_pr_curves(prs, plot_file_name[0:-4] + "_PR.png")     
    
    
def get_normal_clfs():
    """
    This function returns a list of scikit classifiers to be used for training.
    """
    clfs = []
    
    logistic = linear_model.LogisticRegression()
    clfs.append(logistic) #0
    
    rf1 = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    rf2 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    rf3 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf4 = RandomForestClassifier(n_estimators=150, n_jobs=-1)
    
    clfs.append(rf1) #1
    clfs.append(rf2) #2
    clfs.append(rf3) #3
    clfs.append(rf4) #4
    
#     nn = KNeighborsClassifier()
#     
#     clfs.append(nn)#5
    
    return clfs

def get_deep_clf(options, random_state=0):
    n_components = options['RBM_n_components']
    n_iter = options['RBM_n_iter']
    n_RBMs = options['n_RBMs']
    
    steps = []
    
    logistic = linear_model.LogisticRegression()
    
    for i in xrange(n_RBMs):
        rbm = BernoulliRBM(random_state=random_state, n_components=n_components, n_iter=n_iter)
        name = "rbm" + str(i)
        steps.append( (name, rbm) )
    
    steps.append( ('logistic', logistic) )
    rbm_logistic_clf = Pipeline(steps=steps) 
    return rbm_logistic_clf
    
    
def cross_validation_for_unsupervised_local_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10,
                                                    extended_graphs = False):
    print '\n############### CROSS VALIDATION for Local unsupervised methods #############\n'
    aucs_holder_roc = []
    aucs_holder_pr = []
    
    for i in xrange(n_folds):
        Gx, U, Y, pp_list, np_list, nn_list = \
        graph_utils.prepare_graph_for_training(G, removal_perc, random_state = i)
        
        if extended_graphs:
            Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes)
            original_degrees_list = graphsim.get_degrees_list(Gx, original_degrees)
        else:
            original_degrees_list = None
        
        all_aucs = test_local_unsupervised_methods(Gx, U, Y, node_to_index, options, original_degrees_list)
        aucs_holder_roc.append(all_aucs['roc'])
        aucs_holder_pr.append(all_aucs['pr'])
        
    print '\n############### DONE #################\n'
    
    print "\n############ Cross validation aucs ##########\n"
    
    cn_auc = salton_auc = jacard_auc = sorensen_auc = hpi_auc =\
    hdi_auc = lhn1_auc = pa_auc = aa_auc =\
    ra_auc = lp_auc = 0
    
    cn_auc_pr = salton_auc_pr = jacard_auc_pr = sorensen_auc_pr = hpi_auc_pr =\
    hdi_auc_pr = lhn1_auc_pr = pa_auc_pr = aa_auc_pr =\
    ra_auc_pr = lp_auc_pr = random_prec = 0
    
    for all_aucs in aucs_holder_roc:
        cn_auc += all_aucs["CN"]
        salton_auc += all_aucs["salton"]
        jacard_auc += all_aucs["jacard"]
        sorensen_auc += all_aucs["sorensen"]
        hpi_auc += all_aucs["hpi"]
        hdi_auc += all_aucs["hdi"]
        lhn1_auc += all_aucs["lhn1"]
        pa_auc += all_aucs["pa"]
        aa_auc += all_aucs["aa"]
        ra_auc += all_aucs["ra"]
        lp_auc += all_aucs["lp"]
        
    for all_aucs in aucs_holder_pr:
        cn_auc_pr += all_aucs["CN"]
        salton_auc_pr += all_aucs["salton"]
        jacard_auc_pr += all_aucs["jacard"]
        sorensen_auc_pr += all_aucs["sorensen"]
        hpi_auc_pr += all_aucs["hpi"]
        hdi_auc_pr += all_aucs["hdi"]
        lhn1_auc_pr += all_aucs["lhn1"]
        pa_auc_pr += all_aucs["pa"]
        aa_auc_pr += all_aucs["aa"]
        ra_auc_pr += all_aucs["ra"]
        lp_auc_pr += all_aucs["lp"]
        random_prec += all_aucs["random"]
        
    n_folds = float(n_folds)
    cn_auc /= n_folds
    salton_auc /= n_folds
    jacard_auc /= n_folds
    sorensen_auc /= n_folds
    hpi_auc /= n_folds
    hdi_auc /= n_folds
    lhn1_auc /= n_folds
    pa_auc /= n_folds
    aa_auc /= n_folds
    ra_auc /= n_folds
    lp_auc /= n_folds
    
    cn_auc_pr /= n_folds
    salton_auc_pr /= n_folds
    jacard_auc_pr /= n_folds
    sorensen_auc_pr /= n_folds
    hpi_auc_pr /= n_folds
    hdi_auc_pr /= n_folds
    lhn1_auc_pr /= n_folds
    pa_auc_pr /= n_folds
    aa_auc_pr /= n_folds
    ra_auc_pr /= n_folds
    lp_auc_pr /= n_folds
    random_prec /= n_folds
    
    print "CN: ", cn_auc, "\n"
    print "CN PR: ", cn_auc_pr, "\n"
    
    print "Salton: ", salton_auc, "\n"
    print "Salton PR: ", salton_auc_pr, "\n"
    
    print "Jacard: ", jacard_auc, "\n"
    print "Jacard PR: ", jacard_auc_pr, "\n"
    
    print "Sorensen: ", sorensen_auc, "\n"
    print "Sorensen PR: ", sorensen_auc_pr, "\n"
    
    print "HPI: ", hpi_auc, "\n"
    print "HPI PR: ", hpi_auc_pr, "\n"
    
    print "HDI: ", hdi_auc, "\n"
    print "HDI PR: ", hdi_auc_pr, "\n"
    
    print "LHN1: ", lhn1_auc, "\n"
    print "LHN1 PR: ", lhn1_auc_pr, "\n"
    
    print "PA: ", pa_auc, "\n"
    print "PA PR: ", pa_auc_pr, "\n"
    
    print "AA: ", aa_auc, "\n"
    print "AA PR: ", aa_auc_pr, "\n"
    
    print "RA: ", ra_auc, "\n"
    print "RA PR: ", ra_auc_pr, "\n"
    
    print "LP: ", lp_auc, "\n"
    print "LP PR: ", lp_auc_pr, "\n"
    
    print 'Random PR: ', random_prec, "\n"
    
    print '####### The End. ########'
    
def cross_validation_for_unsupervised_global_methods(G, X_nodes, removal_perc, node_to_index, options, n_folds = 10,
                                                      extended_graphs = False):
    print '\n############### CROSS VALIDATION For Unsupervised Global methods #############\n'
    aucs_holder_roc = []
    aucs_holder_pr = []
    
    for i in xrange(n_folds):
        Gx, U, Y, pp_list, np_list, nn_list = \
        graph_utils.prepare_graph_for_training(G, removal_perc, random_state = i) 
        
        if extended_graphs:
            Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes) 
        
        all_aucs = test_global_unsupervised_methods(Gx, U, Y, node_to_index, options)
        aucs_holder_roc.append(all_aucs['roc'])
        aucs_holder_pr.append(all_aucs['pr'])
        
    print '\n############### DONE #################\n'
    
    print "\n############ Cross validation aucs ##########\n"
    
    
    katz_auc = rwr_auc = lrw_auc = srw_auc = 0
    katz_auc_pr = rwr_auc_pr = lrw_auc_pr = srw_auc_pr = random_prec = 0
    
    for all_aucs in aucs_holder_roc:
        katz_auc += all_aucs["katz"]
        rwr_auc += all_aucs["rwr"]
        lrw_auc += all_aucs["lrw"]
        srw_auc += all_aucs["srw"]
        
    for all_aucs in aucs_holder_pr:
        katz_auc_pr += all_aucs["katz"]
        rwr_auc_pr += all_aucs["rwr"]
        lrw_auc_pr += all_aucs["lrw"]
        srw_auc_pr += all_aucs["srw"]
        random_prec = all_aucs["random"]
        
    n_folds = float(n_folds)
    katz_auc /= n_folds
    rwr_auc /= n_folds
    lrw_auc /= n_folds
    srw_auc /= n_folds
    
    katz_auc_pr /= n_folds
    rwr_auc_pr /= n_folds
    lrw_auc_pr /= n_folds
    srw_auc_pr /= n_folds
    random_prec /= n_folds
    
    print "Katz: ", katz_auc, "\n"
    print "RWR: ", rwr_auc, "\n"
    print "LRW: ", lrw_auc, "\n"
    print "SRW: ", srw_auc, "\n"
    
    print "Katz PR: ", katz_auc_pr, "\n"
    print "RWR PR: ", rwr_auc_pr, "\n"
    print "LRW PR: ", lrw_auc_pr, "\n"
    print "SRW PR: ", srw_auc_pr, "\n"
    print 'Random PR: ', random_prec, "\n"
    
    print '####### The End. ########'

# @profile    
def exp_supervised(clfs, X_pp_flags, G, X_nodes, node_to_index, options, removal_perc,
                    enabled_features, tests_names, plot_path, undersample=False, extended_graphs = False):
    """
    A wrapper code for the supervised classifier experiment
    """
    for clf_id, clf in enumerate(clfs):
        t = timer.Timerx()
        print "Testing with clf id: ", clf_id, "\n"
        for X_pp_flag in X_pp_flags:
            print "Testing with X_pp_flag", X_pp_flag, "\n"
            rocs = []
            prs = []
            
            for enabled_feature, test_name in zip(enabled_features, tests_names):
                temp_enabled_features = [enabled_feature] 
                Xs, Ys, X_pps, Y_pps = \
                testing_utils.build_training_datasets(G, X_nodes, node_to_index, params=options, 
                            edge_removal_perc=removal_perc, enabled_features=temp_enabled_features, 
                            undersample=undersample, extend_graph =extended_graphs )
                for X, Y, X_pp, Y_pp in zip(Xs, Ys, X_pps, Y_pps):
                    print 'one dataset only.\n'
                    t.start()
                    all_curves = supervised_methods.general_learner(clf, X, Y, X_pp, Y_pp, X_pp_flag, test_name)
                    roc = all_curves["roc"]
                    pr_cur = all_curves["pr"]
                    all_prec, all_rec, all_aucs_pr_d = all_curves["all_prec"], all_curves["all_rec"], all_curves["all_auc_pr"]
                    random_perc = all_curves['random']
                    print "Test name: %s, time: %s, ROC auc: %f\n" % (test_name, t.stop(), roc[3])
                    print "PR auc: %f\n" % (pr_cur[3])
                    rocs.append(roc)
                    prs.append(pr_cur)
                    fold_file_name = plot_path + "_clf" + str(clf_id) + "_folds_" + test_name + "_PR_XppFlag" + str(X_pp_flag) + '.png'
                    testing_utils.draw_pr_curves_n_folds(10, test_name, all_prec, all_rec, all_aucs_pr_d, random_perc, fold_file_name)
            plot_file_name_roc = plot_path + "_clf" + str(clf_id) + "_XppFlag" + str(X_pp_flag) + '.png'
            plot_file_name_pr = plot_path + "_clf" + str(clf_id) + "PR_XppFlag" + str(X_pp_flag) + '.png'
            testing_utils.draw_rocs(rocs, plot_file_name_roc)
            testing_utils.draw_pr_curves(prs, plot_file_name_pr)
    
if __name__ == '__main__':
    run_tests()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    