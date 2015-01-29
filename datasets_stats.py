"""
This module contains functions that helps to get statistics about a datasets and also generate 
some visualizations.
"""
from __future__ import division
import pylab
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
import pandas as pd
from pandas import Series
from sklearn import tree
import networkx as nx
import light_srw
import random
import graph_utils
from mpltools import style
style.use('ggplot')

def save_twobar_barchart( v1dataname, v1dic, v1dic_errors, v2dataname, v2dic, v2dic_errors,
                    all_keys, filename, sortby=None ):
#     pr_dic = {"CN": 0.3, "jacard": 0.6, "sorenson": 0.8}
#     roc_dic = {"CN": 0.5, "jacard": 0.8, "sorenson": 0.6}
    
#     pr_dic_err = {"CN": 0.1, "jacard": 0.17, "sorenson": 0.2}
#     roc_dic_err = {"CN": 0.2, "jacard": 0.1, "sorenson": 0.1}

#     all_keys = ['CN', "jacard", "sorenson"]

#     v1dataname = "AU-PR"
#     v2dataname = "AU-ROC"
#     sortby = v2dataname
    
    roc_dic, roc_dic_err, pr_dic, pr_dic_err, all_keys =\
        v1dic, v1dic_errors, v2dic, v2dic_errors, all_keys
    
    roc_data = []
    pr_data = []
    roc_data_err = []
    pr_data_err = []
    
    
    for key in all_keys:
        roc_data.append( roc_dic[key] )
        pr_data.append( pr_dic[key] )
        roc_data_err.append( roc_dic_err[key] )
        pr_data_err.append( pr_dic_err[key] )
        
    roc_data = np.array(roc_data)
    pr_data = np.array(pr_data)
    roc_data_err = np.array(roc_data_err)
    pr_data_err = np.array(pr_data_err)
    
    d = {v1dataname: Series(pr_data, index=all_keys), 
         v2dataname: Series(roc_data, index=all_keys)}
    
    d_err = {v1dataname: Series(pr_data_err, index=all_keys), 
         v2dataname: Series(roc_data_err, index=all_keys)}
    
    
    
    means = pd.DataFrame( d )
    errors = pd.DataFrame( d_err )
#     means = means.sort('AU-PR', ascending=False)
    means = means.sort(sortby, ascending=False)
    means.plot(yerr=errors, kind='bar')
    
#     plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    
    plt.tight_layout()
       
    plt.savefig(filename)
    
def save_onebar_barchart( v1dataname, v1dic, v1dic_errors, all_keys, filename ):
    
    roc_dic, roc_dic_err, all_keys =\
        v1dic, v1dic_errors, all_keys
    
    roc_data = []
    roc_data_err = []
    
    
    for key in all_keys:
        roc_data.append( roc_dic[key] )
        roc_data_err.append( roc_dic_err[key] )
        
    roc_data = np.array(roc_data)
    roc_data_err = np.array(roc_data_err)
    
    d = {v1dataname: Series(roc_data, index=all_keys)}
    
    d_err = {v1dataname: Series(roc_data_err, index=all_keys)}
    
    
    
    means = pd.DataFrame( d )
    errors = pd.DataFrame( d_err )
    sortby = v1dataname
    means = means.sort(sortby, ascending=False)
    ax = means.plot(yerr=errors, kind='bar')
    
    #####
    for i, label in enumerate(list(means.index)):
        score = means.ix[label][v1dataname]
        ax.annotate("{0:.4f}".format(score), (i, score + 0.13), rotation=90)
    
    #####
    
    plt.ylim([-0.05, 1.5])
    
    plt.tight_layout()
       
    plt.savefig(filename)
    
def save_onebar_barchart_no_errbars( v1dataname, v1dic, all_keys, filename ):
    
    roc_dic, all_keys =\
        v1dic, all_keys
    
    roc_data = []
    
    
    for key in all_keys:
        roc_data.append( roc_dic[key] )
        
    roc_data = np.array(roc_data)
    
    d = {v1dataname: Series(roc_data, index=all_keys)}
    
    means = pd.DataFrame( d )
    sortby = v1dataname
    means = means.sort(sortby, ascending=False)
    ax = means.plot(kind='bar')
    
    
    #####
    for i, label in enumerate(list(means.index)):
        score = means.ix[label][v1dataname]
        ax.annotate("{0:.4f}".format(score), (i, score + 0.13), rotation=90)
    
    #####
    
    plt.ylim([-0.05, 1.5])
    
    plt.tight_layout()
       
    plt.savefig(filename)
    
def plot_features_importances(features, importances, plot_file_name):
    
    all_keys_local = ["CN", "LP","Salton", "Jacard", "Sorensen", "HPI", "HDI", "LHN1", "PA", "AA", "RA"]
    all_keys_global = ["Katz", "RWR", "LRW", "SRW"]
    all_keys_loc_glob = ["CN", "LP","Salton", "Jacard", "Sorensen", "HPI", "HDI", "LHN1", "PA", "AA", "RA",
                          "Katz", "RWR", "LRW", "SRW"]
    
    
    
    if len(features) == 1 and features[0] == 1:
        values_dic_loc = {"CN": importances[0], "LP": importances[1],"Salton": importances[2],
                           "Jacard": importances[3], "Sorensen": importances[4],
                           "HPI": importances[5], "HDI": importances[6], "LHN1": importances[7],
                            "PA": importances[8], "AA": importances[9], "RA": importances[10]}
        save_onebar_barchart_no_errbars("Local", values_dic_loc, all_keys_local, plot_file_name+'_importances_local.pdf')
    elif len(features) == 1 and features[0] == 2:
        values_dic_glob = {"Katz": importances[0], "RWR": importances[1], "LRW": importances[2], "SRW": importances[3]}
        save_onebar_barchart_no_errbars("Global", values_dic_glob, all_keys_global, plot_file_name+'_importances_global.pdf')
    elif len(features) == 2:
        values_dic_loc_glob = {"CN": importances[0], "LP": importances[1],"Salton": importances[2],
                           "Jacard": importances[3], "Sorensen": importances[4],
                           "HPI": importances[5], "HDI": importances[6], "LHN1": importances[7],
                            "PA": importances[8], "AA": importances[9], "RA": importances[10],
                            "Katz": importances[11], "RWR": importances[12], "LRW": importances[13], "SRW": importances[14]}
        save_onebar_barchart_no_errbars("Local+Global", values_dic_loc_glob, all_keys_loc_glob, plot_file_name+'_importances_loc_glob.pdf')
    

def save_correlation_plot(X, name):
    """
    Solves the re-name mistake
    """
    save_covariance_plot(X, name)
    
def save_covariance_plot(X, name):
    """
    Visualizes the Correlation matrix for a design matrix.
    
    Parameters:
    -----------
    X:a design matrix where each column is a feature and each row is an observation.
    name: the name of the plot. (you can prepend the path before the name also)
    """
    pylab.clf()
    n_cols = X.shape[1]
    X = X.T
    
    the_labels = []
    
    if n_cols == 11:
        the_labels = ["CN", "LP", "Sal", "Jacard", "Sorensen", "HP", "HD", "LHN1", "PA", "AA", "RA"]
    elif n_cols == 15:
        the_labels = ["CN", "LP", "Sal", "Jac", "Sor", "HPI", "HD", "LHN1", "PA", "AA", "RA",
                      "Katz", "RWR", "LRW", "SRW"]
    elif n_cols == 4:
        the_labels = ["Katz", "RWR", "LRW", "SRW"]
    else:
        the_labels = range(0, n_cols)
    
    
    R = np.corrcoef(X)
    pylab.pcolor(R)
    cb = pylab.colorbar()
    cb.ax.set_ylabel('Correlation Strength', rotation=270, labelpad=25)
    pylab.yticks(np.arange(0, n_cols) + 0.5, the_labels)
    pylab.xticks(np.arange(0, n_cols) + 0.5, the_labels, rotation='vertical')
    pylab.xlim(0, n_cols)
    pylab.ylim(0, n_cols)
    pylab.xlabel("Feature")
    pylab.ylabel("Feature")
    pylab.tight_layout()
    pylab.savefig(name + ".pdf")
    
    print "Printing correlations"
    labels_dics = {}
    
    for i, a_label in enumerate(the_labels):
        labels_dics[i] = a_label
    
    for i in xrange(n_cols):
        print labels_dics[i] +' > 0.75: ',
        temp_ar = np.where(R[i] > 0.75)[0]
        for j in temp_ar:
            print labels_dics[j] +', ',
        print '\n'
        
        print labels_dics[i] +' < 0.75: ',
        temp_ar = np.where(R[i] < 0.75)[0]
        for j in temp_ar:
            print labels_dics[j] +', ',
        print '\n'
        
    print "Done Printing correlations"
    
def save_scatterplot_matix(X, name):    
    """
    Outputs a scatterplot matrix for a design matrix.
    
    Parameters:
    -----------
    X:a design matrix where each column is a feature and each row is an observation.
    name: the name of the plot. (you can prepend the path before the name also)
    """
    n_cols = X.shape[1]
    the_labels = []
    
    if n_cols == 11:
        the_labels = ["CN", "LP", "Sal", "Jac", "Sor", "HP", "HD", "LHN1", "PA", "AA", "RA"]
    elif n_cols == 15:
        the_labels = ["CN", "LP", "Sal", "Jac", "Sor", "HPI", "HD", "LHN1", "PA", "AA", "RA",
                      "Katz", "RWR", "LRW", "SRW"]
    elif n_cols == 4:
        the_labels = ["Katz", "RWR", "LRW", "SRW"]
    else:
        the_labels = range(0, n_cols)
    
    pylab.clf()
    df = pd.DataFrame(X, columns=the_labels)
    axs = scatter_matrix(df, alpha=0.2, diagonal='kde')
    
    for ax in axs[:,0]: # the left boundary
        ax.grid('off', axis='both')
        ax.set_yticks([])
  
    for ax in axs[-1,:]: # the lower boundary
        ax.grid('off', axis='both')
        ax.set_xticks([])
    
    pylab.tight_layout()
    pylab.savefig(name + ".png")
    
    
def get_important_features(X, Y):
    """
    Calculates the importance of each feature.
    
    Parameters:
    -----------
    X: the design matrix.
    Y: labels
    
    Returns:
    -----------
    importances: array that holds the importance value for each feature.
    feature_order: array that holds the indices of the features sorted according to importance.
                    So the first element from this array is the index of the most important feature 
                    and so on. Putting importances[feature_order] will give the importances sorted from 
                    highest to lowest.
    """
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    importances = clf.feature_importances_
    feature_order = np.argsort(-importances)
    
    return importances, feature_order



def get_statistic(G,X):
    """
    Returns a string that represents statistics computed for the graph and design matrix.
    
    It computes the following  stats:
    #nodes, #edges, #non-edges, imbalance ratio, mean degree, #connected components.
    
    Parameters:
    ------------
    G: networkx graph.
    X: node attributes design matrix. Each row is the feature vector of a node.
    
    """
    n_nodes = n_edges = n_non_edges = imbalace_ratio \
         = mean_degree = n_connected_components = 0


    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    N = G.number_of_nodes()
    n_non_edges = ( (N*(N-1))/2 ) - n_edges
    imbalace_ratio = n_non_edges/n_edges

    degrees = []
    
    for node in G.nodes():
        d = G.degree(node)
        degrees.append(d)
    
    degrees = np.array(degrees)
    mean_degree = np.mean(degrees)
    n_connected_components = nx.number_connected_components(G)
    
    
    output = "\n\n#Nodes: %d, #Edges: %d, #Non-edges: %s, \
    Imbalance ratio: 1:%d, Mean Degree: %f, \
    #Connected Components: %d.\n\n" % (n_nodes, n_edges, n_non_edges, imbalace_ratio, 
                                   mean_degree, n_connected_components)
    
    return output
    
    
def srw_stats(G, X, k, delta):
    """
    Return the number of sources, the average number of candidates
    and the average number of destination nodes for the supervised random walk method.
    
    Parameters:
    -----------
    G: networkx graph.
    X: node attributes matrix. Each row i is the features vector for node i.
    test_name: the name to be plotted next to the curve.
    k: the number of neighbours a node must have in order to be considered a candidate source node.
    delta: the number of edges the candidate source node made in the future that close a triangle. 
           (i.e. the future/destination node is a friend of a friend, so delta is a threshold that sees how many
             of these future/destination nodes are friends of current friends). A candidate source node
             that has a degree above k, and made future friends above delta, then becomes a source node
             for training.
             
    Returns:
    --------
    output_string: a string that has the number of sources, the average number of candidates
    and the average number of destination nodes
    """
    
    psi = light_srw.GeneralPSI(G=G, X=X, k=k, delta=delta)
    n_sources = len(psi.get_S()) + len(psi.get_testingS())
    
    n_dests = []
    n_non_dests = []
    
    for source in psi.get_S():
        nd = len( psi.get_D(source) )
        nnd = len( psi.get_L(source) )
        n_dests.append(nd)
        n_non_dests.append(nnd)
    
    n_dests = np.array(n_dests)
    n_non_dests = np.array(n_non_dests)
    
    avg_dests = np.mean(n_dests)
    avg_non_dests = np.mean(n_non_dests)
    
    output = "\n\n#Sources: %d, Avg Destinations: %f, \
            Avg Non-Destinations: %f.\n\n" % (n_sources, avg_dests, avg_non_dests)
    
    return output


def neighbourhood_datasets_stats(G, edge_removal_perc=0.5, random_state=0):
    """
    Calculates statistics for the neibourhood datsets.
    
    G: networkx graph object (before removing any edges).
    edge_removal_perc: how much edges to remove.
    """
    random.seed(random_state)
    U = graph_utils.build_U(G.nodes())
    Gx = G.copy()
    
    num_edges_to_remove = int( edge_removal_perc * Gx.number_of_edges() )
    removed_edges = random.sample( Gx.edges(), num_edges_to_remove )
    Gx.remove_edges_from(removed_edges)
    
    U_n = graph_utils.build_Un_neighborhoods(Gx, U, [2,3,4])
    Y_2 = graph_utils.build_Y(G, U_n[2])
    Y_3 = graph_utils.build_Y(G, U_n[3])
    Y_4 = graph_utils.build_Y(G, U_n[4])
    
    Ys = []
    Ys.append(Y_2)
    Ys.append(Y_3)
    Ys.append(Y_4)
    
    output = "\n\n"
    for i, y in enumerate(Ys):
        num_zeros = len(y[ y == 0 ])
        num_ones = len(y[ y == 1 ])
        if num_ones > 0:
            imbalance = num_zeros / num_ones
        else:
            imbalance = "inf_no_ones!"
        output += 'i: '+ str(i) + ', #Zeros: ' + str(num_zeros) + ", #Ones: " + str(num_ones) + ", Imbalance ratio: " + str(imbalance) + '.\n'
        
    return output





    