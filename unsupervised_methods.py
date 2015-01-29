import networkx as nx
import numpy as np
import testing_utils
import graph_utils
import graphsim

def calculate_method(method_name, G, X_nodes, n_folds, node_to_index, removal_perc, options, undersample, extended_graphs):
    aucs_holder_roc = []
    aucs_holder_pr = []
    
    for i in xrange(n_folds):
        Gx, U, Y, _, _, _ = \
        graph_utils.prepare_graph_for_training(G, removal_perc, undersample, random_state = i)
        
        if extended_graphs:
            Gx, original_degrees = graph_utils.get_extended_graph(Gx, X_nodes)
            original_degrees_list = graphsim.get_degrees_list(Gx, original_degrees)
        else:
            original_degrees_list = None
        
        roc_auc, pr_auc = compute_a_method(method_name, Gx, U, Y, node_to_index, options, original_degrees_list)
        aucs_holder_roc.append(roc_auc)
        aucs_holder_pr.append(pr_auc)
    
    return aucs_holder_roc, aucs_holder_pr

def compute_a_method(method_name, G, U, Y, node_to_index, options, original_degrees_list):
    degrees = graphsim.get_degrees_list(G)
    lp_katz_h = options['lp_katz_h']
    rwr_alpha = options['rwr_alpha']
    lrw_nSteps = options['lrw_nSteps']
    A = np.array( nx.adj_matrix(G) )
    
    preds = None
    
    if method_name == "CN":
        preds = graphsim.predict_scores(U, graphsim.cn(A), node_to_index)
    elif method_name == "Salton":
        preds = graphsim.predict_scores(U, graphsim.salton(A, degrees), node_to_index)
    elif method_name == "Jacard":
        preds = graphsim.predict_scores(U, graphsim.jacard(A, degrees), node_to_index)
    elif method_name == "Sorensen":
        preds = graphsim.predict_scores(U, graphsim.sorensen(A, degrees), node_to_index)
    elif method_name == "HP":
        preds = graphsim.predict_scores(U, graphsim.hpi(A, degrees), node_to_index)
    elif method_name == "HD":
        preds = graphsim.predict_scores(U, graphsim.hdi(A, degrees), node_to_index)
    elif method_name == "LHN1":
        preds = graphsim.predict_scores(U, graphsim.lhn1(A, degrees), node_to_index)
    elif method_name == "PA":
        preds =  graphsim.predict_scores(U, graphsim.pa(A, degrees), node_to_index)
    elif method_name == "AA":
        if original_degrees_list == None:    
            preds = graphsim.predict_scores(U, graphsim.aa(A, degrees), node_to_index)
        else:
            preds = graphsim.predict_scores(U, graphsim.aa(A, original_degrees_list), node_to_index)
    elif method_name =="RA":
        if original_degrees_list == None: 
            preds = graphsim.predict_scores(U, graphsim.ra(A, degrees), node_to_index)
        else:
            preds = graphsim.predict_scores(U, graphsim.ra(A, original_degrees_list), node_to_index)
    elif method_name == "LP":
        preds = graphsim.predict_scores(U, graphsim.lp(A, h=lp_katz_h), node_to_index)
    elif method_name == "Katz":
        katz_h = graphsim.katz_h(A)
        katz_h = katz_h * 0.1
        preds = graphsim.predict_scores(U, graphsim.katz(A, katz_h), node_to_index)
    elif method_name == "RWR":
        preds = graphsim.RWR_Clf(A, rwr_alpha).score(U, node_to_index) 
    elif method_name == "LRW":
        preds = graphsim.LRW_Clf(A, lrw_nSteps, G.number_of_edges()).score(U, node_to_index)
    elif method_name == "SPRW":
        preds = graphsim.SRW_Clf(A, lrw_nSteps, G.number_of_edges()).score(U, node_to_index)
        
    roc_auc = testing_utils.compute_AUC(Y, preds)
    pr_auc = testing_utils.compute_PR_auc(Y, preds)
    
    return roc_auc, pr_auc
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    