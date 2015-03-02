"""
The correct version
"""
from __future__ import division
import numpy as np
import networkx as nx
from scipy import optimize
import graph_utils as gu
import random
from sklearn.utils import shuffle
import mathutils
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import interp
import testing_utils

def main():
    """
    To show how to use it!
    """
    print 'Started running...\n'
    
    num_nodes = 100
    num_latent_features = 10
    num_node_features = 5
    G = nx.gnp_random_graph(num_nodes, 0.3) #This should be a real graph!!
    #create a fake matrix that holds the features for each node
    X = np.random.randn( G.number_of_nodes(), num_node_features )
    
    #The real business.
    MF = Matrix_Factorization(k = num_latent_features, G=G, X=X )
    
    print 'Running cross validation on the normal model...'
    
    _, _, n_auc = MF.train_test_normal_model(n_folds = 2, alpha=0.1, n_iter=3, with_sampling = False)
    print "N: ", n_auc
    
    print 'Running cross validation on the ranking model...'
    _, _, r_auc = MF.train_test_ranking_model(n_folds = 2, alpha=0.1, n_iter=1)
    print "R: ", r_auc


class Matrix_Factorization:
    
    def __init__(self, k, G, X=None, Z=None, lambdas=None, random_state=0):
        """
        k: number of latent features.
        G: a networkx graph object.
        X: a 2D numpy array that holds the features for the nodes. So row_i 
            is the feature vector of node_i, where node_i is the node index in 
            the adjacency matrix A.
        Z: an object that implements a method get(node_i,node_j) that returns 
            the feature vector of the edge between node_i and node_j, where node_i and
            node_j are the nodes' indices in the adjacency matrix A.
            It also implements the method number_of_features().
        lambdas: a python dictionary that holds the regularization parameters values for the
                following keys: U, L, B, W, V. If None, then use fixed values in the code.
                L corresponds to the Big delta in the paper, and B corresponds to the nodes biases.
                Otherwise the rest of the names corresponds to the parameters names in the paper.
        random_state: a seed.
        """
        
        self.G = G
        self.X = X
        self.Z = Z
        self.seed = random_state
        self.k = k
        self.n = self.G.number_of_nodes()
        self.lambdas = lambdas
        
        #initializing the weights for the normal model
        #the n lower script stands for the normal model, so U_n is the U matrix of normal model.
        self.U_n = 1/np.sqrt(self.k) * np.random.randn(self.n,self.k) 
        self.UBias_n = np.random.randn(self.n)
        if self.G.is_directed():
            self.L_n = 1/np.sqrt(self.k) * np.random.randn(self.k, self.k) #the big delta in the paper formula
        
        if self.Z != None:
            self.W_n = np.random.randn(self.Z.number_of_features())
            self.WBias_n = np.random.randn()
            
        if self.X != None:
            d = self.X.shape[1] #number of features for the nodes 
            self.V_n = np.random.randn(d, d)
            
            
        #initializing the weights for the ranking model
        # the r lower script stands for the ranking model.
        self.U_r = 1/np.sqrt(self.k) * np.random.randn(self.n,self.k) 
        self.UBias_r = np.random.randn(self.n)
        if self.G.is_directed():
            self.L_r = 1/np.sqrt(self.k) * np.random.randn(self.k, self.k) #the big delta in the paper formula
        
        if self.Z != None:
            self.W_r = np.random.randn(self.Z.number_of_features())
            self.WBias_r = np.random.randn()
            
        if self.X != None:
            d = self.X.shape[1] #number of features for the nodes 
            self.V_r = np.random.randn(d, d)
        
        #initializing the regularization parameters 
        if self.lambdas == None:
            self.lambdas = {}
            self.lambdas['U'] = 0.1
            self.lambdas['L'] = 0
            self.lambdas['B'] = 0
            self.lambdas['W'] = 1e-5
            self.lambdas['V'] = 1e-5
        
        
        
        #a dictionary that convert from the nodes' names to the nodes indices in the adjacency matrix
        self.node_to_index = {}
        self.index_to_node = {}
        for i, node in enumerate(G.nodes()):
            self.node_to_index[node] = i
            self.index_to_node[i] = node
        
    def initialize_n_params(self):
        """
        This function re-initializes all parameters for the normal model when using cross validation.
        """
        #initializing the weights for the ranking model
        # the r lower script stands for the ranking model.
        self.U_r = 1/np.sqrt(self.k) * np.random.randn(self.n,self.k) 
        self.UBias_r = np.random.randn(self.n)
        if self.G.is_directed():
            self.L_r = 1/np.sqrt(self.k) * np.random.randn(self.k, self.k) #the big delta in the paper formula
        
        if self.Z != None:
            self.W_r = np.random.randn(self.Z.number_of_features())
            self.WBias_r = np.random.randn()
            
        if self.X != None:
            d = self.X.shape[1] #number of features for the nodes 
            self.V_r = np.random.randn(d, d)
            
    def initialize_r_params(self):
        """
        This function re-initializes all parameters for the ranking model when using cross validation.
        """
        #initializing the weights for the ranking model
        # the r lower script stands for the ranking model.
        self.U_r = 1/np.sqrt(self.k) * np.random.randn(self.n,self.k) 
        self.UBias_r = np.random.randn(self.n)
        if self.G.is_directed():
            self.L_r = 1/np.sqrt(self.k) * np.random.randn(self.k, self.k) #the big delta in the paper formula
        
        if self.Z != None:
            self.W_r = np.random.randn(self.Z.number_of_features())
            self.WBias_r = np.random.randn()
            
        if self.X != None:
            d = self.X.shape[1] #number of features for the nodes 
            self.V_r = np.random.randn(d, d)
    
    def compute_prediction(self, node1, node2, normal=True):
        """
        Computes the probability that node1 and node2 are linked
        
        Parameters:
        node1: the name of node1 in the network (i.e not its index in the adjacency matrix)
        node2: the name of node2 in the network (not its index)
        normal: boolean. Do you want to compute the predication with respect to the normal model
                or the AUC ranking model? Default is the normal model.
                
        Returns:
        --------
        probability that node1 and node2 are linked.
        """
        
        #Get the nodes indices
        i = self.node_to_index[node1]
        j = self.node_to_index[node2]
        
        if normal:
            U = self.U_n
            UBias = self.UBias_n
            if self.G.is_directed():
                L = self.L_n
            
            if self.Z != None:
                W = self.W_n
                WBias = self.WBias_n
                
            if self.X != None:    
                V = self.V_n
        else:
            U = self.U_r
            UBias = self.UBias_r
            if self.G.is_directed():
                L = self.L_r
            
            if self.Z != None:
                W = self.W_r
                WBias = self.WBias_r
                
            if self.X != None:    
                V = self.V_r
        
        score = 0
        
        if self.G.is_directed():
            u_i = U[i].reshape(-1,1)#reshaping is stupid, i know, but too sleepy to check for dimensions now.
            u_j = U[j].reshape(-1,1)#forgive me, again ...
            
            #compute the latent part
            score = np.dot(np.dot(u_i.T, L), u_j)[0,0] #it will be a 1x1 numpy matrix, so we take [0,0].
        else:
            score = np.dot(U[i], U[j])
            
        b_i = UBias[i]
        b_j = UBias[j]            
        score += b_i + b_j #add the bias terms
        
        #the dyad features
        if self.Z != None:
            z_ij = self.Z.get(i,j)
            score += np.dot(W, z_ij) + WBias
        
        #the bilinear regression model (node features)
        if self.X != None:
            x_i = self.X[i].reshape(-1,1)
            x_j = self.X[j].reshape(-1,1)
            score += np.dot(np.dot(x_i.T, V), x_j)[0,0]
        
        prediction = self.link(score)
        
        return prediction
            
    def cost_n(self, edges, Y):
        """
        Calculates the cost function of the normal factorization model.
        
        Parameters:
        -----------
        edges: the data to compute the cost with respect to. It's a list of tuples (node1, node2),
                where node1 is the node name (not its index).
        Y: the labels of the data.
        """
        error = 0
        for (node1, node2), y in zip(edges, Y):
            prediction = self.compute_prediction(node1, node2, normal=True)
            error += self.loss(prediction - y)
                
        return error
                
        
    def optimize_n(self, edges, Y, alpha=0.1, n_iter=10):
        """
        Optimizes the normal factorization model's parameters,
        using stochastic gradient descent.
        
        Parameters:
        -----------
        edges: the data to optimize with respect to. It's a list of tuples (node1, node2),
                where node1 is the node name (not its index).
        Y: the labels of the data.
        alpha: gradient descent step size (learning rate).
        n_iter: number of gradient descent iterations/epochs.
        """
        
        self.link = mathutils.logistic_function
        self.loss = mathutils.square_loss
        
        U = self.U_n
        UBias = self.UBias_n
        if self.G.is_directed():
            L = self.L_n
        
        if self.Z != None:
            W = self.W_n
            WBias = self.WBias_n
            
        if self.X != None:    
            V = self.V_n
            
        for epoch in xrange(n_iter):
            for (node1, node2), y in zip(edges, Y):
                prediction = self.compute_prediction(node1, node2, normal=True)
                error = prediction - y
                #computing the gradients
#                 gradLink = prediction * (1- prediction) #gradient of the logistic function
#                 gradCommon = 2 * error * gradLink #a common gradient term because of the chain rule.
                gradCommon = error #since we are using the log loss.
                i = self.node_to_index[node1]
                j = self.node_to_index[node2]
                u_i = U[i].reshape(-1,1)
                u_j = U[j].reshape(-1,1)
                if self.G.is_directed():
                    gradU_i = ( gradCommon * np.dot( L, u_j ) ) + ( self.lambdas['U'] * u_i ) #gradU_i's shape is kx1
                    gradU_j = ( gradCommon * np.dot( L.T, u_i ) ) + ( self.lambdas['U'] * u_j )
                    gradL = ( gradCommon * np.dot(u_i, u_j.T) ) + ( self.lambdas['L'] * L )
                else:
                    gradU_i = ( gradCommon * u_j ) + ( self.lambdas['U'] * u_i )
                    gradU_j = ( gradCommon * u_i ) + ( self.lambdas['U'] * u_j )
            
                if self.Z != None:
                    z_ij = self.Z.get(i,j)
                    gradW = (gradCommon * z_ij) + ( self.lambdas['W'] * W )#gradW's shape is (d,)
                    
                if self.X != None: 
                    x_i = self.X[i].reshape(-1,1)
                    x_j = self.X[j].reshape(-1,1)
                    gradV = (gradCommon * np.dot( x_i, x_j.T )) + ( self.lambdas['V'] * V )

#                     gradV = np.dot( x_i, x_j.T )
#                     gradV = gradV + gradV.T
#                     gradV = (gradCommon * gradV) + ( self.lambdas['V'] * V )
                
                gradU_i = gradU_i.reshape(-1)
                gradU_j = gradU_j.reshape(-1) 
                #update parameters
                self.U_n[i] = self.U_n[i] - (alpha * gradU_i)
                self.U_n[j] = self.U_n[j] - (alpha * gradU_j)
                self.UBias_n[i] = self.UBias_n[i] - (alpha * gradCommon)
                self.UBias_n[j] = self.UBias_n[j] - (alpha * gradCommon)
                
                if self.G.is_directed():
                    self.L_n = self.L_n - (alpha * gradL)
                
                if self.Z != None:
                    self.W_n = self.W_n - (alpha * gradW)
                    self.WBias_n = self.WBias_n - (alpha * gradCommon)
                    
                if self.X != None:
                    self.V_n = self.V_n - (alpha * gradV)
            
    def optimize_r(self, quads_edges, alpha=0.1, n_iter=10, loss="log"):
            """
            Optimizes the AUC ranking factorization model's parameters,
            using stochastic gradient descent.
            
            Parameters:
            -------------
            quads_edges: a list of tuples of three elements i.e. (node1, node2, node3), where
                    node1 and node2 represent a positive training example (i.e. they have an edge in the graph) 
                     and node1 and node3 represent a negative training example.
            alpha: gradient descent step size (learning rate).
            n_iter: number of gradient descent iterations/epochs.
            loss: which loss function to use: log or square.
            """
            
            self.link = mathutils.identity_link
            
            
            U = self.U_r
            UBias = self.UBias_r
            if self.G.is_directed():
                L = self.L_r
            
            if self.Z != None:
                W = self.W_r
                WBias = self.WBias_r
                
            if self.X != None:    
                V = self.V_r
                
            for epoch in xrange(n_iter):
                for (node1, node2, node3, node4) in quads_edges:
                    prediction_p = self.compute_prediction(node1, node2, normal=False)
                    prediction_n = self.compute_prediction(node3, node4, normal=False)
                    #computing the gradients
                    if loss == "square":
                        gradCommon = 2 * ( (prediction_p - prediction_n) - 1 )
                    else:
                        gradCommon = mathutils.logistic_function( prediction_p - prediction_n ) - 1
                        
            
                    i = self.node_to_index[node1]
                    j = self.node_to_index[node2]
                    c = self.node_to_index[node3]
                    k = self.node_to_index[node4]
                    u_i = U[i].reshape(-1,1)
                    u_j = U[j].reshape(-1,1)
                    u_c = U[c].reshape(-1,1)
                    u_k = U[k].reshape(-1,1)
                    if self.G.is_directed():
                        gradU_i = ( gradCommon * ( np.dot( L, u_j ) ) ) + ( self.lambdas['U'] * u_i )
                        gradU_j = ( gradCommon * ( np.dot( L.T, u_i ) ) ) + ( self.lambdas['U'] * u_j )
                        gradU_c = ( gradCommon * (- np.dot( L, u_k ) ) ) + ( self.lambdas['U'] * u_c )
                        gradU_k = ( gradCommon * ( - np.dot( L.T, u_c ) ) ) + ( self.lambdas['U'] * u_k )
                        gradL = ( gradCommon * ( np.dot( u_i, u_j.T )  - np.dot( u_c, u_k.T ) ) ) + ( self.lambdas['L'] * L )
                    else:
                        gradU_i = ( gradCommon * ( u_j ) ) + ( self.lambdas['U'] * u_i )
                        gradU_j = ( gradCommon * ( u_i ) ) + ( self.lambdas['U'] * u_j )
                        gradU_c = ( gradCommon * ( -u_k ) ) + ( self.lambdas['U'] * u_c )
                        gradU_k = ( gradCommon * ( -u_c ) ) + ( self.lambdas['U'] * u_k )
                    
                    gradUBias_i = ( gradCommon * ( 0 ) ) #dumb I know
                    gradUBias_j = ( gradCommon * ( 1 ) )
                    gradUBias_c = ( gradCommon * ( -1 ) )
                    gradUBias_k = ( gradCommon * ( -1 ) )
                    
                    if self.Z != None:
                        z_ij = self.Z.get(i,j)
                        z_ck = self.Z.get(c,k)
                        gradW = ( gradCommon * ( z_ij - z_ck ) ) + ( self.lambdas['W'] * W )#gradW's shape is (d,)
                        gradWBias = ( gradCommon * ( 0 ) ) #dumb I know
        
                    if self.X != None: 
                        x_i = self.X[i].reshape(-1,1)
                        x_j = self.X[j].reshape(-1,1)
                        x_c = self.X[c].reshape(-1,1) 
                        x_k = self.X[k].reshape(-1,1) 
                        gradV = ( gradCommon * ( np.dot( x_i, x_j.T ) - np.dot( x_c, x_k.T ) ) ) + ( self.lambdas['V'] * V )
        
                    
                    gradU_i = gradU_i.reshape(-1)
                    gradU_j = gradU_j.reshape(-1)
                    gradU_c = gradU_c.reshape(-1)
                    gradU_k = gradU_k.reshape(-1) 
                    #update parameters
                    self.U_r[i] = self.U_r[i] - (alpha * gradU_i)
                    self.U_r[j] = self.U_r[j] - (alpha * gradU_j)
                    self.U_r[c] = self.U_r[c] - (alpha * gradU_c)
                    self.U_r[k] = self.U_r[k] - (alpha * gradU_k)
                    self.UBias_r[i] = self.UBias_r[i] - (alpha * gradUBias_i)
                    self.UBias_r[j] = self.UBias_r[j] - (alpha * gradUBias_j)
                    self.UBias_r[c] = self.UBias_r[c] - (alpha * gradUBias_c)
                    self.UBias_r[k] = self.UBias_r[k] - (alpha * gradUBias_k)
                    
                    if self.G.is_directed():
                        self.L_r = self.L_r - (alpha * gradL)
                    
                    if self.Z != None:
                        self.W_r = self.W_r - (alpha * gradW)
                        self.WBias_r = self.WBias_r - (alpha * gradWBias)
                        
                    if self.X != None:
                        self.V_r = self.V_r - (alpha * gradV)
        
    def predict_proba(self, test_edges, using_normal):
        """
        Given a test set, predict the probability that edges between two nodes.
        
        Parameters:
        -----------
        test_edges: a list of tuples of the form (node1, node2) where node1 is the node's name 
                NOT its index in the adjacency matrix.
        using_normal: compute the predication using the normal model or the ranking model.
                    True for normal and False for ranking model.
                
        Returns:
        --------
        probas: a 1D numpy array of probabilities for the existence of edges
        """
        probs = []
        for (node1, node2) in test_edges:
            prediction = self.compute_prediction(node1, node2, normal=using_normal)
            probs.append(prediction)
            
        probs = np.array(probs)
        
        return probs
    
    def predict_proba_r(self, test_set):
        """
        Given a test set of *triple edges*, predict the probability that an edge exist between the
        positive node pair and the negative node pair.
        
        Parameters:
        ------------
        test_set: a list of tuples (node1, node2) to compute their predictions. 
        
        Returns:
        ------------
        probas: a 1D numpy array of probabilities for the existence of edges. Given a triple example,
                probas is appended by the probability of the positive example first, then by the negative example.
        """
        
        probs = []
        for (node1, node2) in test_set:
            prediction = self.compute_prediction(node1, node2, normal=False)
            probs.append(prediction)
            
        probs = np.array(probs)
        
        return probs
        
    def train_test_normal_model(self, n_folds = 10, alpha=0.1, n_iter=10, 
                                with_sampling = False, edge_removal_perc=0.3):
        """
        Uses cross-validation to train and test the normal factorization model.
        
        Parameters:
        ------------
        n_folds: the number of cross validation folds.
        alpha: gradient descent step size (learning rate).
        n_iter: number of gradient descent iterations/epochs.
        with_sampling: if true, then make the dataset consists of equal number of positive
                    and negative training data. This is mainly to reduce the number
                    of training data.
        edge_removal_perc: the percentage of edges to hide for testing.
                        
        Returns:
        --------
        [mean_fpr, mean_tpr, mean_auc], [mean_prec, mean_recall, mean_pr_auc]
        """
        random.seed(self.seed)
        kfold = n_folds
    
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        all_aucs = []
        all_aucs_pr = []
        
        all_y_test = []
        all_props = []
        all_prec = {}
        all_rec = {}
        all_aucs_pr_d = {}
        i = 0
        
        mean_recall = mean_prec = 0.0

        for iter in xrange(n_folds):
            
            random_state = random.randint(0,1000)
            train_set, y_train, test_set, y_test = gu.get_train_test_sets(self.G, edge_removal_perc, with_sampling, random_state)
            X_train, Y_train = shuffle(train_set, y_train, random_state=random_state)
            X_test, Y_test = shuffle(test_set, y_test, random_state=random_state)
            
            self.initialize_n_params()
            self.optimize_n(X_train, Y_train, alpha=alpha, n_iter=n_iter)
            probas = self.predict_proba(X_test, using_normal=True)
            fpr, tpr, thresholds = roc_curve(Y_test, probas)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            all_aucs.append( auc(fpr, tpr) )
            
            precision, recall, average_precision = testing_utils.get_PR_curve_values(Y_test, probas)
            all_prec[i] =  precision
            all_rec[i] = recall
            all_aucs_pr.append( average_precision )
            
            all_y_test.extend(Y_test)
            all_props.extend(probas)
            
            i += 1
        
        all_aucs = np.array(all_aucs)
        all_aucs_pr = np.array(all_aucs_pr)
        
        mean_tpr /= kfold
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        mean_prec, mean_recall, mean_pr_auc = testing_utils.get_PR_curve_values(all_y_test, all_props)
#         mean_pr_auc = np.mean(all_aucs_pr)
        all_prec['mean'] = mean_prec
        all_rec['mean'] = mean_recall
        all_aucs_pr_d['mean'] = mean_pr_auc
        all_y_test = np.array(all_y_test)
        random_prec = all_y_test[all_y_test.nonzero()].size / all_y_test.size
        
        print "Cross-validation ROC stats MF-  \
         STD: %f, Variance: %f.\n" % (np.std(all_aucs), np.var(all_aucs))
         
        print "Cross-validation PR auc stats MF-\
         STD: %f, Variance: %f.\n" % (np.std(all_aucs_pr), np.var(all_aucs_pr))
         
        all_curves = {"all_prec": all_prec, "all_rec": all_rec, "all_auc_pr": all_aucs_pr_d, 'random':random_prec}
        
        return [mean_fpr, mean_tpr, mean_auc], [mean_prec, mean_recall, mean_pr_auc], all_curves
        
        
    def train_test_ranking_model(self, n_folds = 10, alpha=0.1, n_iter=10, edge_removal_perc=0.3, with_sampling = False):
        """
        Uses cross-validation to train and test the ranking factorization model.
        
        Parameters:
        ------------
        n_folds: the number of cross validation folds.
        alpha: gradient descent step size (learning rate).
        n_iter: number of gradient descent iterations/epochs.
                        
        Returns:
        --------
        [mean_fpr, mean_tpr, mean_auc], [mean_prec, mean_recall, mean_pr_auc]
        """
        random.seed(self.seed)
        kfold = n_folds
        
    
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        all_aucs = []
        all_aucs_pr = []
        
        all_y_test = []
        all_props = []
        all_prec = {}
        all_rec = {}
        all_aucs_pr_d = {}
        i = 0
        
        mean_recall = mean_prec = 0.0

        for iter in xrange(n_folds):
            random_state = random.randint(0,1000)
            train_set, y_train, test_set, y_test = gu.get_train_test_sets(self.G, edge_removal_perc, with_sampling, random_state)
            X_train, Y_train = shuffle(train_set, y_train, random_state=random_state)
            train_quads = gu.get_train_quads(X_train, Y_train)
            train_quads = shuffle(train_quads, random_state=random_state)
            
            self.initialize_r_params()
            self.optimize_r(train_quads, alpha=alpha, n_iter=n_iter)
            probas = self.predict_proba_r(test_set)
            fpr, tpr, thresholds = roc_curve(y_test, probas)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            all_aucs.append( auc(fpr, tpr) )
            
            precision, recall, average_precision = testing_utils.get_PR_curve_values(y_test, probas)
            all_prec[i] =  precision
            all_rec[i] = recall
            all_aucs_pr_d[i] = average_precision
            all_aucs_pr.append( average_precision )
            
            all_y_test.extend(y_test)
            all_props.extend(probas)
            
            i += 1
       
        all_aucs = np.array(all_aucs)
        all_aucs_pr = np.array(all_aucs_pr)
        
        mean_tpr /= kfold
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        
        mean_prec, mean_recall, mean_pr_auc = testing_utils.get_PR_curve_values(all_y_test, all_props)
#         mean_pr_auc = np.mean(all_aucs_pr)
        all_prec['mean'] = mean_prec
        all_rec['mean'] = mean_recall
        all_aucs_pr_d['mean'] = mean_pr_auc
        all_y_test = np.array(all_y_test)
        random_prec = all_y_test[all_y_test.nonzero()].size / all_y_test.size
        
        print "Cross-validation ROC stats MF-  \
         STD: %f, Variance: %f.\n" % (np.std(all_aucs), np.var(all_aucs))
         
        print "Cross-validation PR auc stats MF-\
         STD: %f, Variance: %f.\n" % (np.std(all_aucs_pr), np.var(all_aucs_pr))
         
         
        all_curves = {"all_prec": all_prec, "all_rec": all_rec, "all_auc_pr": all_aucs_pr_d, 'random':random_prec}
        
        return [mean_fpr, mean_tpr, mean_auc], [mean_prec, mean_recall, mean_pr_auc], all_curves
        
        
    

if __name__ == '__main__':
    main()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        