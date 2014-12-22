from rakg import testing_utils as tstu
import rakg.facebook100_parser as fb_parser
import sys
import numpy as np
from pylearn2.utils import serial
from sklearn import cross_validation

def main():
    args = sys.argv
    choice = int(args[1])
    save_path = "/home/rdirbas/datasets/1/"
    
    #load datasets
    X1_set = serial.load(save_path+"X1_set.pkl")
    X2_set = serial.load(save_path+"X2_set.pkl")
    X3_set = serial.load(save_path+"X3_set.pkl")
    
    X11_set = serial.load(save_path+"X11_set.pkl")
    X22_set = serial.load(save_path+"X22_set.pkl")
    X33_set = serial.load(save_path+"X33_set.pkl")
    
    if choice == 0:
        pass
    elif choice == 1:
        pass
    
def load_set(set_name):
    pass

def pickle_datasets(save_path):
    file_path = "/home/rdirbas/Graph/Graph/data/Caltech36.mat"
    
    SEED = 0
    random_state = SEED
    
    test_size = 0.33
    
    X1, Y1 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="A" )
    X2, Y2 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="B" )
    X3, Y3 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="C" )
    
    X11, Y11 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="A", 
                                raw_features = True )
    X22, Y22 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="B",
                                raw_features = True )
    X33, Y33 = tstu.prepare_training_set(file_path, fb_parser, tstu.standard_features_adder,
                                random_state, edge_removal_perc=0.5, enabled_features="C",
                                raw_features = True )
    
    X1 = (X1 - np.min(X1, 0)) / (np.max(X1, 0) + 0.0001) # 0-1 scaling
    X2 = (X2 - np.min(X2, 0)) / (np.max(X2, 0) + 0.0001) # 0-1 scaling
    X3 = (X3 - np.min(X3, 0)) / (np.max(X3, 0) + 0.0001) # 0-1 scaling
    X11 = (X11 - np.min(X11, 0)) / (np.max(X11, 0) + 0.0001) # 0-1 scaling
    X22 = (X22 - np.min(X22, 0)) / (np.max(X22, 0) + 0.0001) # 0-1 scaling
    X33 = (X33 - np.min(X33, 0)) / (np.max(X33, 0) + 0.0001) # 0-1 scaling
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X1, Y1, test_size=test_size, random_state=random_state)
        
    X1_set = tstu.PlainDataSet()
    X1_set.X = X_train
    X1_set.Y = y_train
    X1_set.X_test = X_test
    X1_set.Y_test = y_test
    
    serial.save(save_path+"X1_set.pkl", X1_set)
    
    #pickling X2
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X2, Y2, test_size=test_size, random_state=random_state)
        
    X2_set = tstu.PlainDataSet()
    X2_set.X = X_train
    X2_set.Y = y_train
    X2_set.X_test = X_test
    X2_set.Y_test = y_test
    
    serial.save(save_path+"X2_set.pkl", X2_set)
    
    #pickling X3
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X3, Y3, test_size=test_size, random_state=random_state)
        
    X3_set = tstu.PlainDataSet()
    X3_set.X = X_train
    X3_set.Y = y_train
    X3_set.X_test = X_test
    X3_set.Y_test = y_test
    
    serial.save(save_path+"X3_set.pkl", X3_set)
    
    
    
    #pickling X11
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X11, Y11, test_size=test_size, random_state=random_state)
        
    X11_set = tstu.PlainDataSet()
    X11_set.X = X_train
    X11_set.Y = y_train
    X11_set.X_test = X_test
    X11_set.Y_test = y_test
    
    serial.save(save_path+"X11_set.pkl", X11_set)


    #pickling X22
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X22, Y22, test_size=test_size, random_state=random_state)
        
    X22_set = tstu.PlainDataSet()
    X22_set.X = X_train
    X22_set.Y = y_train
    X22_set.X_test = X_test
    X22_set.Y_test = y_test
    
    serial.save(save_path+"X22_set.pkl", X22_set)
    
    #pickling X33
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X33, Y33, test_size=test_size, random_state=random_state)
        
    X33_set = tstu.PlainDataSet()
    X33_set.X = X_train
    X33_set.Y = y_train
    X33_set.X_test = X_test
    X33_set.Y_test = y_test
    
    serial.save(save_path+"X33_set.pkl", X33_set)

    
def load_standard_sets(save_path):
    X1 = serial.load(save_path+"X1_set.pkl")
    X2 = serial.load(save_path+"X2_set.pkl")
    X3 = serial.load(save_path+"X3_set.pkl")
    return X1, X2, X3

def load_raw_sets(save_path):
    X11 = serial.load(save_path+"X11_set.pkl")
    X22 = serial.load(save_path+"X22_set.pkl")
    X33 = serial.load(save_path+"X33_set.pkl")
    return X11, X22, X33

main()















