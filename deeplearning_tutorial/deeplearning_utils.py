import numpy as np
import cPickle, gzip
import theano
import theano.tensor as T

def main():
    print 'hi'
    path = "/Users/rockyrock/Desktop/DeepLearningTutorials/data/mnist.pkl.gz"
    train_set, valid_set, test_set = load_dataset(path)
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    batch_size = 500
    
    #accessing the third minibatch of the training set
#     data = train_set_x[2 * batch_size: 3 * batch_size]
#     label = train_set_y[2 * batch_size: 3 * batch_size]
    print test_set[0].shape
    print valid_set[0].shape
    
    print 'end'
    
def load_dataset(path):
    f = gzip.open(path, 'rb')
    dataset = cPickle.load(f)
    f.close()
    return dataset

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

main()