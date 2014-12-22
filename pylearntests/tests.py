from pylearn2.config import yaml_parse
import pylearn2
from pylearn2.datasets import iris
from pylearn2.models import softmax_regression
from pylearn2.training_algorithms import sgd
import pylearn2.termination_criteria as termination_criteria
from pylearn2 import train
import theano
import rakg.pylearnutils
from rakg import pylearnutils
import sklearn.datasets

def main():
    print 'hi\n'
#     path = "softreg.yaml"
#      
#     with open(path, 'r') as f:
#         yaml_script = f.read()
#          
#     train = yaml_parse.load(yaml_script)
#     print type(train)
#     train.main_loop()
    
    dataset = iris.Iris()
    model = softmax_regression.SoftmaxRegression(n_classes=3, irange=0.05, nvis=4)
    termination_criterion = termination_criteria.EpochCounter(max_epochs =20)
    algorithm = sgd.SGD(learning_rate=1e-1, batch_size=15, 
                                            termination_criterion=termination_criterion)
    trainOb = train.Train(dataset=None, model=model, algorithm=algorithm)
    
    clf = pylearnutils.PylearnClf(trainOb)
    clf.fit(dataset.X, dataset.original_Y)
    
    probs = clf.predict_proba(dataset.X)
    
    print probs
    print probs.shape
    
    print 'done\n'
    
    
def test():
    iris = sklearn.datasets.load_iris()
    X = iris.data
    Y = iris.target
    
    model = softmax_regression.SoftmaxRegression(n_classes=3, irange=0.05, nvis=4)
    termination_criterion = termination_criteria.EpochCounter(max_epochs =20)
    algorithm = sgd.SGD(learning_rate=1e-1, batch_size=15, 
                                            termination_criterion=termination_criterion)
    trainOb = train.Train(dataset=None, model=model, algorithm=algorithm)
    
    clf = pylearnutils.PylearnClf(trainOb)
    print type(clf)
    clf.fit(X, Y)
    probs = clf.predict_proba(X)
    print probs.shape
#     print probs
    
    print 'done'
    
def _1():
    print 'hi\n'
    path = "./ymls/1.yaml"
      
    with open(path, 'r') as f:
        yaml_script = f.read()
          
    train = yaml_parse.load(yaml_script)
    dataset = iris.Iris()
    train.dataset = dataset
    print train.model.nvis
    print type(train)
#     train.main_loop()
    
        
# main()
    
test()
    
# _1()
    
    
    
    
    
    
    
    
    
    
    