import numpy as np
from memory_profiler import profile
from scipy.special import expit

@profile
def main():
    print 'Start'
    
#     X1 = np.random.randn(53263, 11)
#     
#     X2 = np.random.randn(53263, 5)
#     
#     X3 = np.random.randn(53263, 448)
#     
    X4 = np.random.randn(500000, 2000)
#     
#     X5 = np.random.randn(2015, 11)
#     
#     X6 = np.random.randn(2015, 5)
#     
#     X7 = np.random.randn(2015, 448)
#     
#     X8 = np.random.randn(2015, 666)
#     
#     L = [X1, X2, X3, X4, X5, X6, X7, X8]

    
    print 'End.'
    

    
main()