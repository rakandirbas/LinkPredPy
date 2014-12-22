"""
This module has an example on how to use SRW.
"""

from hepth_parser import HepthPSI
from light_srw import SRW
import srw
from timer import Timerx
from memory_profiler import profile
import testing_utils
import numpy as np
import sys

# @profile
def main():
    iter = 1
    args = sys.argv

    if len(args) == 2:
        choice = int(args[1])
        iter = choice
   
    print 'Doing ', iter, ' iterations'
    timer = Timerx(True)
    timer.start()
    psi = HepthPSI()
    timer.stop()
    
    alpha = 0.3
    srw_obj = SRW(psi, alpha)
    srw_obj.optimize(iter)
    
    aucs = []
    print 'Ws:', srw_obj.w
    for s in psi.get_testingS():
        P = srw_obj.get_P(s)
        s_index = psi.get_s_index(s)
        probs = srw.rwr(P, s_index, alpha)
        y_test = get_y_test(s, psi)
        auc = testing_utils.compute_AUC(y_test, probs)
        aucs.append(auc)
    
    aucs = np.array(aucs)
    print "\n\nAverage AUC:", np.mean(aucs)

def get_y_test(s, psi):
    """
    Parameters:
    s: the source id
    psi: the psi
    """
    D = psi.get_D(s)
    G = psi.source_nodes_data[s]['G']
    N = G.number_of_nodes()
    y_test = np.zeros((N))
    y_test[D] = 1
    
    return y_test
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    