import itertools
import math
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize_scalar, OptimizeResult
from scipy.special import factorial
from sklearn.neighbors import KDTree
from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate

class Capture(Estimate):



    def capture_total(self) -> int:
        # Errorhandling
        if len(self.set0) == 0 or len(self.set1) == 0:
            return 0
        kdt0 : KDTree = KDTree(np.asarray(list(self.set0)), metric='euclidean')
        kdt1 : KDTree = KDTree(np.asarray(list(self.set1)), metric='euclidean')
        acc : int = 0
        for s1 in self.set1:
            for s0 in self.set0:
                knns0 : np.ndarray = ut.k_nearest_neighbor_set(s0, np.asarray(list(self.set0)), kdt0, self.k)
                knns1 : np.ndarray = ut.k_nearest_neighbor_set(s1, np.asarray(list(self.set1)), kdt1, self.k)
                acc += ut.is_in(s1, knns0) + len(knns0) + \
                        ut.is_in(s0, knns1) + len(knns1)
        return acc



    def maximize_likelihood(self, n : int = 10e2) -> float:

        m_t : int = len(self.set0.union(self.set1))
        c_t : int = self.capture_total()
        t : int = len(self.set0.union(self.set1))

        def likelihood(p):
            
            def log_factorial(x : int) -> int:
                # loop = lambda x, acc, counter : loop(x, (acc + math.log(counter)), (counter + 1)) if counter < x + 1 else acc
                # return loop(x, 0, 1)
                acc : int = 0
                for e in np.arange(1, (x + 1)):
                    acc += np.log(e)
                return acc

            return (\
                log_factorial(p) - log_factorial((p - m_t)) + \
                c_t * np.log(c_t) + \
                (t * p - c_t) * np.log(t * p - c_t) - \
                t * p * np.log(t * p) \
                )

        # lower_bound : int = m_t
        # optim_result : OptimizeResult = minimize_scalar(neg_likelihood, bounds = (lower_bound, sys.maxsize), method = 'bounded', options={'maxiter' : 500, 'disp': True})
        
        # max_val : int = -1
        
        # if optim_result.success:
        #     max_val = int(optim_result.x) # ceiling fct ?
        # else:
        #     # No termination
        #     print(optim_result.message)



        ## PROBLEM
        min_val : int = m_t if m_t > c_t else c_t

        x : np.ndarray = np.arange(start = min_val, stop = n, dtype = int)
        x : pd.core.frame.Series= pd.Series(x).astype(int)
        y : np.ndarray = x.map(likelihood).to_numpy()
        result : int = x[np.argmax(y)]

        return result



    def estimate(self) -> int:
        return 0