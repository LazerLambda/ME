import itertools
import math
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize_scalar, OptimizeResult
from scipy.special import factorial
from sklearn.neighbors import KDTree
from markevaluate.KNneighbors import KNneighbors as knn
from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate

class Capture(Estimate):



    def capture_total(self) -> int:
        # Errorhandling
        if len(self.set0) == 0 or len(self.set1) == 0:
            return 0
        knn0 : knn = knn(np.asarray(list(self.set0)), k = self.k)
        knn1 : knn = knn(np.asarray(list(self.set1)), k = self.k)
        acc : int = 0
        for index1, s1 in enumerate(self.set1):
            for index0, s0 in enumerate(self.set0):
                acc += knn0.in_kngbhd(index0, s1) + knn1.in_kngbhd(index1, s0)
        return acc



    def maximize_likelihood(self, n : int = 10e2) -> float:

        ## OWN ASUMPTIONS
        m_t : int = len(self.set0) + len(self.set1) #len(self.set0.union(self.set1))
        c_t : int = self.capture_total()
        t : int = len(self.set0) + len(self.set1) # len(self.set0.union(self.set1))

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
        min_val : int = m_t # if m_t > c_t else c_t

        x : np.ndarray = np.arange(start = min_val, stop = n, dtype = int)
        x_range : pd.core.frame.Series = pd.Series(x).astype(int)
        y : np.ndarray = x_range.map(likelihood).to_numpy()
        result : int = x_range[np.argmax(y)]

        return result



    def estimate(self) -> int:
        return int(self.maximize_likelihood())