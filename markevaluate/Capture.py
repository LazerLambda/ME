import itertools
import numpy as np

from scipy.optimize import minimize
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



    def maximize_likelihood(self, n : int = 10e10) -> float:
        m_t : int = len(self.set0.union(self.set1))
        c_t : int = self.capture_total()
        t : int = len(self.set0.union(self.set1))
        def neg_likelihood(p):
            return -(\
                np.log(np.math.factorial(p) / np.math.factorial(p - m_t)) #+ \
                # c_t * np.log(c_t) + \
                # (t * p - c_t) * np.log(t * p - c_t) - \
                # t * p * np.log(t * p) \
                )
        max_val : int = minimize(neg_likelihood, [0], method="nelder-mead", options={'xatol':1e-8, }) 
        return max_val