import itertools
import math
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize_scalar, OptimizeResult
from scipy.special import factorial
from markevaluate.KNneighbors import KNneighbors as knn
from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate

class Capture(Estimate):
    """ Computing the ME-CAPTURE-estimator
    
    Class to provide the functions to compute the ME-CAPTURE-estimator.
    """


    def capture_total(self) -> int:
        """ Capture Total

        "The total number of captures corresponds to the number of samples in S
        and S′and their respectiveneighbors, as well as the number of samples 
        in S inside the hypersphere of a givens′and vice-versa[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        This function is altered to fit in Theorem A.3. in Mordido and Meinel 2020.
        The respective cardinality of the k-nearest neighbor set is only iterated
        in the outer loop.

        The complexity is O(n^2)

        Returns:
        -------
        int
            total number of captures as described above
        """

        # Errorhandling
        if len(self.set0) == 0 or len(self.set1) == 0:
            return 0

        acc : int = 0
        for index1, s1 in enumerate(self.knn1.embds):
            for index0, s0 in enumerate(self.knn0.embds):
                acc += self.knn0.in_kngbhd(index0, s1) + self.knn1.in_kngbhd(index1, s0)
            acc += 2 * (self.k + 1)
        return acc



    def maximize_likelihood(self, n : int = 10e2) -> float:
        """ Function that maximes the likelihood
        
        In this function the likelihood is defined and also iteratively maximized, starting
        from len(set0) + len(self.set1) based on the assumption that we have a concatenation here.

        This function is altered to fit in Theorem A.3. in Mordido and Meinel 2020. It is assumed
        that M_T(S,S') = |S concat S'| instead of |S union S'|

        The complexity is O(n).

        Parameters
        ----------
        n : int
            max iterations
        """

        ## OWN ASUMPTIONS
        m_t : int = len(self.set0) + len(self.set1)
        c_t : int = self.capture_total()
        t : int = len(self.set0) + len(self.set1)

        def likelihood(p):
            
            def log_factorial(x : int) -> int:
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


        min_val : int = m_t

        # Iterating over integers
        x : np.ndarray = np.arange(start = min_val, stop = n, dtype = int)
        x_range : pd.core.frame.Series = pd.Series(x).astype(int)
        y : np.ndarray = x_range.map(likelihood).to_numpy()
        result : int = x_range[np.argmax(y)]

        return result



    def estimate(self) -> int:
        """ Estimate function

        Computes the ME-CAPTURE-estimator.

        Complexity is O(n^2)

        Returns
        -------
        float
            ME-CAPTURE-estimation of the population
        """
        return int(self.maximize_likelihood())