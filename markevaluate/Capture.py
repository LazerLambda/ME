"""Implementation of the CAPTURE estimator for Mark-Evaluate."""

import itertools
import math
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize_scalar, OptimizeResult
from scipy.special import factorial
from .Estimate import Estimate
from . import DataOrg as do


class Capture(Estimate):
    """Computing the ME-CAPTURE-estimator.

    Class to provide the functions to compute the ME-CAPTURE-estimator.
    """

    def capture_total(self) -> int:
        """Capture Total.

        "The total number of captures corresponds to the number of samples in S
        and S′and their respectiveneighbors, as well as the number of samples
        in S inside the hypersphere of a givens′and vice-versa[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        This function uses the properties of Theorem A.3. in Mordido and Meinel
        2020 by default. The respective cardinality of the k-nearest neighbor
        set is only iterated in the outer loop.
        The function proposed in the main part of the paper can be called by
        setting the `orig` paramter to True when calling Capture().

        The complexity is O(n^2)

        Returns:
        -------
        int
            total number of captures as described above
        """
        assert len(self.cand) != 0 and len(self.ref) != 0

        acc: int = 0
        if self.orig:
            # original
            acc += self.data.cand_in_hypsphr_knn() +\
                self.data.ref_in_hypsphr_knn()

            # \sum_{s \in S} \sum_{s' \in S'} |NN_k(s, S)|
            # = |S'| * (k + 1)

            # \sum_{s \in S} \sum_{s' \in S'} |NN_k(s', S')|
            # = |S| * (k + 1)
            acc += len(self.cand) * (self.k + 1) +\
                len(self.ref) * (self.k + 1)
        else:
            # theorem based
            for ic, s1 in enumerate(self.cand):
                for ir, s0 in enumerate(self.ref):
                    acc += (
                        self.data.in_knghbd_ref_cand(ir, ic) +
                        self.data.in_knghbd_cand_ref(ic, ir)
                    )
                # |NN_k(s, S)| = (k + 1)
                acc += 2 * (self.k + 1)
        return acc

    @staticmethod
    def log_factorial(x: int) -> int:
        """Compute log-factorial more efficient."""
        return np.log(np.arange(1, (x + 1))).sum()

    def maximize_likelihood(self, n: int = 10e2) -> float:
        """Maximize likelihood.

        In this function the likelihood is defined
        and also iteratively maximized, starting from
        len(set0) + len(self.set1) based on the assumption
        that we have a concatenation here.

        This function uses the properties of Theorem A.3.
        in Mordido and Meinel 2020 by default. It is assumed
        that M_T(S,S') = |S concat S'| instead of |S union S'|.
        The function proposed in the main part of the paper can
        be called by setting the `orig` paramter to True when
        calling Capture().

        The complexity is O(n).

        Parameters
        ----------
        n : int
            max iterations
        """
        # Difference between theorem based and original implementation
        m_t: int = 0
        t: int = 0
        if self.orig:
            # original
            m_t = len((set([tuple(e) for e in self.ref])).union(
                (set([tuple(e) for e in self.cand]))))
        else:
            # theorem based
            m_t = len(self.ref) + len(self.cand)

        t = m_t

        c_t: int = self.capture_total()

        def likelihood(p):
            # likelihood function

            y: float = (
                self.log_factorial(p) - self.log_factorial((p - m_t)) +
                c_t * np.log(c_t) +
                (t * p - c_t) * np.log(t * p - c_t) -
                t * p * np.log(t * p)
                )
            return y

        if self.orig:
            min_val: int = m_t if m_t > c_t else c_t
        else:
            min_val: int = m_t

        stop: int = n + min_val

        # Iterating over integers
        x: np.ndarray = np.arange(start=min_val, stop=stop, dtype=int)
        x_range: pd.core.frame.Series = pd.Series(x).astype(int)
        y: np.ndarray = x_range.map(likelihood).to_numpy()
        result: int = x_range[np.argmax(y)]

        return result

    def estimate(self) -> int:
        """Estimate function.

        Computes the ME-CAPTURE-estimator.

        Complexity is O(n^2)

        Returns
        -------
        float
            ME-CAPTURE-estimation of the population
        """
        return int(self.maximize_likelihood())
