"""Implementation of the Schnabel estimator for Mark-Evaluate."""

import itertools
import numpy as np

from sklearn.neighbors import KDTree
from .Estimate import Estimate
from . import DataOrg as do


class Schnabel(Estimate):
    """Computing the ME-Schnabel-estimator.

    Class to provide the functions to compute the ME-Schnabel-estimator.
    """

    market_one: set = None
    marked_tmp: set = set()

    def mark(self, t: int) -> set:
        """Mark samples.

        Starting with all samples marked from
        the ref set and each sample from the cand
        set which are in the k-nearest-neighborhood
        of a sample from the ref set. Continuing until
        all samples are marked.
        """
        assert t > 0, "t must be greater or equal than 1."

        if t == 1:
            if self.market_one is None:
                self.marked_tmp: set = set([tuple(e) for e in self.ref]).union(
                    {tuple(s1)
                        for i, s1 in enumerate(self.cand)
                        if int(self.data.in_hypsphr_cand(i))})
                self.market_one = self.marked_tmp
            else:
                self.marked_tmp = self.market_one
            return self.marked_tmp
        else:
            knns: set = self.data.get_knn_set_cand(t - 1)
            self.marked_tmp = self.marked_tmp.union(knns)
            return self.marked_tmp

    def capture_sum(self) -> int:
        """Compute sum for capture computation."""
        acc: int = 0
        if self.orig:
            # Original
            acc += self.data.ref_in_hypsphr_knn()
        else:
            # theorem based
            for ic, _ in enumerate(self.cand):
                for ir, s0 in enumerate(self.ref):
                    acc += self.data.in_knghbd_ref_cand(ir, ic)
        return acc

    def capture(self) -> int:
        """Capture function.

        "After all recapture steps, which excludes the first
        marking step, the number of captured samples will be
        the number of samples in S′and their respective k’th
        nearest neighbors as well as samples in S that are
        inside the hypersphere of each s′[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        Complexity is O(n^2) (function call).

        Returns
        -------
        int
            amount of total captured samples
        """
        return\
            (self.k + 1) * len(self.cand) +\
            self.capture_sum()

    def recapture(self) -> int:
        """Recapture function.

        "Since all samplesinShave been marked in the first
        marking step, the number of total recaptures is the
        number of samples in S inside the hypersphere of each
        s′as well as the number of k-nearest neighbors of the
        iterated s′thathave already been marked[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        This function is using an indicator function to determine
        if a sample is inside the k-nearest-neighborhood instead
        of the binary function (2) to comply with Theorem A.2. This
        function uses the properties of Theorem A.2. in Mordido and
        Meinel 2020 by default. This function is using an indicator
        function to determine if a sample is inside the k-nearest-
        neighborhood instead of the binary function. The function
        proposed in the main part of the paper can be called by
        setting the `orig` paramter to True when calling Schnabel().

        Complexity is O(n^2).

        Returns
        -------
        int
            total amount of recaptured samples
        """
        acc: int = 0
        if self.orig:
            acc += self.data.ref_in_hypsphr_knn()
        for ic, _ in enumerate(self.cand):
            # start_time = time.time()
            for ir, s0 in enumerate(self.ref):
                if self.orig:
                    # original
                    knn_tmp: set = self.data.get_knn_set_cand(ic)
                    acc += len(self.mark(ic + 1).
                               intersection(knn_tmp))
                else:
                    # theorem based
                    acc += self.data.in_knghbd_ref_cand(ir, ic)
            if not self.orig:
                # theorem based
                knn_tmp: set = self.data.get_knn_set_cand(ic)
                acc += len(self.mark(ic + 1).
                           intersection(knn_tmp))
            # print("--- %s took %s seconds ---"
            #             % ("FOR LOOP", str(time.time() - start_time)))
        return acc

    def estimate(self) -> float:
        """Estimate function.

        Computes the ME-Schnabel-estimator.

        Complexity is O(n^2)

        Returns
        -------
        float
            ME-Schnabel-estimation of the population
        """
        c: int = self.capture()
        m: int = len(self.ref) + len(self.cand)
        r: int = self.recapture()

        return c * m / r if r != 0 else 0
