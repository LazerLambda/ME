import itertools
import numpy as np

from sklearn.neighbors import KDTree
from .Estimate import Estimate
from .KNneighbors import KNneighbors as knn


class Schnabel(Estimate):
    """ Computing the ME-Schnabel-estimator
    
    Class to provide the functions to compute the ME-Schnabel-estimator.
    """

    def mark_t(self, t : int) -> set:
        """ Marked function for t != T and t !=1

        Function that captures new samples which are already marked to 
        the t-th marking step.

        Complexity is O(n) / O(t-1). 

        Parameters
        ----------
        t : int
            t-th marking step

        Returns
        -------
        set
            set of already marked samples
        """

        set_tmp = self.mark(t = 1)

        for elem in range((t - 1)):

            knns : set = self.knn1.get_knn_set(elem)
            set_tmp = set_tmp.union(knns)

        return set_tmp



    # T = |S union S'|
    def mark(self, t : int) -> set:
        """ Marked function

        Function to get the sets of the already marked samples.

        Complexity is O(n) (function call).

        Parameters
        ----------
        t : int
            t-th marking step

        Returns
        -------
        set
            set of already marked samples
        """

        if t == 1:
            return self.set0.union({tuple(s1) for s1 in self.knn1.embds if self.knn0.in_hypsphr(tuple(s1)) })
        else:

            return self.mark_t(t)
            


    def capture_sum(self) -> int:
        """ Capture sum function

        This function is using an indicator function to determine if a sample is inside the k-nearest-neighborhood
        instead of the binary function (2) to comply with Theorem A.2.

        Complexity is O(n^2).

        Returns
        -------
        int 
            sum of captured samples
        """

        acc : int = 0
        for index1, s1 in enumerate(self.knn1.embds):
            for index0, s0 in enumerate(self.knn0.embds):
                # Original vs theorem based implementation
                if self.orig:
                    # original
                    acc += self.knn1.is_in_hypersphere(elem=tuple(s0), sample=np.asarray(list(self.knn0.get_knn_set(index1))), k=self.k)
                else:
                    # theorem based
                    acc += self.knn1.in_kngbhd(index=index1, sample=s0)
        return acc



    def capture(self) -> int:
        """ Capture function

        "After all recapture steps, which excludes the first marking step, the number of captured samples will 
        be the number of samples in S′and their respective k’th nearest neighbors as well as samples in S that 
        are inside the hypersphere of each s′[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        Complexity is O(n^2) (function call).

        Returns
        -------
        int 
            amount of total captured samples
        """
        
        return (self.k + 1) * len(self.set1) + self.capture_sum()



    def recapture(self) -> int:
        """ Recapture function

        "Since all samplesinShave been marked in the first marking step, the 
        number of total recaptures is the number of samples in S inside the 
        hypersphere of each s′as well as the number of k-nearest neighbors 
        of the iterated s′thathave already been marked[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        This function is using an indicator function to determine if a sample is 
        inside the k-nearest-neighborhood instead of the binary function (2) 
        to comply with Theorem A.2.
        This function uses the properties of Theorem A.2. in Mordido and Meinel 
        2020 by default. This function is using an indicator function to determine 
        if a sample is inside the k-nearest-neighborhood instead of the binary function.
        The function proposed in the main part of the paper can be called by 
        setting the `orig` paramter to True when calling Schnabel().

        Complexity is O(n^2).

        Returns
        -------
        int
            total amount of recaptured samples

        """

        acc : int = 0
        for index, s1 in enumerate(self.knn1.embds):
            for s0 in self.knn0.embds:
                if self.orig:
                    # original
                    acc += self.knn1.is_in_hypersphere(elem=tuple(s0), sample=np.asarray(list(self.knn1.get_knn_set(index))), k=self.k)
                    acc += len(self.mark(index).intersection(self.knn1.get_knn_set(index=index)))
                else:
                    # theorem based
                    acc += self.knn1.in_kngbhd(index = index, sample = s0)
            if not self.orig:
                # theorem based
                acc += len(self.mark(index).intersection(self.knn1.get_knn_set(index=index)))
        return acc
        


    def estimate(self) -> float:
        """ Estimate function

        Computes the ME-Schnabel-estimator.

        Complexity is O(n^2)

        Returns
        -------
        float
            ME-Schnabel-estimation of the population
        """
        return self.capture() * (len(self.set0) + len(self.set1)) / self.recapture()
 