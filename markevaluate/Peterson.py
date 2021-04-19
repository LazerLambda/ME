import numpy as np


from markevaluate.Estimate import Estimate
from markevaluate.KNneighbors import KNneighbors as knn


class Peterson(Estimate):
    """ Computing the ME-Peterson-estimator
    
    Class to provide the functions to compute the ME-Peterson-estimator.
    """

    def mark(self) -> int:
        """ Marking function

        "During the marking step, we mark all samples inside at least one hypersphere of s[...]"
        Mordido, Meinel, 2020: https://arxiv.org/abs/2010.04606

        Complexity is O(n^2).

        Returns
        -------
        int
            number of marked samples 
        """

        return len(self.set0) + sum([self.knn0.in_hypsphr(elem) for elem in self.set1])



    def capture(self) -> int:
        """ Capture function

        Opposite of the `mark` function. 
        
        Complexity is O(n^2).

        Returns
        -------
        int
            number of captured samples 
        """

        return len(self.set1) + sum([self.knn1.in_hypsphr(elem) for elem in self.set0])



    def recapture(self) -> int:
        """ Recapture function

        # TODO

        Returns
        -------
        int
            number of recaptured samples 
        """

        return sum([self.knn0.in_hypsphr(elem) for elem in self.set1]) + \
            sum([self.knn1.in_hypsphr(elem) for elem in self.set0])



    def estimate(self) -> float:
        """ Estimate function

        Computes the ME-Peterson-estimator.

        Complexity is O(n^2)

        Returns
        -------
        float
            ME-Peterson-estimation of the population
        """

        return self.capture() * self.mark() / self.recapture()