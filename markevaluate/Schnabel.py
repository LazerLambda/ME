import itertools
import numpy as np

from sklearn.neighbors import KDTree
from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate
from markevaluate.KNneighbors import KNneighbors as knn


class Schnabel(Estimate):



    def mark_t(self, t : int) -> set:

        set_tmp = self.mark(t = 1)

        for elem in range((t - 1)):

            knns : set = self.knn1.get_knn_set(elem)
            set_tmp = set_tmp.union(knns)

        return set_tmp



    # T = |S union S'|
    def mark(self, t : int) -> set:
        
        if t == 1:
            return self.set0.union({tuple(s1) for s1 in self.knn1.embds if self.knn0.in_hypsphr(tuple(s1)) })
        else:

            return self.mark_t(t)
            


    def capture_sum(self) -> int:
        # O(n²)
        acc : int = 0
        for index, _ in enumerate(self.knn1.embds):
            for s0 in self.knn0.embds:
                acc += self.knn1.in_kngbhd(index=index, sample=s0)
        return acc



    def capture(self,) -> int:
        
        return (self.k + 1) * len(self.set1) + self.capture_sum()



    def recapture(self) -> int:
        # O(n^2)s ??

        acc : int = 0
        for index, s1 in enumerate(self.knn1.embds):
            for s0 in self.knn0.embds:
                acc += self.knn1.in_kngbhd(index = index, sample = s0)
            acc += len(self.mark(index).intersection(self.knn1.get_knn_set(index=index)))
        return acc
        


    def estimate(self) -> float:

        knn0 : knn = knn(np.asarray(list(self.set0)), k = self.k)
        knn1 : knn = knn(np.asarray(list(self.set1)), k = self.k)
        return self.capture() * (len(self.set0) + len(self.set1)) / self.recapture()
 