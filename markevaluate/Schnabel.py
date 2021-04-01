import itertools
import numpy as np

from sklearn.neighbors import KDTree
from markevaluate.Utilities import Utilities as ut


class Schnabel:

    def __init__(self, set0 : set, set1 : set, k : int) -> None:

        self.set0 : set = set0
        self.set1 : set = set1
        self.k = k



    def mark_t(self, t : int, set1_nparr : np.ndarray, kdt : KDTree) -> set:

        set_tmp = self.mark(t = 1)

        for s1 in set1_nparr[:t]:

            k_nearest_neighbors : np.ndarray = ut.k_nearest_neighbor_set(s1, set1_nparr, kdt, self.k)
            set_tmp = set_tmp.union({tuple(elem) for elem in k_nearest_neighbors})

        return set_tmp



    # T = |S union S'|
    def mark(self, t : int) -> set:
        
        if t == 1:
            set0_nparr : np.ndarray = np.asarray(list(self.set0))

            return self.set0.union(set([s1 for s1 in self.set1 if ut.is_in_hypersphere(s1, set0_nparr, k=self.k)]))
        else:
            set1_nparr : np.ndarray = np.asarray(list(self.set1))
            kdt : KDTree = KDTree(set1_nparr, metric='euclidean')

            return self.mark_t(t, set1_nparr, kdt)
            


    def capture_sum(self, kdt : KDTree) -> int:
        # O(n^3)
        acc : int = 0
        for s1 in self.set1:
            for s0 in self.set0:
                knns : np.ndarray = ut.k_nearest_neighbor_set(s1, np.asarray(list(self.set1)), kdt, self.k)
                acc += ut.is_in_hypersphere(s0, knns, k=self.k)
        return acc



    def capture(self, set0 : set, set1 : set) -> int:
        
        kdt : KDTree = KDTree(np.asarray(list(set1)), metric='euclidean')
        return (self.k + 1) * len(self.set1) + self.capture_sum(kdt)



    def recapture(self) -> int:
        # O(n^3)s
        kdt : KDTree = KDTree(np.asarray(list(set1)), metric='euclidean')
        acc : int = 0
        for index, s1 in enumerate(self.set1):
            for s0 in self.set0:
                knns = ut.k_nearest_neighbor_set(s1, np.asarray(list(self.set1)), kdt, self.k)
                acc += ut.is_in_hypersphere(s0, knns, self.k) + \
                    len(self.mark(index).intersection(set(knns)))
        return acc
        


    def estimate(self, set0 : set, set1 : set, k : int) -> float:

        return self.capture(set0, set1) * len(set0.union(set1)) / self.recapture(set0, set1)
 