import itertools
import numpy as np

from sklearn.neighbors import KDTree
from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate
from markevaluate.KNneighbors import KNneighbors as knn


class Schnabel(Estimate):



    def mark_t(self, t : int, knn0 : knn, knn1 : knn) -> set:

        set_tmp = self.mark(t = 1, knn0 = knn0, knn1 = knn1)

        for elem in range((t - 1)):

            knns : set = knn1.get_knn_set(elem)
            set_tmp = set_tmp.union(knns)

        return set_tmp



    # T = |S union S'|
    def mark(self, t : int, knn0 : knn, knn1 : knn) -> set:
        
        if t == 1:
            return self.set0.union(set([s1 for s1 in self.set1 if knn0.in_hypsphr(s1) ]))
        else:

            return self.mark_t(t, knn0 , knn1)
            


    def capture_sum(self, knn1 : knn) -> int:
        # O(n²)
        acc : int = 0
        for index in range(len(self.set1)):
            for s0 in self.set0:
                acc += knn1.in_kngbhd(index=index, sample=s0)
        return acc



    def capture(self, knn1 : knn) -> int:
        
        return (self.k + 1) * len(self.set1) + self.capture_sum(knn1)



    def recapture(self, knn0 : knn, knn1 : knn) -> int:
        # O(n^2)s ??

        acc : int = 0
        for index in range(len(self.set1)):
            for s0 in self.set0:
                acc += knn1.in_kngbhd(index = index, sample = s0)
            acc += len(self.mark(index, knn0, knn1).intersection(knn1.get_knn_set(index=index)))
        return acc
        


    def estimate(self) -> float:

        knn0 : knn = knn(np.asarray(list(self.set0)), k = self.k)
        knn1 : knn = knn(np.asarray(list(self.set1)), k = self.k)
        return self.capture(knn1=knn1) * (len(self.set0) + len(self.set1)) / self.recapture(knn0, knn1)
 


    #  def mark_t(self, t : int, set1_nparr : np.ndarray, kdt0 : KDTree, kdt1 : KDTree) -> set:

    #     set_tmp = self.mark(t = 1, kdt0 = kdt0, kdt1 = kdt1)

    #     for s1 in set1_nparr[:(t - 1)]:

    #         knns : np.ndarray = ut.k_nearest_neighbor_set(s1, set1_nparr, kdt1, self.k)
    #         set_tmp = set_tmp.union({tuple(elem) for elem in knns})

    #     return set_tmp



    # # T = |S union S'|
    # def mark(self, t : int, kdt0 : KDTree, kdt1 : KDTree) -> set:
        
    #     if t == 1:
    #         set0_nparr : np.ndarray = np.asarray(list(self.set0))

    #         return self.set0.union(set([s1 for s1 in self.set1 if ut.is_in_hypersphere(s1, set0_nparr, kdt0, k=self.k)]))
    #     else:
    #         set1_nparr : np.ndarray = np.asarray(list(self.set1))

    #         return self.mark_t(t, set1_nparr, kdt0 , kdt1)
            

    # def capture_sum(self, kdt1 : KDTree) -> int:
    #     # O(n^2 * n^(1 - 1 / k) * k)
    #     acc : int = 0
    #     for s1 in self.set1:
    #         for s0 in self.set0:
    #             knns : np.ndarray = ut.k_nearest_neighbor_set(s1, np.asarray(list(self.set1)), kdt1, self.k)
    #             acc += ut.is_in(s0, knns)
    #     return acc



    # def capture(self, kdt1 : KDTree) -> int:
        
    #     return (self.k + 1) * len(self.set1) + self.capture_sum(kdt1)



    # def recapture(self, kdt0 : KDTree, kdt1 : KDTree) -> int:
    #     # O(n^2)s ??

    #     acc : int = 0
    #     for index, s1 in enumerate(self.set1):
    #         knns = ut.k_nearest_neighbor_set(s1, np.asarray(list(self.set1)), kdt1, self.k)
    #         for s0 in self.set0:
    #             acc += ut.is_in(s0, knns)
    #         acc += len(self.mark(index, kdt0, kdt1).intersection(set({tuple(elem) for elem in knns})))
    #     return acc
        


    # def estimate(self) -> float:
    #     kdt0 : KDTree = KDTree(np.asarray(list(self.set0)), metric='euclidean')
    #     kdt1 : KDTree = KDTree(np.asarray(list(self.set1)), metric='euclidean')

    #     return self.capture(kdt1) * (len(self.set0) + len(self.set1)) / self.recapture(kdt0, kdt1)
 