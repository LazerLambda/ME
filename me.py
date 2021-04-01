import itertools
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KDTree
from typing import Tuple


### Schnabel

def is_in_hypersphere(s : np.ndarray, sample: np.ndarray, k : int = 2) -> bool:

        # Error handling
        if k > len(sample):
            raise Error("k cannot be larger than sample!")

        kdt : KDTree = KDTree(sample, metric='euclidean')

        for s_ in sample:
            # get k-nearest neighbor (k + 1 since ||s_ - s_|| = 0)
            k_nearest_neighbor_distances, _ = \
                kdt.query([s_], k = k + 1, return_distance = True)

            # check wether s_ is closer to s than it's k-nearest neighbor
            if np.linalg.norm(s_ - s) <= np.amax(k_nearest_neighbor_distances):
                return True
        return False




class Peterson():
    def __init__(self, reference : np.ndarray, candidate : np.ndarray):
        self.reference = reference
        self.candidate = candidate

    is_in_hypersphere = staticmethod(is_in_hypersphere)

    def estimate(self) -> float:
        mc = lambda s_, s : len(s) + sum([is_in_hypersphere(elem, s) for elem in s_])
        r = lambda s_, s : sum([self.is_in_hypersphere(elem, s) for elem in s_]) + sum([self.is_in_hypersphere(elem, s_) for elem in s])
        return mc(self.reference, self.candidate) * mc(self.candidate, self.reference) / r(self.reference, self.candidate)

    def accuracy_loss(p_hat, p):
        # Errorhandling
        if p == 0:
            return -1
        a = abs((p_hat - p) / p)
        return 1 if a > 1 else a




### Schnabel

class Schnabel():
    @staticmethod
    def k_nearest_neighbor_set(sample : np.ndarray, data_set : np.ndarray, kdt : KDTree, k : int, point_in_data : bool = False) -> np.ndarray:

        k_nearest_neighbor_indices = None

        if point_in_data:
            # taking the quaried point into account
            k_nearest_neighbor_indices = kdt.query([sample], k = k + 1)  
            k_nearest_neighbor_indices = k_nearest_neighbor_indices[1:]
        else:
            k_nearest_neighbor_indices = kdt.query([sample], k = k)

        return data_set[k_nearest_neighbor_indices]

        
    # T = |S union S'|
    @staticmethod
    def mark(t : set, set0 : set, set1 : set, k : int) -> set:
        if t == 1:
            return set0.union(set([s1 for s1 in set1 if is_in_hypersphere(s1, set0)]))
        else:
            kdt : KDTree = KDTree(np.asarray(list(set1)), metric='euclidean')
            set_tmp = set0.union(set([s1 for s1 in set1 if is_in_hypersphere(s1, set0)]))
            for s1 in set(itertools.islice(set1, t - 1)): # range from 0 until t -1 
                k_nearest_neighbors =k_nearest_neighbor_set(s1, set1, kdt, k, point_in_data=True)
                set_temp.union(set(k_nearest_neighbors))
            return set_tmp

    @staticmethod
    def capture(set0 : set, set1 : set, k : int) -> int:
        acc = 0
        kdt : KDTree = KDTree(np.asarray(list(set1)), metric='euclidean')
        for s1 in set1:
            for s0 in set0:

                k_nearest_neighbors  = k_nearest_neighbor_set(s1, set1, kdt, k, point_in_data=True)
                acc += is_in_hypersphere(s0, k_nearest_neighbors)
                
        return (self.k + 1) * len(s1) + acc

    @staticmethod
    def recapture(set0 : set, set1 : set, k : int) -> int:
        acc = 0
        kdt : KDTree = KDTree(np.asarray(list(set1)), metric='euclidean')
        for index, s1 in enumerate(set1):
            for s0 in set0:
                k_nearest_neighbors = k_nearest_neighbor_set(s1, set1, kdt, k, point_in_data=True)
                acc += is_in_hypersphere(s0, k_nearest_neighbors) + \
                    len(self.mark(index, set0, set1, k).\
                    union(set(k_nearest_neighbors)))
        return acc
        
    @staticmethod
    def estimate(set0 : set, set1 : set, k : int) -> float:
        return capture(set0, set1, k) * len(set0.union(set1)) / recapture(set0, set1, k)




#### CAPTURE

def capture_total(set0 : set, set1 : set, k : int) -> int:
    kdt0 : KDTree = KDTree(np.asarray(list(set0)), metric='euclidean')
    kdt1 : KDTree = KDTree(np.asarray(list(set1)), metric='euclidean')
    acc : int = 0
    for s1 in set1:
        for s0 in set0:
            k_nearest_neighbors_0 = k_nearest_neighbor_set(s0, set0, kdt0, k, point_in_data=True)
            k_nearest_neighbors_1 = k_nearest_neighbor_set(s1, set1, kdt1, k, point_in_data=True)
            acc += is_in_hypersphere(s1, k_nearest_neighbors_0) + len(k_nearest_neighbors_0) + \
                    is_in_hypersphere(s0, k_nearest_neighbors_1) + len(k_nearest_neighbors_1)
    return acc

def maximize_likelihood(set0 : set, set1 : set, n : int = 10e10) -> float:
    m_t : int = len(set0.union(set1))
    c_t : int = capture_total(set0, set1)
    t : int = len(set0.union(set1))
    def likelihood(p):
        return -(\
            np.log(np.math.factorial(p) / np.math.factorial(p - m_t)) + \
            c_t * np.log(c_t) + \
            (t * p - c_t) * np.log(t * p - c_t) - \
            t * p * np.log(t * p) \
            )
    int_range = np.arange(n)
    max_val = minimize(likelihood, int_range, method="nelder-mead", options={'xatol':1e-8, }) 
    return 0
