
import numpy as np

from sklearn.neighbors import KDTree


class Utilities:


    @staticmethod
    def is_in_hypersphere(elem : tuple, sample: np.ndarray, kdt : KDTree, k : int) -> int:

        for s in sample:
            k_nn_dist, _ = kdt.query([s], k= k + 1)

            mx = np.amax(k_nn_dist)
            dist = np.linalg.norm(elem - s)

            if  dist <= mx:
                return 1
        return 0

    
    @staticmethod
    def is_in(elem : tuple, sample : np.ndarray) -> int:
        return 1 if elem in sample else 0


    @staticmethod
    def accuracy_loss(p_hat : int, p : int) -> float:
        # Error handling
        if p == 0:
            return -1
        a : float = abs((p_hat - p) / p)
        return 1 if a > 1 else a

    
    @staticmethod
    def k_nearest_neighbor_set(sample : tuple, data_set : np.ndarray, kdt : KDTree, k : int) -> np.ndarray:

        k_nearest_neighbor_indices : np.ndarray = None

        _, k_nearest_neighbor_indices = kdt.query([sample], k = k + 1)

        retVal = data_set[k_nearest_neighbor_indices[0]]
        return retVal

