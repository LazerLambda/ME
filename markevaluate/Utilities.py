
import numpy as np

from sklearn.neighbors import KDTree


class Utilities:


    @staticmethod
    def is_in_hypersphere(sample : tuple, arr: np.ndarray, kdt : KDTree, k : int) -> int:
        """Binary function to determine wether a sample is in a hypersphere.

        Binary function that determines wether a sample is in the k-nearest neighborhood of
        at least one element in the provided array arr. Complexity of this function is
        O(n ^ 2 log(n)).

        Parameters
        ----------
        sample : tuple
            sample which is to be checked weather it is in the k-nearest distance or not.
        arr : np.ndarray
            array on which every element will be examined regarding it's neighborhood.
        kdt : KDTree
            KDTree which provides the datastructure the search will be conducted on.
        k : int
            integer, determines how large the neighborhood will be.

        Returns
        -------
        int
            1 if sample is in at least one neighborhood, 0 if not
        """

        for s in arr:
            k_nn_dist, _ = kdt.query([s], k= k + 1)

            mx = np.amax(k_nn_dist)
            dist = np.linalg.norm(sample - s)

            if  dist <= mx:
                return 1
        return 0

    
    @staticmethod
    def is_in(elem : tuple, arr : np.ndarray) -> int:
        """Indicator function

        Simple indicator function. Complexity is O(n)

        Parameters
        ----------
        elem : tuple
            element which is to be checked if it belongs to arr or not
        arr : np.ndarray
            array where it is to be checked if elem is part of it or not
        
        Returns
        -------
        int
            1 if elem is part of arr, 0 otherwise
        """

        return 1 if elem in arr else 0


    @staticmethod
    def accuracy_loss(p_hat : int, p : int) -> float:
        """Accuracy loss function

        Function that returns the accuracy of the population estimate by calculating (p_hat - p) / p,
        which is furthermore bounded to the top with 1. Complexity is O(1).
        
        Parameters
        ----------
        p_hat: int
            population estimate
        p : int
            real population, known before

        Returns
        -------
        float
            accuracy of estimation
        """

        if p == 0:
            return -1
        a : float = abs((p_hat - p) / p)
        return 1 if a > 1 else a

    
    @staticmethod
    def k_nearest_neighbor_set(sample : tuple, arr : np.ndarray, kdt : KDTree, k : int) -> np.ndarray:
        """K-nearest-neighbors of a sample

        Function that returns a numpy array of the k-nearest neighbors of a sample and the sample itself.
        Complexity is O(n log(n)).

        Parameters
        ----------
        sample : tuple
            sample which is an element of the provided array arr
        arr : np.ndarray
            array on which the search is to be conducted on 

        Returns
        np.ndarray
            array which includes the k-nearest neighbors of the provided sample
        -------
        """

        _, k_nearest_neighbor_indices = kdt.query([sample], k = k + 1)

        retVal = arr[k_nearest_neighbor_indices[0]]
        return retVal

