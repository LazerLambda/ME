import math
import numpy as np
import sys

from typing import Tuple
from sklearn.neighbors import KDTree

class KNneighbors:
    """Managing k-nearest-neighbors

    class to organize the k-nearest neighbors of each element of the provided set.
    Performance increase due to just one computation of a KDTree. All necessary
    values are then stored and can be easily accessed through indexing.
    """

    def __init__(self, embds : np.ndarray, k : int) -> None:
        """Initializing function

        Building up the datastructure to store the k-nearest-neighbor information.
        The indices, distances and the most distant distance are stored. Calcuation
        through a KDTree with euclidean metric.

        Complexity is O(n*log(n)).

        Parameters
        ----------
        embds : numpy.ndarray
            numpy array of sentence of word embeddings
        k : int
            integer to determine how many neighbors are to be in the neighborhood
        """

        self.embds : np.ndarray = np.asarray(list({tuple(elem) for elem in embds}))

        self.knns_dist : np.ndarray = np.zeros((len(self.embds), k + 1))
        self.knns_indx : np.ndarray = np.zeros((len(self.embds), k + 1))
        self.kmaxs : np.ndarray = np.zeros((len(self.embds), 1))

        self.kdt : KDTree = KDTree(self.embds, metric='euclidean')

        for i in range(len(self.embds)):
            
            knns_dist, knns_indx  = self.kdt.query([self.embds[i]], k = k + 1)

            self.knns_dist[i] = knns_dist[0]
            self.knns_indx[i] = knns_indx[0]
            self.kmaxs[i] = max(knns_dist[0])



    def in_kngbhd(self, index : int, sample : tuple) -> int:
        """ In k-nearest-neighborhood function

        Indicatorfunction that determines weather a sample is in the 
        k-nearest-neighborhood of another a sample. 

        Complexity is O(1).

        Parameters
        ----------
        index : int
            index of the sample which determines the k-nearest-neighborhood
        sample : tuple
            sample which is to be determined wether it is in the neighbor
            hood or not.

        Returns
        -------
        int 
            1 if the sample is in the k-nearest-neighborhood of the sample at index `index`
            0 if not
        """

        dist : float = np.linalg.norm(sample - self.embds[index])
        kmax : float = self.kmaxs[index][0]
        # due to precision issues
        return 1 if dist < kmax or math.isclose(dist, kmax) else 0 

    

    def get_knn_set(self, index : int) -> set:
        """ Getter function for knn

        Getter function to return a set of the k-nearest-neighbors of sample at `index`.
        
        Complexity is O(1).

        Parameters
        ----------
        index : int
            index of the sample

        Returns
        -------
        set
            set of the k-nearest-neighbors
        """

        return {tuple(elem) for elem in self.embds[self.knns_indx[index].astype(int)]}



    def in_hypsphr(self, sample : tuple) -> int:
        """ Binary function for k-nearest-neighborhood

        Function that determines wether the provided sample is in the hypersphere
        of the k-nearest neighbor of an element of the already in the __init__ 
        function provided embeddings.
        
        Complexity is O(n).

        Parameters
        ----------
        sample : tuple
            single embedding which is to be checked on its proximity to the sample at index
        
        Returns
        -------
        int
            1 if the sample is in the k-nearest neighborhood, 0 if not
        """

        for index, embd in enumerate(self.embds):

            dist : float = np.linalg.norm(embd - sample)
            if dist <= self.kmaxs[index]:
                return 1
        return 0
