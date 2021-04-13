import numpy as np
import sys

from typing import Tuple
from sklearn.neighbors import KDTree

class KNneighbors:

    def __init__(self, embds : np.ndarray, k : int) -> None:

        self.embds : np.ndarray = np.asarray(list({tuple(elem) for elem in embds}))

        # dist, index
        self.knns : Tuple[np.ndarray, np.ndarray] = (np.zeros((len(self.embds), k + 1)), np.zeros((len(self.embds), k + 1)))
        self.kmaxs : np.ndarray = np.zeros((len(self.embds), 1))
        self.dst_matrix = np.zeros((len(self.embds), len(self.embds)))

        self.kdt : KDTree = KDTree(self.embds, metric='euclidean')

        for i in range(len(self.embds)):
            
            knns_dist, knns_indx  = self.kdt.query([self.embds[i]], k = k + 1)

            self.knns[0][i], self.knns[1][i] = knns_dist[0], knns_indx[0]
            self.kmaxs[i] = max(knns_dist[0])



    def in_kngbhd(self, index : int, sample : tuple) -> int:
        print(self.kmaxs[index], self.knns[0][index])       
        print(1 if np.linalg.norm(sample - self.embds[index]) <= self.kmaxs[index] else 0)
        return 1 if np.linalg.norm(sample - self.embds[index]) <= self.kmaxs[index] else 0 



    def get_knn(self, index : int) -> Tuple[np.ndarray, np.ndarray]:
        return self.knns[0][index], self.knns[1][index]

    

    def get_knn_set(self, index : int) -> set:
        return {tuple(elem) for elem in self.embds[self.knns[1][index].astype(int)]}



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
