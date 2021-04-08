import numpy as np
import sys

from sklearn.neighbors import KDTree

class KNneighbors:

    def __init__(self, embds : np.ndarray, k : int) -> None:

        self.embds : np.ndarray = np.asarray(list({tuple(elem) for elem in embds}))

        self.knns : np.ndarray = (np.zeros((len(self.embds), k + 1)), np.zeros((len(self.embds), k + 1)))
        self.kmaxs : np.ndarray = np.zeros((len(self.embds), 1))
        self.dst_matrix = np.zeros((len(self.embds), len(self.embds)))

        self.kdt : KDTree = KDTree(self.embds, metric='euclidean')

        for i in range(len(self.embds)):
            
            knns_dist, knns_indx = self.kdt.query([self.embds[i]], k = k + 1)

            self.knns[0][i], self.knns[1][i] = knns_dist[0], knns_indx[0]
            self.kmaxs[i] = max(knns_dist[0])


        # for i in range(len(self.embds)):

        #     knns_index : list = sys.maxsize * np.ones(k + 1)
        #     knns_dist : list = sys.maxsize * np.ones(k + 1)
        #     kmax : float = 0

        #     for j in range(len(self.embds)):
        #         dist : float = np.linalg.norm(self.embds[i] - self.embds[j])

        #         for indx in range(k + 1):
        #             if dist <= knns_dist[indx]:

        #                 tmp_dist : list = knns_dist[indx:-1]
        #                 knns_dist[indx] = dist
        #                 knns_dist[(indx + 1):] = tmp_dist

        #                 tmp_indx : list = knns_index[indx:-1]
        #                 knns_index[indx] = j
        #                 knns_index[(indx + 1):] = tmp_indx
        #                 print(knns_index)
        #                 print(knns_dist)
        #                 break
                        

        #         self.dst_matrix[i][j] = dist

        #     self.knns[0][i], self.knns[1][i] = np.asarray(knns_dist), np.asarray(knns_index)
        #     self.kmaxs[i] = kmax

        # pass



    def get_knn(self, index : int) -> np.ndarray:
        return self.knns[0][index], self.knns[1][index]



    def is_in_hypersphere(self, index : int, sample : tuple) -> int:
        dist : float = np.linalg.norm(self.embds[index] - sample)
        return 1 if dist <= self.kmaxs[index] else 0
