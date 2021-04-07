import collections
import numpy as np


class KNneighbors:

    def __init__(self, embds : np.ndarray, k : int) -> None:

        self.embds : np.ndarray = np.ndarray(list(set(self.embds)))

        self.knns : np.ndarray = (np.zeros((len(self.embds), k)), np.zeros((len(self.embds), k)))
        self.kmaxs : np.ndarray = np.zeros((len(self.embds), 1))
        self.dst_matrix = np.zeros((len(self.embds), len(self.embds)))

        for i in self.embds:

            knns_index : collections.deque = collections.deque(range(k), maxlen=k)
            knns_dist : collections.deque = collections.deque(range(k), maxlen=k)
            kmax : float = 0

            for j in self.embds:
                dist : float = np.linalg.norm(self.embds[i] - self.embds[j])

                if dist > knns_dist[(k - 1)]:
                    knns_index.append(j)
                    knns_dist.append(dist)

                if dist > kmax:
                    kmax = dist

                self.embds[i][j] = dist

            self.knns[0][i], self.knns[1][i] = np.asarray(knns_index), np.asarray(knns_dist)
            self.kmaxs[i] = kmax
        pass



    def get_knn(self, index : int) -> np.ndarray:
        return self.knns[0][index], self.knns[1][index]



    def is_in_hypersphere(self, index : int, sample : tuple) -> int:
        dist : float = np.linalg.norm(self.embds[index] - sample)
        return 1 if dist < self.kmaxs[index] else 0
