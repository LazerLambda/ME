
import numpy as np


class KNneighbors:

    def __init__(self, embds : np.ndarray):

        embds : np.ndarray = np.ndarray(list(set(embds)))
        
        self.dst_matrix = np.zeros((len(embds), len(embds)))

        for i in embds:
            for j in embds:
                dist = np.linalg.norm(embds[i] - embds[j])
