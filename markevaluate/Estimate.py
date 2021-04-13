import numpy as np

from markevaluate import KNneighbors as knn



class Estimate():


    def __init__(self, set0 : set, set1 : set, k : int) -> None:

        if len(set0) == 0 or len(set1) == 0:
            exc_str : str = "Set cannot be empty!\n\t\'-> len(set0) == {}, len(set1) == {}".format(len(set0), len(set1))
            raise Exception(exc_str)

        if len(set0) < k or len(set1) < k:
            exc_str : str = "Set cannot be smaller than k!\n\t\'-> len(set0) == {}, len(set1) == {}, k = {}".format(len(set0), len(set1), k)
            raise Exception(exc_str)

        self.knn0 : knn = knn.KNneighbors(np.asarray(list(set0)), k = k)
        self.knn1 : knn = knn.KNneighbors(np.asarray(list(set1)), k = k)
    
        self.set0 : set = set0
        self.set1 : set = set1     

        self.k = k