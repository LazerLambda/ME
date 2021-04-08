import unittest
import numpy as np
import random

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import KNneighbors as knn
from sklearn.neighbors import KDTree

class TestKNneighbors(unittest.TestCase):

    @staticmethod
    def is_in_hypersphere(sample : tuple, arr: np.ndarray, kdt : KDTree, k : int) -> int:

        for s in arr:
            k_nn_dist, _ = kdt.query([s], k= k + 1)

            mx = np.amax(k_nn_dist)
            dist = np.linalg.norm(sample - s)

            if  dist <= mx:
                return 1
        return 0

    def test_knns1(self):
        s_len = 7
        k = 2
        dim = 5

        arr = np.random.rand(s_len, dim)

        test_knns = knn.KNneighbors(arr, k)
        arr = np.asarray(list({tuple(elem) for elem in arr}))
        test_KDTree = KDTree(arr, metric="euclidean")

        for i in range(s_len):
            kd_dist, kd_index = test_KDTree.query([arr[i]], k + 1)
            knn_dist, knn_index = test_knns.get_knn(i)
            self.assertTrue((knn_index == kd_index[0]).all())

    def test_knns2(self):
        s_len = 7
        k = 2
        dim = 5

        arr = np.random.rand(s_len, dim)

        test_knns = knn.KNneighbors(arr, k)
        arr = np.asarray(list({tuple(elem) for elem in arr}))
        test_KDTree = KDTree(arr, metric="euclidean")

        for i in range(s_len):
            self.assertEqual(\
                test_knns.is_in_hypersphere(i, tuple(arr[i])),\
                self.is_in_hypersphere(tuple(arr[i]), arr, test_KDTree, k))