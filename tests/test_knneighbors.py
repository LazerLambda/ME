import unittest
import numpy as np
import random

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import KNneighbors as knn
from sklearn.neighbors import KDTree
from scipy import spatial

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
            _, kd_index = test_KDTree.query([arr[i]], k + 1)
            _, knn_index = test_knns.get_knn(i)
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
                test_knns.in_hypsphr(tuple(arr[i])),\
                self.is_in_hypersphere(tuple(arr[i]), arr, test_KDTree, k))

    def test_knns3(self):
        s_len = 6
        k = 2
        arr = np.random.rand(s_len, 2)

        test_knns = knn.KNneighbors(arr, k)

        self.assertEqual(test_knns.in_hypsphr(tuple(arr[0])), True)

    def test_knns4(self):
        s_len = 6
        k = 2
        arr = np.random.rand(s_len, 2)

        candidates = arr[spatial.ConvexHull(arr).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        dist = np.linalg.norm(candidates[i] - candidates[j])
        new_point = 2 * dist + candidates[i]

        test_knns = knn.KNneighbors(arr, k)

        self.assertEqual(test_knns.in_hypsphr(new_point), False)
