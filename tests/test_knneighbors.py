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
            knn_set= test_knns.get_knn_set(i)
            self.assertTrue((knn_set == {tuple(e) for e in arr[kd_index[0]]}))

    def test_knns2(self):
        s_len = 7
        k = 2
        dim = 5

        arr = np.random.rand(s_len, dim)

        test_knns = knn.KNneighbors(arr, k)
        arr = np.asarray(list({tuple(elem) for elem in arr}))
        test_KDTree = KDTree(arr, metric="euclidean")

        for i in range(s_len):
            kd_dist, kd_index = test_KDTree.query([arr[i]], k + 1)
            self.assertTrue(test_knns.kmaxs[i][0], max(kd_dist[0]))

    def test_knns3(self):
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

    def test_knns4(self):
        s_len = 6
        k = 2
        arr = np.random.rand(s_len, 2)

        test_knns = knn.KNneighbors(arr, k)

        self.assertEqual(test_knns.in_hypsphr(tuple(arr[0])), True)

    def test_knns5(self):
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

    def test_knns6(self):
        s_len = 7
        k = 2
        dim = 5

        arr = np.random.rand(s_len, dim)

        test_knns = knn.KNneighbors(arr, k)

        acc = 0
        for i, _ in enumerate(test_knns.embds):
            acc += test_knns.in_kngbhd(index = 0, sample=tuple(arr[i]))

        self.assertEqual(acc, (k + 1), msg="Check wether in_kngbhd function returns exactly k + 1 on two equal sets")

    def test_knns7(self):
        s_len = 7
        k = 2
        dim = 5

        arr = np.random.rand(s_len, dim)

        test_knns = knn.KNneighbors(arr, k)

        acc_indx = 0
        acc_dist = 0
        for i, _ in enumerate(test_knns.embds):
            acc_indx += len(test_knns.knns_indx[i])
            acc_dist += len(test_knns.knns_dist[i])

        self.assertEqual(acc_indx + acc_dist, 2 * s_len * (k + 1), msg="check wether exactly k + 1 indices and distances are been stored.")

    
if __name__ == '__main__':
    unittest.main()