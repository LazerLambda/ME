import unittest
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.neighbors import KDTree
from scipy import spatial
from markevaluate import Schnabel as sn
from markevaluate import Utilities as ut

class TestUtilities(unittest.TestCase):

    def test_pass(self):
        pass

    # def test_utilities0(self):
    #     s_len = 6
    #     k = 2
    #     arr = np.random.rand(s_len, 2)
    #     set0 = {tuple(elem) for elem in arr}
    #     set1 = {tuple(elem) for elem in arr}

    #     kdt = KDTree(arr, metric="euclidean")
    #     _, knn_index = kdt.query([arr[0]], k = k + 1)
    #     knns = arr[knn_index[0]]
    #     acc = 0
    #     for s0 in set0:
    #         acc = acc + ut.Utilities.is_in(s0, knns)

    #     self.assertEqual(acc, k + 1)

    # def test_utilities1(self):
    #     s_len = 6
    #     k = 2
    #     arr = np.random.rand(s_len, 2)

    #     kdt = KDTree(arr, metric='euclidean')

    #     self.assertEqual(ut.Utilities.is_in_hypersphere(arr[0], arr, kdt, k), True)

    # def test_utilities2(self):
    #     s_len = 6
    #     k = 2
    #     arr = np.random.rand(s_len, 2)

    #     candidates = arr[spatial.ConvexHull(arr).vertices]
    #     dist_mat = spatial.distance_matrix(candidates, candidates)
    #     i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    #     dist = np.linalg.norm(candidates[i] - candidates[j])
    #     new_point = 2 * dist + candidates[i]
    #     kdt = KDTree(arr, metric='euclidean')

    #     self.assertEqual(ut.Utilities.is_in_hypersphere(new_point, arr, kdt, k), False)

if __name__ == '__main__':
    unittest.main()
