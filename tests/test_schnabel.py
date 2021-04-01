import unittest
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.neighbors import KDTree
from markevaluate import Schnabel as sn
from markevaluate import Utilities

class TestSchnabel(unittest.TestCase):


    # def test_schnabel0(self):
    #     test_sn = sn.Schnabel(k = 2)

    def test_schnabel1(self):
        s_len = 5
        k = 2
        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(len(test_sn.mark(1)), s_len)

    def test_schnabel2(self):
        s_len = 5 
        k = 2
        t = 4

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_sn = sn.Schnabel(set0, set1, k)
        kdt : KDTree = KDTree(arr, metric='euclidean')
        self.assertEqual(len(test_sn.mark_t(t, np.asarray(list(set1)), kdt)), s_len)

    def test_schnabel3(self):
        s_len = 5 
        k = 0

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        kdt : KDTree = KDTree(arr, metric='euclidean')

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.capture_sum(kdt), (k + 1) * s_len)

    # def test_schnabel4(self):
    #     s_len = 5 
    #     k = 2

    #     arr = np.random.rand(s_len, 2)
    #     set0 = {tuple(elem) for elem in arr}
    #     set1 = {tuple(elem) for elem in arr}
    #     kdt : KDTree = KDTree(arr, metric='euclidean')

    #     test_sn = sn.Schnabel(set0, set1, k)
    #     self.assertEqual(test_sn.capture_sum(kdt), (k + 1) * s_len)
        
    # def test_schnabel4(self):
    #     s_len = 5 
    #     k = 2

    #     arr = np.random.rand(s_len, 2)
    #     set0 = {tuple(elem) for elem in arr}
    #     set1 = {tuple(elem) for elem in arr}
    #     kdt : KDTree = KDTree(arr, metric='euclidean')

    #     test_sn = sn.Schnabel(set0, set1, k)
    #     self.assertEqual(test_sn.recapture(), 2 * len(arr) * (k + 1))

    # def test_schnabel3(self):
    #     s_len = 5
    #     k = 2
    #     arr = np.random.rand(s_len, 2)
    #     set0 = {tuple(elem) for elem in arr}
    #     set1 = {tuple(elem) for elem in arr}
    #     test_sn = sn.Schnabel(set0, set1, k)
    #     self.assertEqual(test_sn.recapture(set0, set1), s_len * 2 * (k + 1))