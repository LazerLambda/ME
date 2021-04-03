import unittest
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.neighbors import KDTree
from markevaluate import Schnabel as sn
from markevaluate import Utilities as ut

class TestSchnabel(unittest.TestCase):


    # def test_schnabel0(self):
    #     test_sn = sn.Schnabel(k = 2)

    # def test_schnabel0(self):
    #     self.assertRaises(Exception, sn.Schnabel(set(), set(), k=2))


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

    def test_schnabel4(self):
        ## PROBLEM
        s_len = 5 
        k = 2

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        kdt : KDTree = KDTree(arr, metric='euclidean')

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.capture_sum(kdt), (k + 1) * s_len)

    def test_schnabel5(self):
        s_len = 5 
        k = 2
        t = 3

        arr0 = np.random.rand(s_len, 2)
        arr1 = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        maximum = len(arr0) + sum([ut.Utilities.is_in_hypersphere(elem, arr0, k = k) for elem in arr1]) + t
        self.assertLessEqual(len(test_sn.mark(t)), maximum, msg="Test cardinality of mark function, t != 1 and t != T")

        
    def test_schnabel6(self):
        ## PROBLEM
        s_len = 6 
        k = 2

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.recapture(), len(arr) * len(arr) * (k + 1) + len(arr) * (k + 1), msg="Test recatpture function")