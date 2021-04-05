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
        kdt0 = KDTree(arr, metric='euclidean')
        # kdt1 = kdt0
        self.assertEqual(len(test_sn.mark(1, kdt0, kdt0)), s_len)

    def test_schnabel2(self):
        s_len = 5 
        k = 2
        t = 4

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_sn = sn.Schnabel(set0, set1, k)
        kdt0 = KDTree(arr, metric='euclidean')
        # kdt1 = kdt0
        self.assertEqual(len(test_sn.mark_t(t, arr, kdt0, kdt0)), s_len)

    def test_schnabel3(self):
        s_len = 5 
        k = 0

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        kdt = KDTree(arr, metric='euclidean')

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.capture_sum(kdt), (k + 1) * s_len)

    def test_schnabel4(self):
        ## PROBLEM
        s_len = 5 
        k = 2

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        kdt1 = KDTree(arr, metric='euclidean')

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.capture(kdt1=kdt1), (k + 1) * s_len * 2, msg="Test capture sum")

    def test_schnabel5(self):
        s_len = 5 
        k = 2
        t = 3

        arr0 = np.random.rand(s_len, 2)
        arr1 = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        kdt0 = KDTree(arr0, metric='euclidean')
        kdt1 = KDTree(arr1, metric='euclidean')

        maximum = len(arr0) + sum([ut.Utilities.is_in_hypersphere(elem, arr0, kdt0, k = k) for elem in arr1]) + t
        self.assertLessEqual(len(test_sn.mark(t, kdt0=kdt0, kdt1=kdt1)), maximum, msg="Test cardinality of mark function, t != 1 and t != T")

    
    def test_schnabel6(self):

        k = 1

        arr0 = np.array([[1,1], [2,2], [3,3]])
        arr1 = np.array([[-3,-3], [-4,-4], [-5,-5]])
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        kdt0 = KDTree(arr0, metric='euclidean')
        kdt1 = KDTree(arr1, metric='euclidean')

        self.assertEqual(len(test_sn.mark(1, kdt0, kdt1)), len(arr0), msg="test for two distant sets if the knn is correct and no sample from the other set is in the marked sample")


    def test_schnabel7(self):

        k = 1

        arr0 = np.array([[1,1], [2,2], [3,3]])
        arr1 = np.array([[-3,-3], [-4,-4], [-5,-5]])
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        kdt0 = KDTree(arr0, metric='euclidean')
        kdt1 = KDTree(arr1, metric='euclidean')

        self.assertEqual(len(test_sn.mark(2, kdt0=kdt0, kdt1=kdt1)), len(arr0) + 2, \
            msg="test for two distant sets if the union in the marking process is correct and if with t = 2 there is only one sample and its first neighbor in the set") 

        
    def test_schnabel8(self):
        ## PROBLEM
        s_len = 6 
        k = 2

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}

        kdt0 = KDTree(arr, metric='euclidean')

        test_sn = sn.Schnabel(set0, set1, k)
        # kdt1 = kdt0
        self.assertEqual(test_sn.recapture(kdt0=kdt0, kdt1=kdt0), len(arr) * (k + 1) * 2, msg="Test recatpture function")

    
    def test_schnabel9(self):

        ## PROBLEM
        s_len = 6 
        k = 1

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.estimate(), 2 * s_len)


    def test_schnabel9(self):
        s_len0 = 10
        s_len1 = 10
        k = 1

        arr0 = np.random.rand(int(s_len0), 2)
        arr1 = np.random.rand(int(s_len1), 2)
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        self.assertGreaterEqual(test_sn.estimate(), s_len0 + s_len1)


if __name__ == '__main__':
    unittest.main()