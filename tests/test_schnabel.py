import unittest
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.neighbors import KDTree
from markevaluate import Schnabel as sn
from markevaluate import Utilities as ut
from markevaluate import KNneighbors as knn

class TestSchnabel(unittest.TestCase):


    # def test_schnabel0(self):
    #     test_sn = sn.Schnabel(k = 2)

    # def test_schnabel0(self):
    #     self.assertRaises(Exception, sn.Schnabel(set(), set(), k=2))
    # knn0 : knn = knn(np.asarray(list(self.set0)), k = self.k)
    # knn1 : knn = knn(np.asarray(list(self.set1)), k = self.k)

    def test_schnabel1(self):
        s_len = 5
        k = 2
        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_sn = sn.Schnabel(set0, set1, k)
        knn0 : knn = knn.KNneighbors(arr, k = k)
        
        # knn1 = knn0
        self.assertEqual(len(test_sn.mark(1, knn0, knn0)), s_len, msg="Test if M(1, S, S') retuns S with len(arr)")

    def test_schnabel2(self):
        s_len = 5 
        k = 2
        t = 4

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_sn = sn.Schnabel(set0, set1, k)

        knn0 : knn = knn.KNneighbors(arr, k = k)

        # knn1 = knn0
        self.assertEqual(len(test_sn.mark_t(t, knn0, knn0)), s_len)

    def test_schnabel3(self):
        s_len = 5 
        k = 0

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        knn0 : knn = knn.KNneighbors(arr, k = k)

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.capture_sum(knn0), (k + 1) * s_len)

    def test_schnabel4(self):
        ## PROBLEM
        s_len = 5 
        k = 2

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        knn1 : knn = knn.KNneighbors(arr, k = k)

        test_sn = sn.Schnabel(set0, set1, k)
        self.assertEqual(test_sn.capture(knn1=knn1), (k + 1) * s_len * 2, msg="Test capture sum")

    def test_schnabel5(self):
        s_len = 5 
        k = 2
        t = 3

        arr0 = np.random.rand(s_len, 2)
        arr1 = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        knn0 : knn = knn.KNneighbors(arr0, k = k)
        knn1 : knn = knn.KNneighbors(arr1, k = k)

        maximum = len(arr0) + sum([knn0.in_hypsphr(elem) for elem in arr1]) + t  #ut.Utilities.is_in_hypersphere(elem, arr0, kdt0, k = k) for elem in arr1]) + t
        self.assertLessEqual(len(test_sn.mark(t, knn0=knn0, knn1=knn1)), maximum, msg="Test cardinality of mark function, t != 1 and t != T")

    
    def test_schnabel6(self):

        k = 1

        arr0 = np.array([[1,1], [2,2], [3,3]])
        arr1 = np.array([[-3,-3], [-4,-4], [-5,-5]])
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        knn0 : knn = knn.KNneighbors(arr0, k = k)
        knn1 : knn = knn.KNneighbors(arr1, k = k)

        self.assertEqual(len(test_sn.mark(1, knn0=knn0, knn1=knn1)), len(arr0), msg="test for two distant sets if the knn is correct and no sample from the other set is in the marked sample")


    def test_schnabel7(self):

        k = 1

        arr0 = np.array([[1,1], [2,2], [3,3]])
        arr1 = np.array([[-3,-3], [-4,-4], [-5,-5]])
        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}
        test_sn = sn.Schnabel(set0, set1, k)

        knn0 : knn = knn.KNneighbors(arr0, k = k)
        knn1 : knn = knn.KNneighbors(arr1, k = k)

        self.assertEqual(len(test_sn.mark(2, knn0=knn0, knn1=knn1)), len(arr0) + 2, \
            msg="test for two distant sets if the union in the marking process is correct and if with t = 2 there is only one sample and its first neighbor in the set") 

        
    def test_schnabel8(self):
        ## PROBLEM
        s_len = 6 
        k = 2

        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}

        knn0 : knn = knn.KNneighbors(arr, k = k)

        test_sn = sn.Schnabel(set0, set1, k)
        # kdt1 = kdt0
        self.assertEqual(test_sn.recapture(knn0=knn0, knn1=knn0), len(arr) * (k + 1) * 2, msg="Test recapture function")

    
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