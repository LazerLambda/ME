import unittest
import numpy as np

import os
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from sklearn.neighbors import KDTree
from markevaluate import Schnabel as sn
from markevaluate import DataOrg as do

class TestSchnabel(unittest.TestCase):

    def test_schnabel1(self):
        s_len = 5
        k = 2
        arr = np.random.rand(s_len, 2)

        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org)
        self.assertEqual(len(test_sn.mark(1)), s_len, msg="Test if M(1, S, S') retuns S with len(arr)")

    def test_schnabel2(self):
        s_len = 5 
        k = 2
        t = 4

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org)
        
        for i in range(t - 1):
            test_sn.mark(i + 1)
        self.assertEqual(len(test_sn.mark(t)), s_len)

    def test_schnabel3(self):
        s_len = 5 
        k = 1

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org)
        self.assertEqual(test_sn.capture_sum(), (k + 1) * s_len)


    def test_schnabel3_1(self):
        s_len = 5 
        k = 1

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org, orig=True)
        self.assertGreaterEqual(test_sn.capture_sum(), (k + 1) * s_len)

    def test_schnabel3_2(self):
        s_len = random.randint(4, 20)
        k = 1 #random.randint(0, s_len - 1)

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org, orig=True)
        self.assertGreaterEqual(test_sn.capture_sum(), (k + 1) * s_len)

    def test_schnabel4(self):
        ## PROBLEM
        s_len = 5 
        k = 1

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org)
        self.assertEqual(test_sn.capture(), (k + 1) * s_len * 2, msg="Test capture sum")

    def test_schnabel5(self):
        s_len = 5 
        k = 1
        t = 3

        arr0 = np.random.rand(s_len, 2)
        arr1 = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr0, arr1,k=k)
        test_sn = sn.Schnabel(data_org)

        maximum = 2 * s_len
        marked = len(test_sn.mark(t))
        self.assertLessEqual(marked, maximum, msg="Test cardinality of mark function, t != 1 and t != T")

    
    def test_schnabel6(self):

        k = 1

        arr0 = np.array([[1,1], [2,2], [3,3]])
        arr1 = np.array([[-3,-3], [-4,-4], [-5,-5]])
        data_org = do.DataOrg(arr0, arr1,k=k)
        test_sn = sn.Schnabel(data_org)

        self.assertEqual(len(test_sn.mark(1)), len(arr0), msg="test for two distant sets if the knn is correct and no sample from the other set is in the marked sample")


    def test_schnabel7(self):

        k = 1

        arr0 = np.array([[1,1], [2,2], [3,3]])
        arr1 = np.array([[-3,-3], [-4,-4], [-5,-5]])
        data_org = do.DataOrg(arr0, arr1,k=k)
        test_sn = sn.Schnabel(data_org)

        # previous step
        test_sn.mark(1)
        self.assertEqual(len(test_sn.mark(2)), len(arr0) + 2, \
            msg="test for two distant sets if the union in the marking process is correct and if with t = 2 there is only one sample and its first neighbor in the set") 

        
    def test_schnabel8(self):
        s_len = 6 
        k = 1

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org)
        self.assertEqual(test_sn.recapture(), len(arr) * (k + 1) * 2, msg="Test recapture function, theorem based")
    
    def test_schnabel8_1(self):
        s_len = 6 
        k = 1

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr, k=k)
        test_sn = sn.Schnabel(data_org, orig=True)
        self.assertGreaterEqual(test_sn.recapture(), (len(arr) + len(arr) ** 2) * (k + 1), msg="Test recapture function, original description")

    
    def test_schnabel9(self):

        ## PROBLEM
        s_len = 6 
        k = 1

        arr = np.random.rand(s_len, 2)
        data_org = do.DataOrg(arr, arr,k=k)
        test_sn = sn.Schnabel(data_org)
        self.assertEqual(test_sn.estimate(), 2 * s_len)


    def test_schnabel10(self):
        s_len0 = 10
        s_len1 = 10
        k = 1

        arr0 = np.random.rand(int(s_len0), 2)
        arr1 = np.random.rand(int(s_len1), 2)
        data_org = do.DataOrg(arr0, arr1,k=k)
        test_sn = sn.Schnabel(data_org)

        self.assertGreaterEqual(test_sn.estimate(), s_len0 + s_len1)


if __name__ == '__main__':
    unittest.main()