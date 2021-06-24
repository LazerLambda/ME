import unittest
import numpy as np
import random

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from markevaluate import DataOrg as do
from sklearn.neighbors import KDTree
from scipy import spatial


class TestDataOrg(unittest.TestCase):


    def test_1(self):

        s_len = int(10e4)
        k = 2
        dim = 768

        arr0 = np.random.rand(int(100), dim)
        arr1 = np.random.rand(int(100), dim)

        do_test = do.DataOrg(arr0, arr1)
        assert True

    def test_2(self):

        s_len = int(10e4)
        k = 2
        dim = 768

        arr = np.random.rand(int(50), dim)
        do_test = do.DataOrg(arr, arr)

        assert int(do_test.bin_mat_cand.sum()) == len(do_test.bin_mat_cand)
        assert int(do_test.bin_mat_ref.sum()) == len(do_test.bin_mat_ref)


    def test_3(self):

        s_len = int(10e4)
        k = 2
        dim = 768
        start_time = time.time()

        arr0 = np.random.rand(int(100), dim)
        arr1 = np.random.rand(int(100), dim)

        do_test = do.DataOrg(arr0, arr1)

        assert int(do_test.bin_mat_cand.sum()) <= len(do_test.bin_mat_cand)
        assert int(do_test.bin_mat_ref.sum()) <= len(do_test.bin_mat_ref)

if __name__ == '__main__':
    unittest.main()
