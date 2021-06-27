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

        assert int(do_test.bin_vec_cand.sum()) == len(do_test.bin_vec_cand)
        assert int(do_test.bin_vec_ref.sum()) == len(do_test.bin_vec_ref)


    def test_3(self):

        s_len = int(10e4)
        k = 2
        dim = 768
        start_time = time.time()

        arr0 = np.random.rand(int(100), dim)
        arr1 = np.random.rand(int(100), dim)

        do_test = do.DataOrg(arr0, arr1)

        assert int(do_test.bin_vec_cand.sum()) <= len(do_test.bin_vec_cand)
        assert int(do_test.bin_vec_ref.sum()) <= len(do_test.bin_vec_ref)

    def test_4(self):
        s_len = int(10e4)
        k = 1
        dim = 768

        arr = np.random.rand(5, dim)
        do_test = do.DataOrg(arr, arr, orig=True)

        bin_mat_cand_sum = do_test.bin_mat_cand.sum(axis=0)
        bin_mat_ref_sum = do_test.bin_mat_ref.sum(axis=0)

        assert (bin_mat_cand_sum > k).all()
        assert (bin_mat_ref_sum > k).all()

    def test_5(self):

        arr = np.array([[1,1], [2, 2], [3,3], [4,4]])
        do_test = do.DataOrg(arr, arr, orig=True)

        bin_mat_cand_sum = do_test.bin_mat_cand.sum(axis=0)
        bin_mat_ref_sum = do_test.bin_mat_ref.sum(axis=0)

        assert (bin_mat_cand_sum >= 3).all()
        assert (bin_mat_ref_sum >= 3).all()

    def test_6(self):
        k = 1

        arr = np.array([[1,1], [2, 2], [5,5], [6,6]])
        do_test = do.DataOrg(arr, arr, orig=True)

        bin_mat_cand_sum = do_test.bin_mat_cand.sum(axis=0)
        bin_mat_ref_sum = do_test.bin_mat_ref.sum(axis=0)

        assert (bin_mat_cand_sum == k + 1).all()
        assert (bin_mat_ref_sum == k + 1).all()

    def test_7(self):
        s_len = 30
        k = 1
        dim = 768

        arr = np.random.rand(s_len, dim)
        do_test = do.DataOrg(arr, arr, orig=True)

        bin_mat_cand_sum = do_test.bin_mat_cand.sum(axis=0).sum()
        bin_mat_ref_sum = do_test.bin_mat_ref.sum(axis=0).sum()

        assert bin_mat_cand_sum >= s_len * (k + 1)
        assert bin_mat_ref_sum >= s_len * (k + 1)

    def test_7(self):
        k = 1

        arr = np.array([[1,1], [2, 2], [5,5], [6,6]])
        do_test = do.DataOrg(arr, arr, orig=True)

        bin_mat_cand_sum = do_test.bin_mat_cand.sum(axis=0).sum()
        bin_mat_ref_sum = do_test.bin_mat_ref.sum(axis=0).sum()
        
        assert bin_mat_cand_sum == len(arr) * (k + 1)
        assert bin_mat_ref_sum == len(arr) * (k + 1)

if __name__ == '__main__':
    unittest.main()
