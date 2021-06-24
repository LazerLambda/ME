import unittest
import numpy as np
import random

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import Petersen as pt
from markevaluate import DataOrg as do

class TestPetersen(unittest.TestCase):

    def test_Petersen1(self):
        s_len = 7
        k = 2
        arr = np.random.rand(s_len, 768)
        data_org: do = do.DataOrg(arr, arr)
        test_pt = pt.Petersen(data_org)
        result = 2 * s_len
        self.assertEqual(test_pt.estimate(), result)
    
    def test_Petersen2(self):

        s0_len = random.randrange(4, 10) #10e2)
        s1_len = random.randrange(4, 10) #10e2)
        rand_dim = random.randrange(1, 20)
        k = 2

        arr0 = np.random.rand(s0_len, 768)
        arr1 = np.random.rand(s1_len, 768)

        data_org: do = do.DataOrg(arr0, arr1)
        test_pt = pt.Petersen(data_org)
        self.assertGreaterEqual(int(test_pt.estimate()), s0_len + s1_len)

    def test_Petersen3(self):

        # PERFORMANCE
        s0_len = random.randrange(4, int(10e2))
        s1_len = random.randrange(4, int(10e2))
        rand_dim = random.randrange(1, 20)
        k = 2

        arr0 = np.random.rand(s0_len, 768)
        arr1 = np.random.rand(s1_len, 768)

        data_org: do = do.DataOrg(arr0, arr1)
        test_pt = pt.Petersen(data_org)
        self.assertGreaterEqual(int(test_pt.estimate()), s0_len + s1_len)


if __name__ == '__main__':
    unittest.main()