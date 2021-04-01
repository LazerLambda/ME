import unittest
import numpy as np
import random

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import Peterson as pt

class TestPeterson(unittest.TestCase):

    # def test_peterson_0(self):
    #     test_pt = pt.Peterson(set([]), set([]), k = 2)
    #     self.assertRaises(ZeroDivisionError, test_pt.estimate)

    def test_peterson_1(self):
        s_len = 5
        k = 2
        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_pt = pt.Peterson(set0, set1, k)
        result = 2 * s_len
        self.assertEqual(test_pt.estimate(), result)
    
    def test_peterson_2(self):

        s0_len = random.randrange(4, 100) #10e2)
        s1_len = random.randrange(4, 100) #10e2)
        rand_dim = random.randrange(1, 20)
        k = 2

        arr0 = np.random.rand(s0_len, rand_dim)
        arr1 = np.random.rand(s1_len, rand_dim)

        set0 = {tuple(elem) for elem in arr0}
        set1 = {tuple(elem) for elem in arr1}

        test_pt = pt.Peterson(set0, set1, k)
        self.assertGreater(test_pt.estimate(), 1)