import unittest
import math
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import Capture
from markevaluate import Utilities

class TestCapture(unittest.TestCase):

    # def test_capture_total_0(self):
    #     test_capture = Capture.Capture(set([]), set([]), k = 2)
    #     self.assertEqual(test_capture.capture_total(), 0, msg="capture_total with empty input")

    def test_capture1(self):
        # # TODO: change to random numbers
        s_len = 5
        k = 1
        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_capture = Capture.Capture(set0, set1, k = k)
        # OWN ASSUMPTIONS
        self.assertEqual(test_capture.capture_total(), 4 * s_len * (k + 1)) #2 * s_len * (k + 1) + 2 * s_len ** 2 * (k + 1))
        pass

    def test_capture4(self):
        ## PROBLEM
        s_len = 5
        k = 1
        arr = np.random.rand(s_len, 2)
        set0 = {tuple(elem) for elem in arr}
        set1 = {tuple(elem) for elem in arr}
        test_capture = Capture.Capture(set0, set1, k = k)
        # print(test_capture.maximize_likelihood(), 2 * s_len)