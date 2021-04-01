import unittest
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import Capture
from markevaluate import Utilities

class TestCapture(unittest.TestCase):

    def test_capture_total_0(self):
        test_capture = Capture.Capture(set([]), set([]), k = 2)
        self.assertEqual(test_capture.capture_total(), 0, msg="capture_total with empty input")

    # def test_capture_total_1(self):
    #     # TODO: change to random numbers
    #     s_len = 5
    #     k = 2
    #     arr = np.random.rand(s_len, 2)
    #     set0 = {tuple(elem) for elem in arr}
    #     set1 = {tuple(elem) for elem in arr}
    #     test_capture = Capture.Capture(set0, set1, k = k)
    #     result0 = 4 * s_len * (k + 1)
    #     result1 = s_len * (k + 1) + s_len ** 2 * (k + 1)
    #     print(result0)
    #     print(result1)
    #     print(test_capture.capture_total())
    #     #self.assertEqual(test_capture.capture_total(), 2 * s0_len * (k + 1) + 2 * s1_len * (k + 1))
    
