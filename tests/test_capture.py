import unittest
import math
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import Capture
from markevaluate import DataOrg as do

class TestCapture(unittest.TestCase):

    # def test_capture_total_0(self):
    #     test_capture = Capture.Capture(set([]), set([]), k = 2)
    #     self.assertEqual(test_capture.capture_total(), 0, msg="capture_total with empty input")

    def test_capture1(self):
        # # TODO: change to random numbers
        # ERROR 
        s_len = 5
        k = 1
        arr = np.random.rand(s_len, 2)
        data_org: do = do.DataOrg(arr, arr)
        test_capture = Capture.Capture(data_org)
        self.assertEqual(test_capture.capture_total(), 4 * s_len * (k + 1), msg="Test Capture_T function, theorem based")

    def test_capture1_1(self):
        # # TODO: change to random numbers
        # ERROR 
        s_len = 5
        k = 1
        arr = np.random.rand(s_len, 2)
        data_org: do = do.DataOrg(arr, arr)
        test_capture = Capture.Capture(data_org, orig=True)
        self.assertGreaterEqual(test_capture.capture_total(), s_len ** 2 * (k + 1) + s_len ** 2 * (k + 1), msg="Test Capture_T function, original") 

    def test_capture2(self):
        ## PROBLEM
        s_len = 5
        k = 1
        arr = np.random.rand(s_len, 2)
        data_org: do = do.DataOrg(arr, arr)
        test_capture = Capture.Capture(data_org)
        self.assertEqual(test_capture.maximize_likelihood(), 2 * s_len, msg="Test ML function for correct estimation")

if __name__ == '__main__':
    unittest.main()