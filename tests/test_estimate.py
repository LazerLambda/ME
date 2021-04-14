import unittest

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import Estimate

class TestEstimate(unittest.TestCase):

    def test_estimate1(self):
        with self.assertRaises(Exception, msg="Test behavior with empty sets"):
            Estimate.Estimate(set(), set(), k = 1)

    
    def test_estimate2(self):
        with self.assertRaises(Exception, msg="Test cardinality of input sets with respect to k."):
            Estimate.Estimate(set([1,2]), set([3,4]), k = 3)



if __name__ == '__main__':
    unittest.main()