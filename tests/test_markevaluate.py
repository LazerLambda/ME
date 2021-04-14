import unittest
import math
import numpy as np

import os
import sys

import markevaluate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import MarkEvaluate

class TestMarkEvaluate(unittest.TestCase):

    sentences = ['This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images.',\
        'This framework generates embeddings for each input sentence',\
        'Sentences are passed as a list of string.',\
        'The quick brown fox jumps over the lazy dog.',\
        'This naming convention informs the test runner about which methods represent tests',\
        'The remainder of the documentation explores the full feature set from first principles.']

    def test_markevaluate1(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences, ref=self.sentences)
        self.assertEqual(me.estimate()['Peterson'], 1, msg="Test Peterson Estimator using Theorem A.1.")
    
    def test_markevaluate2(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences, ref=self.sentences)
        self.assertEqual(me.estimate()['Schnabel'], 1, msg="Test Schnabel Estimator using Theorem A.2.")

    def test_markevaluate3(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences, ref=self.sentences, quality="")
        self.assertEqual(me.estimate()['Schnabel'], 1, msg="Test Schnabel Estimator using Theorem A.2.")

    def test_markevaluate4(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences, ref=self.sentences, quality="")
        self.assertEqual(me.estimate()['CAPTURE'], 1, msg="Test CAPTURE Estimator using Theorem A.3.")
        

if __name__ == '__main__':
    unittest.main()
