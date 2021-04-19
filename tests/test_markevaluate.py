import unittest
import math
import numpy as np

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import MarkEvaluate

class TestMarkEvaluate(unittest.TestCase):

    sentences0 = ['This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images.',\
        'This framework generates embeddings for each input sentence',\
        'sentences0 are passed as a list of string.',\
        'The quick brown fox jumps over the lazy dog.',\
        'This naming convention informs the test runner about which methods represent tests',\
        'The remainder of the documentation explores the full feature set from first principles.']

    sentences1 = ["The move is expected to delay the country's vaccination programme by several weeks.",\
        "Drug watchdog the European Medicines Agency last week announced a possible link with clots but said the risk of dying of Covid-19 was much greater.",\
        "Several European countries had previously briefly suspended the jab.",
        "Most have now resumed vaccinations with AstraZeneca, but often with limits to older age groups.",\
        "South Africa has also paused its use, despite the Johnson & Johnson being its preferred vaccine because of its effectiveness against the South African variant",\
        "Both vaccines work by a similar method, known as adenoviral vectors.",\
        "Danish officials said that all 2.4 million doses of the AstraZeneca vaccine would be withdrawn until further notice."]
    # https://www.bbc.com/news/world-europe-56744474

    def test_markevaluate1(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences0, ref=self.sentences0)
        self.assertEqual(me.estimate()['Peterson'], 1, msg="Test Peterson Estimator using Theorem A.1.")
    
    def test_markevaluate2(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences0, ref=self.sentences0)
        self.assertEqual(me.estimate()['Schnabel'], 1, msg="Test Schnabel Estimator using Theorem A.2.")

    def test_markevaluate3(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences0, ref=self.sentences0, quality="")
        self.assertEqual(me.estimate()['Schnabel'], 1, msg="Test Schnabel Estimator using Theorem A.2.")

    def test_markevaluate4(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences0, ref=self.sentences0)
        self.assertEqual(me.estimate()['CAPTURE'], 1, msg="Test CAPTURE Estimator using Theorem A.3.")

    def test_markevaluate5(self):

        me = MarkEvaluate.MarkEvaluate(cand=self.sentences0, ref=self.sentences1)
        result = me.estimate()
        self.assertTrue(0 <= result['Schnabel'] and result['Schnabel'] <= 1, msg="Test different input with different topics and different lengths.")
        

if __name__ == '__main__':
    unittest.main()
