import unittest
import math
import numpy as np
import time

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from markevaluate import MarkEvaluate

class TestMarkEvaluate(unittest.TestCase):

    BERT_HIDDEN_SIZE = 768

    sentences0 = ['This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images.',
                  'This framework generates embeddings for each input sentence',
                  'sentences0 are passed as a list of string.',
                  'The quick brown fox jumps over the lazy dog.',
                  'This naming convention informs the test runner about which methods represent tests',
                  'The remainder of the documentation explores the full feature set from first principles.',
                  'With this framework, one is able to test these metrics in an exhausitve way.']

    sentences1 = ["The move is expected to delay the country's vaccination programme by several weeks.",
                  "Drug watchdog the European Medicines Agency last week announced a possible link with clots but said the risk of dying of Covid-19 was much greater.",
                  "Several European countries had previously briefly suspended the jab.",
                  "Most have now resumed vaccinations with AstraZeneca, but often with limits to older age groups.",
                  "South Africa has also paused its use, despite the Johnson & Johnson being its preferred vaccine because of its effectiveness against the South African variant",
                  "Both vaccines work by a similar method, known as adenoviral vectors.",
                  "Danish officials said that all 2.4 million doses of the AstraZeneca vaccine would be withdrawn until further notice."] 

    sentences01 = ['This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images.',
                   'This framework generates embeddings for each input sentence']

    sentences11o = ["This action will delay the country's vaccination programme for some time",
                    "Drug watchdog the European Medicines Agency announced last week a possible link with clots but said the risk of dying of Covid-19 was much greater."]

    sentences11o = ["This.",
                    "Drug."]

    sentences11 = ["The move is expected to delay the country's vaccination programme by several weeks.",
                   "Drug watchdog the European Medicines Agency last week announced a possible link with clots but said the risk of dying of Covid-19 was much greater."]
    # https://www.bbc.com/news/world-europe-56744474

    def test_markevaluate1(self):

        me = MarkEvaluate.MarkEvaluate()
        self.assertEqual(me.estimate(cand=self.sentences0, ref=self.sentences0)['Petersen'], 1, msg="Test Petersen Estimator using Theorem A.1.")

    def test_markevaluate2(self):

        me = MarkEvaluate.MarkEvaluate()
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)
        self.assertEqual(res['Schnabel_qul'], 1, msg="Test Schnabel (quality) Estimator using Theorem A.2.")
        self.assertEqual(res['Schnabel_div'], 1, msg="Test Schnabel (diversity) Estimator using Theorem A.2.")

    def test_markevaluate2_2(self):

        me = MarkEvaluate.MarkEvaluate(orig=True)
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_qul']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (quality) Estimator (original).")
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_div']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (diversity) Estimator (original).")

    def test_markevaluate3(self):

        me = MarkEvaluate.MarkEvaluate()
        self.assertEqual(me.estimate(cand=self.sentences0, ref=self.sentences0,)['Schnabel_qul'], 1, msg="Test Schnabel Estimator using Theorem A.2.")

    def test_markevaluate3_3(self):

        me = MarkEvaluate.MarkEvaluate(orig=True)
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_qul']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (quality) Estimator (original).")
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['Schnabel_div']
        self.assertTrue(0 <= res and res <= 1, msg="Test Schnabel (diversity) Estimator (original).")

    def test_markevaluate4(self):

        me = MarkEvaluate.MarkEvaluate()
        self.assertEqual(me.estimate(cand=self.sentences0, ref=self.sentences0)['CAPTURE'], 1, msg="Test CAPTURE Estimator using Theorem A.3.")

    def test_markevaluate4_4(self):

        me = MarkEvaluate.MarkEvaluate(orig=True)
        res = me.estimate(cand=self.sentences0, ref=self.sentences0)['CAPTURE']
        self.assertTrue(0 <= res and res <= 1, msg="Test CAPTURE Estimator (original).")

    def test_markevaluate5(self):

        me = MarkEvaluate.MarkEvaluate()
        result = me.estimate(cand=self.sentences0, ref=self.sentences1)
        self.assertTrue(0 <= result['Schnabel_qul'] and result['Schnabel_qul'] <= 1, msg="Test different input with different topics and different lengths.")
        self.assertTrue(0 <= result['Schnabel_div'] and result['Schnabel_div'] <= 1, msg="Test different input with different topics and different lengths.")

    def test_markevaluate6(self):

        me = MarkEvaluate.MarkEvaluate(sent_transf=False)

        ex_sample = [
            "Hello World.",
            "This is a test for the ME-Metrik."
        ]

        len_token = 0

        for ex in ex_sample:
            len_token += len(me.tokenizer.tokenize(ex))

        len_token_set = len_token * 5
        embds = me.get_embds(sentences=ex_sample)
        x, y = embds.shape
        assert x == len_token_set
        assert y == self.BERT_HIDDEN_SIZE

    def test_markevaluate7(self):

        me = MarkEvaluate.MarkEvaluate(sent_transf=False)

        ex_sample = self.sentences0

        len_token = 0

        for ex in ex_sample:
            len_token += len(me.tokenizer.tokenize(ex))

        len_token_set = len_token * 5
        embds = me.get_embds(sentences=ex_sample)
        x, y = embds.shape
        assert x == len_token_set
        assert y == self.BERT_HIDDEN_SIZE

    # def test_markevaluate8(self):

    #     me = MarkEvaluate.MarkEvaluate(sent_transf=False)
    #     me.estimate(cand=self.sentences01, ref=self.sentences11)

    def test_markevaluate9(self):

        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True)
        re = me.estimate(cand=self.sentences01, ref=self.sentences11)

    def test_markevaluate10(self):
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True)
        re = me.estimate(cand=self.sentences11o, ref=self.sentences11)

    def test_markevaluate11(self):
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=self.sentences11o, ref=self.sentences11)

    def test_markevaluate12(self):
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=self.sentences0, ref=self.sentences1)

    def test_markevaluate13(self):
        # Test behaviour for single sentence using BERT word embeddings
        s0 = [
            "This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images."
        ]
        s1 = [
            "The move is expected to delay the country's vaccination programme by several weeks."
        ]
        me = MarkEvaluate.MarkEvaluate(sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=s0, ref=s1)

    def test_markevaluate14(self):
        # Test behaviour for BERT embeddings on sentence level
        me = MarkEvaluate.MarkEvaluate(
            sent_transf=False, sntnc_lvl=True, orig=True)
        re = me.estimate(cand=self.sentences0, ref=self.sentences1)

    def test_markevaluate15(self):
        # Test behaviour for the automatic reduction of k if the number 
        # of samples is too low
        s0 = [
            "This framework provides an easy method to compute dense vector representations for sentences, paragraphs, and images."
        ]
        s1 = [
            "The move is expected to delay the country's vaccination programme by several weeks."
        ]
        me = MarkEvaluate.MarkEvaluate(
            sent_transf=True)
        re = me.estimate(cand=s0, ref=s1)
        assert me.data_org.k == 0

    def test_markevaluate15(self):
        # Test behaviour with sentences longer than 512 words (BERT limit)
        me = MarkEvaluate.MarkEvaluate(
            sent_transf=False)
        sent = (
            'Teamwork: The former midfielder chats to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
            'to Christian Lattanzio during a training session '
        )
        me.get_embds([sent])
    

if __name__ == '__main__':
    unittest.main()
