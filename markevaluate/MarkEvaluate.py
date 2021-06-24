"""Entry module for Mark-Evaluate."""

import numpy as np
import os
import torch
import tensorflow as tf
import time

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from .Petersen import Petersen as pt
from .Schnabel import Schnabel as sn
from .Capture import Capture as cp
from . import DataOrg as do


class MarkEvaluate:
    """Main class for Mark-Evaluate."""

    def __init__(
            self,
            metric: list = ["Schnabel", "Petersen", "CAPTURE"],
            model_str: str = 'bert-base-nli-mean-tokens',
            quality: str = "diversity",
            orig: bool = False,
            k: int = 1,
            sent_transf: bool = True,
            verbose: bool = False,
            sntnc_lvl: bool = False
    ) -> None:
        """Initialize function for ME class.

        Defining whether to use SBERT or BERT.
        """
        self.metric: list = metric
        self.model: any = None
        self.tokenizer: any = None
        if sent_transf:
            self.model: SentenceTransformer = SentenceTransformer(model_str)
        else:
            path: str = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'bert_base_mnli')
            self.tokenizer =\
                BertTokenizer.from_pretrained(path)
            self.model = \
                BertModel.from_pretrained(path,
                                          output_hidden_states=True)
            # To get the last 5 layers of BERT
            self.model.eval()

        self.k: int = k
        self.orig: bool = orig
        self.quality: str = quality
        self.sent_transf: bool = sent_transf
        self.verbose: bool = verbose
        self.sntnc_lvl: bool = sntnc_lvl

        self.data_org: do = None
        self.result: dict = None

    def get_embds(self, sentences: list) -> np.ndarray:
        """Get Embeddings from input sentences."""
        if self.sent_transf:
            return self.model.encode(sentences)
        else:
            embd_set: list = list()

            for i, sentence in enumerate(sentences):

                tokenized_sent = self.tokenizer.tokenize(
                    sentence)  # + " [SEP]")
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                    tokenized_sent)

                segments_ids = [i + 1] * len(tokenized_sent)

                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])

                outputs = self.model(tokens_tensor, segments_tensors)
                # append last 5 layers
                for j in range(5):
                    with torch.no_grad():
                        last_five_hidden_states = outputs[2][-5:][j][0]
                        for elem in last_five_hidden_states:
                            embd_set.append(elem.numpy())
            return np.asarray(embd_set)

    def petersen(self, p: int) -> float:
        """Estimate Petersen pop estimator."""
        assert self.data_org is not None
        pt_estim: pt = pt(self.data_org, orig=self.orig)
        return 1 - self.accuracy_loss(
            pt_estim.estimate(),
            p)

    def schnabel(self, p: int) -> float:
        """Estimate Schnabel pop estimator."""
        assert self.data_org is not None
        sn_estim: sn = sn(self.data_org, orig=self.orig)
        schn_div = 1 - self.accuracy_loss(
            sn_estim.estimate(),
            p)
        self.data_org.switch_input()
        sn_estim: sn = sn(self.data_org, orig=self.orig)
        schn_qul = 1 - self.accuracy_loss(
            sn_estim.estimate(),
            p)
        return schn_div, schn_qul

    def capture(self, p: int) -> float:
        """Estimate CAPTURE pop estimator."""
        assert self.data_org is not None
        cp_estim: cp = cp(self.data_org, orig=self.orig)
        return 1 - self.accuracy_loss(
            cp_estim.estimate(),
            p)

    def __estimate(self, cand: list, ref: list) -> dict:
        cand: np.ndarray = self.get_embds(cand)
        ref: np.ndarray = self.get_embds(ref)

        self.data_org: do = do.DataOrg(
            cand,
            ref,
            k=self.k,
            verbose=self.verbose)

        p: int = len(cand) + len(ref)

        start_time = time.time()
        me_petersen: float = self.petersen(p)\
            if 'Petersen' in self.metric else None
        if self.verbose:
            print("--- %s took %s seconds ---"
                  % ("Petersen", str(time.time() - start_time)))

        # start_time = time.time()
        me_capture: float = self.capture(p)\
            if 'CAPTURE' in self.metric else None
        if self.verbose:
            print("--- %s took %s seconds ---"
                  % ("CAPTURE", str(time.time() - start_time)))

        # start_time = time.time()
        me_schnabel_div,  me_schnabel_qul =\
            self.schnabel(p)\
            if 'Schnabel' in self.metric else None
        if self.verbose:
            print("--- %s took %s seconds ---"
                  % ("Schnabel", str(time.time() - start_time)))

        self.result = {
            'Petersen': me_petersen,
            'Schnabel_qul': me_schnabel_qul,
            'Schnabel_div': me_schnabel_div,
            'CAPTURE': me_capture
        }
        return self.result

    def estimate(self, cand: list, ref: list) -> dict:
        """Estimate all."""
        assert isinstance(cand, list)
        assert isinstance(ref, list)

        if self.sntnc_lvl:
            assert len(cand) == len(ref)
            ret_dict = {
                'Petersen': [],
                'Schnabel_qul': [],
                'Schnabel_div': [],
                'CAPTURE': []
            }
            for c, r in zip(cand, ref):
                # assert len(c) <= len(r)
                estimated: dict = self.__estimate([c], [r])
                ret_dict['Petersen'].append(estimated['Petersen'])
                ret_dict['Schnabel_qul'].append(estimated['Schnabel_qul'])
                ret_dict['Schnabel_div'].append(estimated['Schnabel_div'])
                ret_dict['CAPTURE'].append(estimated['CAPTURE'])
            return ret_dict
        else:
            return self.__estimate(cand=cand, ref=ref)

    @staticmethod
    def accuracy_loss(p_hat: int, p: int) -> float:
        """Accuracy loss function.

        Function that returns the accuracy of the
        population estimate by calculating (p_hat - p) / p,
        which is furthermore bounded to the top with 1.
        Complexity is O(1).

        Parameters
        ----------
        p_hat: int
            population estimate
        p : int
            real population, known before

        Returns
        -------
        float
            accuracy of estimation
        """
        if p == 0:
            return -1
        a: float = abs((p_hat - p) / p)
        return 1 if a > 1 else a
