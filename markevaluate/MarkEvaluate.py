"""Entry module for Mark-Evaluate."""

import numpy as np
import tensorflow as tf

from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, TFBertModel
from .Petersen import Petersen as pt
from .Schnabel import Schnabel as sn
from .Capture import Capture as cp


class MarkEvaluate:
    """Main class for Mark-Evaluate."""

    def __init__(
            self,
            metric: list = ["Schnabel", "Petersen", "CAPTURE"],
            model_str: str = 'bert-base-nli-mean-tokens',
            quality: str = "diversity",
            orig: bool = False,
            k: int = 1,
            sent_transf: bool = False
            ) -> None:
        """Initialize function for ME class.

        Defining whether to use SBERT or BERT.
        """
        self.metric: list = metric
        if sent_transf:
            self.tokenizer =\
                BertTokenizer.from_pretrained('bert-base-cased')
            self.model: TFBertModels =\
                TFBertModel.from_pretrained('bert-base-cased')
        else:
            self.model: SentenceTransformer = SentenceTransformer(model_str)
        self.k: int = k
        self.orig: bool = orig
        self.quality: str = quality
        self.sent_transf: bool = sent_transf

        self.result: dict = None

    def get_embds(self, sentences: list) -> np.ndarray:
        """Get Embeddings from input sentences."""
        if self.sent_transf:
            return self.model.encode(sentences)
        else:
            inputs =\
                [(self.tokenizer(
                    sentence,
                    return_tensors="tf"))[2][-5:] for sentence in sentences]
            # TODO checks
            return [self.model(
                inpt,
                output_hidden_states=True) for inpt in inputs]

    def petersen(self, cand: list, ref: list) -> float:
        """Estimate Petersen pop estimator."""
        pt_estim: pt = pt(
            {tuple(elem) for elem in cand},
            {tuple(elem) for elem in ref},
            k=self.k,
            orig=self.orig)
        return pt_estim.estimate()

    def schnabel(self, cand: list, ref: list) -> float:
        """Estimate Schnabel pop estimator."""
        sn_estim: sn = sn(
            {tuple(elem) for elem in cand},
            {tuple(elem) for elem in ref},
            k=self.k,
            orig=self.orig)
        schn_div = sn_estim.estimate()
        sn_estim: sn = sn(
            {tuple(elem) for elem in ref},
            {tuple(elem) for elem in cand},
            k=self.k,
            orig=self.orig)
        schn_qul = sn_estim.estimate()
        return schn_div, schn_qul

    def capture(self, cand: list, ref: list) -> float:
        """Estimate CAPTURE pop estimator."""
        cp_estim: cp = cp(
            {tuple(elem) for elem in cand},
            {tuple(elem) for elem in ref},
            k=self.k,
            orig=self.orig)
        return cp_estim.estimate()

    def estimate(self, cand: list, ref: list) -> dict:
        """Estimate all."""
        cand: np.ndarray = self.get_embds(cand)
        ref: np.ndarray = self.get_embds(ref)

        p: int = len(cand) + len(ref)

        me_petersen = 1 - self.accuracy_loss(
            self.petersen(cand=cand, ref=ref), p)\
            if 'Petersen' in self.metric else None
        me_capture = 1 - self.accuracy_loss(
            self.capture(cand=cand, ref=ref), p)\
            if 'CAPTURE' in self.metric else None
        me_schnabel_div,  me_schnabel_qul =\
            self.schnabel(cand=cand, ref=ref)
        me_schnabel_div = 1 - self.accuracy_loss(me_schnabel_div, p)\
            if 'Schnabel' in self.metric else None
        me_schnabel_qul = 1 - self.accuracy_loss(me_schnabel_qul, p)\
            if 'Schnabel' in self.metric else None

        self.result = {
            'Petersen': me_petersen,
            'Schnabel_qul': me_schnabel_qul,
            'Schnabel_div': me_schnabel_div,
            'CAPTURE': me_capture
        }

        return self.result

    def summary(self) -> None:
        """Display results."""
        # Colors for output
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKCYAN = '\033[96m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

        if self.result is None:
            raise Exception(
                ("ERROR:\n\t'-> No results to summarize."
                    " Call ME.estimate(cand, ref) first."))

        # Output
        print("\n")
        print(f"{HEADER}MARK-EVALUATE RESULTS{ENDC}")
        print("\n\n")
        print(
            f"{OKCYAN}Interpretation{ENDC}: 0 poor quality <-> 1 good quality")
        print("\n")
        print(f"Petersen: {OKBLUE}{self.result['Petersen']}{ENDC}")
        print(f"Schnabel: {OKBLUE}{self.result['Schnabel']}{ENDC} *")
        print(f"CAPTURE:  {OKBLUE}{self.result['CAPTURE']}{ENDC}")
        print("\n\n")
        print("* Can be used to assess quality and diversity.")
        print("Use (ref, cand) for quality and (cand, ref) for diversity.")
        print(f"{BOLD}(Mordido, Meinel, 2020){ENDC}")
        print(f"{BOLD}https://arxiv.org/abs/2010.04606{ENDC}")
        print("\n")

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
