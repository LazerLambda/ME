import numpy as np

from sentence_transformers import SentenceTransformer
from markevaluate.Utilities import Utilities
from Capture import Capture
from Schnabel import Schnabel

class MarkEvaluate:

    def __init__(\
            self,\
            cand : str,\
            ref : str,\
            metric : list = ["Schnabel", "Peterson", "CAPTURE"],\
            sbert_model_str : str = 'bert-base-nli-mean-tokens',\
            k : int = 2
            ) -> None:
        self.cand : str = cand
        self.ref : str = ref
        self.metric : list = metric
        self.sbert_model : SentenceTransformer = SentenceTransformer(sbert_model_str) # CUDA support
        self.k : int = k


    def get_embeddings(self, input : str) -> np.ndarray:
        return self.sbert_model.encode(input)

    def peterson(self) -> float:
        reference : np.ndarray = self.get_embeddings(self.ref)
        candidate : np.ndarray = self.get_embeddings(self.cand)
        mc = lambda s_, s : len(s) + sum([Utilities.is_in_hypersphere(elem, s, k=self.k) for elem in s_])
        r = lambda s_, s : sum([Utilities.is_in_hypersphere(elem, s, k=self.k) for elem in s_]) + sum([Utilities.is_in_hypersphere(elem, s_, k=self.k) for elem in s])
        return mc(reference, candidate) * mc(candidate, reference) / r(reference, candidate)

    def schnabel(self) -> float:
        return 0

    def capture(self) -> float:
        return 0

    def estimate(self) -> float:
        return 0

    def summary(self) -> None:
        pass