import numpy as np

from sentence_transformers import SentenceTransformer
from markevaluate.Utilities import Utilities as ut
from Capture import Capture
from Schnabel import Schnabel

class MarkEvaluate:



    def __init__(\
            self,\
            cand : list,\
            ref : list,\
            metric : list = ["Schnabel", "Peterson", "CAPTURE"],\
            sbert_model_str : str = 'bert-base-nli-mean-tokens',\
            quality : str = "diversity",\
            k : int = 1
            ) -> None:
        self.cand : str = cand
        self.ref : str = ref
        self.metric : list = metric
        self.sbert_model : SentenceTransformer = SentenceTransformer(sbert_model_str) # CUDA support
        self.k : int = k
        self.result : dict = None



    def get_embeddings(self, input : str) -> np.ndarray:
        return self.sbert_model.encode(input)



    def peterson(self) -> float:
        reference : np.ndarray = self.get_embeddings(self.ref)
        candidate : np.ndarray = self.get_embeddings(self.cand)
        mc = lambda s_, s : len(s) + sum([ut.is_in_hypersphere(elem, s, k=self.k) for elem in s_])
        r = lambda s_, s : sum([ut.is_in_hypersphere(elem, s, k=self.k) for elem in s_]) + sum([ut.is_in_hypersphere(elem, s_, k=self.k) for elem in s])
        return mc(reference, candidate) * mc(candidate, reference) / r(reference, candidate)



    def schnabel(self, type : str = "quality") -> float:
        return 0



    def capture(self) -> float:
        return 0



    def estimate(self) -> dict:

        p : int = len(self.cand) + len(self.ref)
        
        me_peterson = 1 - ut.accuracy_loss(self.peterson(), p) if 'Peterson' in self.metric else None 
        me_schnabel = 1 - ut.accuracy_loss(self.schnabel(), p) if 'Schnabel' in self.metric else None
        me_capture  = 1 - ut.accuracy_loss(self.capture(), p) if 'CAPTURE' in self.metric else None

        self.result = {
            'Peterson' : me_peterson,
            'Schnabel' : me_schnabel,
            'CAPTURE' : me_capture
        }

        return self.result



    def summary(self) -> None:

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
        
        if self.result == None:
            self.estimate()

        # Output
        print("\n")
        print(f"{HEADER}MARK-EVALUATE RESULTS{ENDC}")
        print("\n\n")
        print(f"{OKCYAN}Interpretation{ENDC}: 0 poor quality <-> 1 good quality")
        print("\n")
        print(f"Peterson: {OKBLUE}{self.result['Peterson']}{ENDC}")
        print(f"Schnabel: {OKBLUE}{self.result['Schnabel']}{ENDC} *")
        print(f"CAPTURE:  {OKBLUE}{self.result['CAPTURE']}{ENDC}")
        print("\n\n")
        print("* Can be used to assess quality and diversity.")
        print("Use (ref, cand) for quality and (cand, ref) for diversity.")
        print(f"{BOLD}(Mordido, Meinel, 2020){ENDC}")
        print(f"{BOLD}https://arxiv.org/abs/2010.04606{ENDC}")
        print("\n")

        