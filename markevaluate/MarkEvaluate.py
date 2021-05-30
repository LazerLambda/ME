import numpy as np

from sentence_transformers import SentenceTransformer
from .Petersen import Petersen as pt
from .Schnabel import Schnabel as sn
from .Capture import Capture as cp

class MarkEvaluate:

        # 'bert-large-nli-mean-tokens'

    def __init__(\
            self,\
            cand : list,\
            ref : list,\
            metric : list = ["Schnabel", "Petersen", "CAPTURE"],\
            sbert_model_str : str = 'bert-base-nli-mean-tokens',\
            quality : str = "diversity",\
            orig : bool = False,\
            k : int = 1
            ) -> None:

        self.metric : list = metric
        self.sbert_model : SentenceTransformer = SentenceTransformer(sbert_model_str) # CUDA support
        self.k : int = k
        self.orig : bool = orig
        self.quality : str = quality

        self.cand : np.ndarray = self.get_embds_sbert(cand)
        self.ref : np.ndarray = self.get_embds_sbert(ref)

        self.result : dict = None



    def get_embds_sbert(self, input : list) -> np.ndarray:
        return self.sbert_model.encode(input)



    def petersen(self) -> float:
        pt_estim : pt = pt({tuple(elem) for elem in self.cand}, {tuple(elem) for elem in self.ref}, k = self.k, orig=self.orig)
        return pt_estim.estimate()



    def schnabel(self, type : str = "quality") -> float:

        sn_estim : sn = sn({tuple(elem) for elem in self.cand}, {tuple(elem) for elem in self.ref}, k = self.k, orig=self.orig)
        schn_div = sn_estim.estimate()
        sn_estim : sn = sn({tuple(elem) for elem in self.ref}, {tuple(elem) for elem in self.cand}, k = self.k, orig=self.orig)
        schn_qul =  sn_estim.estimate()

        return schn_div, schn_qul


    def capture(self) -> float:
        cp_estim : cp = cp({tuple(elem) for elem in self.cand}, {tuple(elem) for elem in self.ref}, k = self.k, orig=self.orig)
        return cp_estim.estimate()


    def estimate(self) -> dict:

        p : int = len(self.cand) + len(self.ref)
        
        me_petersen = 1 - self.accuracy_loss(self.petersen(), p) if 'Petersen' in self.metric else None 
        me_capture  = 1 - self.accuracy_loss(self.capture(), p) if 'CAPTURE' in self.metric else None
        me_schnabel_div,  me_schnabel_qul = self.schnabel()
        me_schnabel_div = 1 - self.accuracy_loss(me_schnabel_div, p) if 'Schnabel' in self.metric else None
        me_schnabel_qul = 1 - self.accuracy_loss(me_schnabel_qul, p) if 'Schnabel' in self.metric else None

        self.result = {
            'Petersen' : me_petersen,
            'Schnabel_qul' : me_schnabel_qul,
            'Schnabel_div' : me_schnabel_div,
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
    def accuracy_loss(p_hat : int, p : int) -> float:
        """Accuracy loss function

        Function that returns the accuracy of the population estimate by calculating (p_hat - p) / p,
        which is furthermore bounded to the top with 1. Complexity is O(1).
        
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
        a : float = abs((p_hat - p) / p)
        return 1 if a > 1 else a