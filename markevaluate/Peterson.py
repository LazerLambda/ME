import numpy as np


from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate
from sklearn.neighbors import KDTree


class Peterson(Estimate):



    def mark(self, kdt0 : KDTree) -> int:
        return len(self.set0) + sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set0)), kdt0, self.k) for elem in self.set1])



    def capture(self, kdt1 : KDTree) -> int:
        return len(self.set1) + sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set1)), kdt1, self.k) for elem in self.set0])



    def recapture(self, kdt0 : KDTree, kdt1 : KDTree) -> int:

        return sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set0)), kdt0, self.k) for elem in self.set1]) + \
            sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set1)), kdt1, self.k) for elem in self.set0])



    def estimate(self) -> float:
        kdt0 : KDTree = KDTree(np.asarray(list(self.set0)), metric='euclidean')
        kdt1 : KDTree = KDTree(np.asarray(list(self.set1)), metric='euclidean')
        return self.capture(kdt1) * self.mark(kdt0) / self.recapture(kdt0, kdt1)