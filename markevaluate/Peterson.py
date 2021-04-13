import numpy as np


from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate
from markevaluate.KNneighbors import KNneighbors as knn


class Peterson(Estimate):


    def mark(self, knn0 : knn) -> int:
        return len(self.set0) + sum([knn0.in_hypsphr(elem) for elem in self.set1])



    def capture(self, knn1 : knn) -> int:
        return len(self.set1) + sum([knn1.in_hypsphr(elem) for elem in self.set0])



    def recapture(self, knn0 : knn, knn1 : knn) -> int:

        return sum([knn0.in_hypsphr(elem) for elem in self.set1]) + \
            sum([knn1.in_hypsphr(elem) for elem in self.set0])



    def estimate(self) -> float:
        knn0 : knn = knn(np.asarray(list(self.set0)), k = self.k)
        knn1 : knn = knn(np.asarray(list(self.set1)), k = self.k)
        return self.capture(knn1 = knn1) * self.mark(knn0 = knn0) / self.recapture(knn0=knn0, knn1=knn1)