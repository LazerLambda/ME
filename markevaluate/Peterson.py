import numpy as np


from markevaluate.Utilities import Utilities as ut
from markevaluate.Estimate import Estimate


class Peterson(Estimate):



    def mark(self) -> int:
        return len(self.set0) + sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set0)), self.k) for elem in self.set1])



    def capture(self) -> int:
        return len(self.set1) + sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set1)), self.k) for elem in self.set0])



    def recapture(self) -> int:
        return sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set0)), self.k) for elem in self.set1]) + \
            sum([ut.is_in_hypersphere(elem, np.asarray(list(self.set1)), self.k) for elem in self.set0])



    def estimate(self) -> float:
        return self.capture() * self.mark() / self.recapture()