

class Utilities:


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

