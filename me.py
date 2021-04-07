import numpy as np
import random
import faiss
import line_profiler
from sklearn.neighbors import KDTree

# @profile
# def is_in_hypersphere(elem : tuple, sample: np.ndarray, k : int) -> int:
#     kdt : KDTree = KDTree(sample, metric='euclidean')
#     for s in sample:
#         k_nn_dist, _ = kdt.query([s], k= k + 1)

#         mx = np.amax(k_nn_dist)
#         dist = np.linalg.norm(elem - s)

#         if  dist <= mx:
#             return 1
#     return 0

# @profile
# def is_in_hypersphere(elem : tuple, sample: np.ndarray, kdt : KDTree, k : int) -> int:

#     for s in sample:
#         k_nn_dist, _ = kdt.query([s], k= k + 1)

#         mx = np.amax(k_nn_dist)
#         dist = np.linalg.norm(elem - s)

#         if  dist <= mx:
#             return 1
#     return 0


@profile
def is_in_hypersphere(elem : tuple, sample: np.ndarray, index : faiss.swigfaiss.IndexFlatL2, k : int) -> int:

    for s in sample:
        k_nn_dist, indices = index.search(np.array([elem]).astype('float32'), k)
        k_nn_dist, indices = k_nn_dist[0], indices[0]
        mx = np.amax(k_nn_dist)
        dist = np.linalg.norm(elem - s)

        if  dist <= mx:
            return 1
    return 0



@profile
def main():

    s0_len = random.randrange(4, 10e3)
    s1_len = random.randrange(4, 10e3)
    rand_dim = random.randrange(1, 20)
    k = 2

    arr0 = np.random.rand(s0_len, rand_dim)
    arr1 = np.random.rand(s1_len, rand_dim)

    set0 = {tuple(elem) for elem in arr0}
    set1 = {tuple(elem) for elem in arr1}

    kdt : KDTree = KDTree(np.asarray(list(set0)), metric='euclidean')
    index : faiss.swigfaiss.IndexFlatL2 = faiss.IndexFlatL2(rand_dim)
    index.add(arr0.astype('float32'))
    #len(set0) + sum([is_in_hypersphere(elem, np.asarray(list(set0)), kdt, k) for elem in set1])
    len(set0) + sum([is_in_hypersphere(elem, np.asarray(list(set0)), index, k) for elem in set1])

if __name__ == '__main__':
    main()