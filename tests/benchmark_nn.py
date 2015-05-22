from __future__ import absolute_import, division, print_function
import random
import time

import dotdot
from learners.nn_set import BatchNNSet, NNSet, BruteForceNNSet

N       = 2000
DIM     = 100
K       = 1

DYN_FREQ    = 1
DYN_QUERIES = 1
END_QUERIES = 50

def testrun(nnset):
    random.seed(0)
    for i in range(N):
        x = [random.random() for _ in range(DIM)]
        nnset.add(x, [random.random()])

        if i%DYN_FREQ == 0:
            for i in range(DYN_QUERIES):
                x = [random.random() for _ in range(DIM)]
                nnset.nn_x(x, k=1)


    for i in range(END_QUERIES):
        x = [random.random() for _ in range(DIM)]
        nnset.nn_x(x, k=1)



if __name__ == '__main__':
    print('Running benchmark... this may take a few minutes.')
    for nnset in [NNSet(), BatchNNSet(), BruteForceNNSet()]:
        start = time.time()
        testrun(nnset)
        print('{}: {:4.2f}s'.format(nnset.__class__.__name__, time.time()-start))
