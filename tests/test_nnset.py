from __future__ import absolute_import, division, print_function
import unittest
import random

import dotdot
from learners.nn_set import NNSet, BatchNNSet, BruteForceNNSet


random.seed(0)

class TestNNSet(unittest.TestCase):

    def test_nn_x(self):
        nnset = NNSet()
        for x_i in range(10):
            nnset.add([x_i, x_i])

        for x_i in range(10):
            dists, idxs = nnset.nn_x([x_i, x_i+0.5], k=1)
            nn_x = list(nnset.xs[idxs[0]])
            self.assertEqual(nn_x, [x_i, x_i])

    def test_nn_xy(self):
        nnset = NNSet()
        for x_i in range(10):
            nnset.add([x_i, x_i], [x_i*2])

        for x_i in range(10):
            dists, idxs = nnset.nn_y([2*x_i+0.1], k=1)
            nn_y = list(nnset.ys[idxs[0]])
            self.assertEqual(nn_y, [2*x_i])

    def test_incrementalnn(self):

        for _ in range(10):
            implementations = [BruteForceNNSet(), BatchNNSet(), NNSet(poolsize=7)]

            n = random.randint(2, 5)
            m = random.randint(2, 5)

            for j in range(100):
                x, y = [random.random() for _ in range(n)], [random.random() for _ in range(m)]
                for imp in implementations:
                    imp.add(x, y)

                for _ in range(10):
                    k = min(random.randint(1, 10), j+1)
                    x, y = [random.random() for _ in range(n)], [random.random() for _ in range(m)]
                    indexes = []
                    for imp in implementations:
                        dists, idxes = imp.nn_x(x, k=k)
                        indexes.append(idxes)
                    self.assertEqual(*indexes)

if __name__ == '__main__':
    unittest.main()
