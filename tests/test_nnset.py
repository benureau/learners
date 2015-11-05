from __future__ import absolute_import, division, print_function
import unittest
import random

import scicfg

import dotdot
from learners.nn_set import NNSet, BatchNNSet, BruteForceNNSet


random.seed(0)

class TestReuse(unittest.TestCase):

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

        for i in range(10):
            implementations = [BruteForceNNSet(), BatchNNSet(), NNSet(poolsize=7)]


            innset = BatchNNSet()
            n = random.randint(1, 5)
            m = random.randint(1, 5)
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
                    # for i in range(len(indexes)-1):
                    #     for a, b in zip(indexes[i], indexes[i+1]):
                    #         self.assertEqual(a, b.all())

                    self.assertEqual(*indexes)

if __name__ == '__main__':
    unittest.main()
