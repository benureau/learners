from __future__ import absolute_import, division, print_function
import unittest
import random

import forest

import dotdot
from learners.nn_set import NNSet


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


if __name__ == '__main__':
    unittest.main()
