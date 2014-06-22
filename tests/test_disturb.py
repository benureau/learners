from __future__ import absolute_import, division, print_function
import unittest
import random

import forest

import dotdot
from learners.nn_set import NNSet


random.seed(0)

class TestDisturb(unittest.TestCase):

    def test_nn_x(self):
        from learners import Channel, DisturbLearner

        ch_x = Channel('x', [0, 10])
        ch_y = Channel('y', [0, 10])
        ch_a = Channel('a', [0, 100])

        cfg = {'m_channels': [ch_x, ch_y],
               's_channels': [ch_a],
               'm_uniformize': True,
               'm_disturb': 0.01}

        learner = DisturbLearner(cfg)

        learner.update({'x': 5, 'y': 4}, {'a': 9})
        p = learner.predict({'x': 3, 'y': 4})
        self.assertEqual(p, None)

        e = learner.infer({'a': 3})
        self.assertTrue(4.9 <= e['x'] <= 5.1 and
                        3.9 <= e['y'] <= 4.1)


if __name__ == '__main__':
    unittest.main()
