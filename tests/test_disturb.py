from __future__ import absolute_import, division, print_function
import unittest
import random

import scicfg

import dotdot
from learners import Channel
from learners import DisturbLearner
from learners import DisturbTwoStepLearner


random.seed(0)

class TestDisturb(unittest.TestCase):

    def _config(self):
        ch_x = Channel('x', [0, 10])
        ch_y = Channel('y', [0, 10])
        ch_a = Channel('a', [0, 100])

        cfg = {'m_channels': [ch_x, ch_y],
               's_channels': [ch_a],
               'm_uniformize': True,
               'm_disturb': 0.01}

        return cfg

    def _learner_check(self, learner):
        learner.update({'x': 5, 'y': 4}, {'a': 9})
        p = learner.predict({'x': 3, 'y': 4})
        self.assertEqual(p, None)

        e = learner.infer({'a': 3})
        self.assertTrue(4.9 <= e['x'] <= 5.1 and
                        3.9 <= e['y'] <= 4.1)


    def test_disturb(self):
        cfg = self._config()
        learner = DisturbLearner(cfg)
        self._learner_check(learner)

    def test_disturb2(self):
        cfg = self._config()
        learner = DisturbTwoStepLearner(cfg)
        self._learner_check(learner)


if __name__ == '__main__':
    unittest.main()
