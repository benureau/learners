from __future__ import absolute_import, division, print_function
import unittest
import random

import dotdot
from learners import Channel
from learners import ImleLearner


random.seed(0)

class TestImle(unittest.TestCase):

    def _config(self):
        ch_x = Channel('x', [0.0, 1.0])
        ch_y = Channel('y', [0.0, 1.0])

        cfg = {'m_channels': [ch_x],
               's_channels': [ch_y],
               'm_uniformize': True}

        return cfg

    def test_linear1D(self):
        learner = ImleLearner(self._config())

        for i in range(100):
            learner.update({'x': i*0.01}, {'y': i*0.02})

        p = learner.predict({'x': 0.5})
        self.assertEqual(p, None)

        e = learner.infer({'y': 0.5})
        self.assertTrue(0.23 <= e['x'] <= 0.27)


if __name__ == '__main__':
    unittest.main()
