from __future__ import absolute_import, division, print_function
import unittest
import random
import copy

import scicfg

import dotdot
import learners
from learners import Channel
from learners import MutateNNLearner


random.seed(0)

class TestDisturb(unittest.TestCase):

    def _config(self):
        ch_x = Channel('x', [0, 10])
        ch_y = Channel('y', [0, 10])
        ch_a = Channel('a', [0, 100])

        cfg = {'m_channels'      : [ch_x, ch_y],
               's_channels'      : [ch_a],
               'm_uniformize'    : True,
               'operator.name'    : 'uniform',
               'operator.d'       : 0.01,
               'operator.p_mutate': 1.0}

        return cfg

    def _learner_check(self, learner):
        learner.update({'x': 5, 'y': 4}, {'a': 9})
        p = learner.predict({'x': 3, 'y': 4})
        self.assertEqual(p, None)

        e = learner.infer({'a': 3})
        self.assertTrue(4.8 <= e['x'] <= 5.2 and
                        3.8 <= e['y'] <= 4.2)

    def test_mutate(self):
        for name in ['uniform', 'uniformclamp', 'uniformsync', 'gauss']:
            cfg = self._config()
            cfg['operator.name'] = name
            learner = MutateNNLearner(cfg)
            self._learner_check(learner)

    def test_mutate2(self):
        cfg = self._config()
        cfg['operator.p_mutate'] = 0.0
        learner = MutateNNLearner(cfg)
        learner.update({'x': random.uniform(0, 10),
                        'y': random.uniform(0, 10)},
                       {'a': random.uniform(0, 100)})
        e = learner.infer({'a': random.uniform(0, 100)})


if __name__ == '__main__':
    unittest.main()
