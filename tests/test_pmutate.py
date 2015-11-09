from __future__ import absolute_import, division, print_function
import unittest
import random

import scicfg

import dotdot
import learners
from learners import Channel
from learners import PredictMutateNNLearner


random.seed(0)

class TestDisturb(unittest.TestCase):

    def _config(self):
        ch_x = Channel('x', [0, 10])
        ch_y = Channel('y', [0, 10])
        ch_a = Channel('a', [0, 100])

        # fwd_cfg = learners.ModelLearner.defcfg._deepcopy()
        # fwd_cfg.models.fwd = 'ES-LWLR'
        # fwd_cfg.models.inv = 'L-BFGS-B'

        cfg = {'m_channels'       : [ch_x, ch_y],
               's_channels'       : [ch_a],
               'm_uniformize'     : True,
               'operator.d'       : 0.01,
               'operator.p_mutate': 1.0,
               'operator.name'    : 'uniform',
               'attempts'         : 5}

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
        learner = PredictMutateNNLearner(cfg)
        self._learner_check(learner)


if __name__ == '__main__':
    unittest.main()
