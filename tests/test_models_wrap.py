from __future__ import absolute_import, division, print_function
import unittest
import random
import collections

import numpy as np

import forest

import dotdot
import learners
from learners import tools

import testenvs


random.seed(0)

def dist(e1, e2, channels):
    return sum((e1[c.name] - e2[c.name])**2 for c in channels)


class TestModelWrap(unittest.TestCase):

    def test_simple(self):

        mbounds = ((23, 34), (-3, -2), (-40, 5))
        env = testenvs.RandomLinear(mbounds, 2)
        exp_cfg = learners.ModelLearner.defcfg._copy(deep=True)
        exp_cfg.m_channels = env.m_channels
        exp_cfg.s_channels = env.s_channels

        for fwd, inv in [('NN', 'NN'), ('ES-LWLR', 'L-BFGS-B')]:
            exp_cfg.models.fwd = fwd
            exp_cfg.models.inv = inv
            learner = learners.ModelLearner(exp_cfg)
            dataset = []

            for t in range(100):
                order  = tools.random_signal(env.m_channels)
                feedback = env.execute(order)
                learner.update_request(feedback)
                dataset.append(feedback)

            for t in range(1000):
                feedback = random.choice(dataset)
                predicted = learner.predict(feedback['m_signal'])
                self.assertTrue(dist(feedback['s_signal'], predicted, env.s_channels) < 1e-3)

            for t in range(1000):
                feedback = random.choice(dataset)
                infered = learner.infer(feedback['s_signal'])
                feedback_infered = env.execute(infered)
                self.assertTrue(dist(feedback['s_signal'], feedback_infered['s_signal'], env.s_channels) < 0.01)

    def test_uuid(self):
        mbounds = ((23, 34), (-3, -2), (-40, 5))
        env = testenvs.RandomLinear(mbounds, 2)
        exp_cfg = learners.ModelLearner.defcfg._copy(deep=True)
        exp_cfg.m_channels = env.m_channels
        exp_cfg.s_channels = env.s_channels

        exp_cfg.models.fwd = 'NN'
        exp_cfg.models.inv = 'NN'
        learner = learners.ModelLearner(exp_cfg)

        uuids = set()
        for t in range(100):
            order  = collections.OrderedDict((c.name, random.uniform(*c.bounds)) for c in env.m_channels)
            feedback = env.execute(order)
            self.assertTrue(feedback['uuid'] not in uuids)
            uuids.add(feedback['uuid'])
            learner.update_request(feedback)
            self.assertEqual(len(learner), t+1)


        order     = collections.OrderedDict((c.name, random.uniform(*c.bounds)) for c in env.m_channels)
        feedback1 = env.execute(order)
        learner.update_request(feedback1)
        self.assertEqual(len(learner), 101)

        order     = collections.OrderedDict((c.name, random.uniform(*c.bounds)) for c in env.m_channels)
        feedback2 = env.execute(order)
        feedback2['uuid'] = feedback1['uuid']
        learner.update_request(feedback2)
        self.assertEqual(len(learner), 101)

if __name__ == '__main__':
    unittest.main()
