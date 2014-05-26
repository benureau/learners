from __future__ import absolute_import, division
import random
import collections

from .. import learner

class RandomLearner(learner.Learner):
    """Random learner. Does not learn much."""

    def _predict(self, m_signal):
        """ Return a random sensory signal
            Require all sensory channels to have bounds"""
        return collections.OrderedDict((c.name, random.uniform(*c.bounds)) for c in self.s_channels)

    def _infer(self, s_signal):
        """ Return a random motor signal"""
        return collections.OrderedDict((c.name, random.uniform(*c.bounds)) for c in self.m_channels)

    def _update(self, observation):
        pass
