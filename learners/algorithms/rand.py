from __future__ import absolute_import, division
import random
import collections

from .. import learner


defcfg = learner.Learner.defcfg._copy(deep=True)
defcfg.classname = 'learners.RandomLearner'

class RandomLearner(learner.Learner):
    """Random learner. Does not learn much."""

    defcfg = defcfg

    def _predict(self, m_signal):
        """ Return a random sensory signal
            Require all sensory channels to have bounds"""
        return {c.name: random.uniform(*c.bounds) for c in self.s_channels}

    def _infer(self, s_signal):
        """ Return a random motor signal"""
        return {c.name: random.uniform(*c.bounds) for c in self.uni_m_channels}

    def _update(self, m_signal, s_signal, uuid=None):
        pass
