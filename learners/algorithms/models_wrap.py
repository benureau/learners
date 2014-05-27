"""A Wrapper class for `models` learners."""
from __future__ import absolute_import, division, print_function
import collections

import forest
import models.learner

from .. import learner


defcfg = learner.Learner.defcfg._copy(deep=True)
defcfg._describe('models.fwd', instanceof=str,
                 docstring='The name of the forward model to use')
defcfg._describe('models.inv', instanceof=str,
                 docstring='The name of the invserse model to use')
defcfg._describe('models.kwargs', instanceof=dict,
                 docstring='optional keyword arguments')
defcfg.models.kwargs = {}


class ModelLearner(learner.Learner):
    """ Interface the old models module for explorer communications"""
    defcfg = defcfg

    def __init__(self, cfg):
        super(ModelLearner, self).__init__(cfg)
        m_bounds = [(0.0, 1.0) if c.bounds[0] != c.bounds[1] else (0.0, 0.0) for c in self.m_channels]
        self.learner = models.learner.Learner(range(-len(self.m_channels), 0), range(len(self.s_channels)),
                                              m_bounds, fwd=self.cfg.models.fwd, inv=self.cfg.models.inv,
                                              **self.cfg.models.kwargs)

    def _predict(self, m_signal):
        """Predict the effect of an order"""
        m_vector = tools.to_vector(m_signal, self.m_channels)
        s_vector = self.learner.predict_effect(m_vector)
        return tools.to_signal(s_vector, self.s_channels)

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        s_vector = tools.to_vector(s_signal, self.s_channels)
        m_vector = self.learner.infer_order(s_vector)
        return tools.to_signal(m_vector, self.m_channels)

    def _update(self, m_signal, s_signal):
        m_vector = tools.to_vector(m_signal, self.m_channels)
        s_vector = tools.to_vector(s_signal, self.s_channels)
        self.learner.add_xy(m_vector, s_vector)
