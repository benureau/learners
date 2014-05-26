from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

from .. import tools
from . import nn


defcfg = nn.NNLearner.defcfg._copy(deep=True)
defcfg._describe('m_disturb', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Maximum distance of disturbance along each dimension. If ')

class DisturbLearner(nn.NNLearner):
    """"""
    defcfg = defcfg

    def __init__(self, cfg, nnset=None):
        super(DisturbLearner, self).__init__(cfg)

        self.m_disturb = self.cfg.m_disturb
        if isinstance(self.m_disturb, numbers.Real):
            self.m_disturb = [self.m_disturb for c in self.m_channels]

    def _predict(self, data):
        """Predict the effect of an order"""
        return None

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        s_v = tools.to_vector(s_signal, self.s_channels)
        dists, s_idx = self.nnset.nn_y(s_v, k = 1)
        m_nn = self.nnset.xs[s_idx[0]]

        m_disturbed = [v_i + random.uniform(-d_i, d_i) for v_i, d_i in zip(m_nn, self.m_disturb)]
        m_disturbed = tools.clip_vector(m_disturbed, self.m_channels)
        m_signal = tools.to_signal(m_disturbed, self.m_channels)

        return m_signal
