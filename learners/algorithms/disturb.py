from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

from .. import tools
from . import nn


defcfg = nn.NNLearner.defcfg._copy(deep=True)
defcfg.classname = 'learners.DisturbLearner'
defcfg._describe('m_disturb', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Maximum distance of disturbance along each dimension.')

class DisturbLearner(nn.NNLearner):
    """"""
    defcfg = defcfg

    def __init__(self, cfg, nnset=None):
        super(DisturbLearner, self).__init__(cfg)

        self.m_disturb = self.cfg.m_disturb
        if isinstance(self.m_disturb, numbers.Real):
            self.m_disturb = [self.m_disturb for c in self.uni_m_channels]

    def _predict(self, data):
        """Predict the effect of an order"""
        return None

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        return self._disturb(s_signal, self.m_disturb)

    def _disturb(self, s_signal, perturb):
        """"""
        if len(self.nnset) == 0:
            return None
        s_v = tools.to_vector(s_signal, self.s_channels)
        dists, s_idx = self.nnset.nn_y(s_v, k = 1)
        m_nn = self.nnset.xs[s_idx[0]]

        # we draw the perturbation inside legal values, rather than clamp it afterward
        m_disturbed = [random.uniform(max(v_i - d_i, c_i.bounds[0]),
                                      min(v_i + d_i, c_i.bounds[1]))
                       for v_i, d_i, c_i in zip(m_nn, self.m_disturb, self.uni_m_channels)]
        m_signal = tools.to_signal(m_disturbed, self.uni_m_channels)

        return m_signal
