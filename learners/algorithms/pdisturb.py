from __future__ import absolute_import, division, print_function
import random
import numbers

from .. import tools
from . import disturb



defcfg = disturb.DisturbLearner.defcfg._deepcopy()
defcfg.classname = 'learners.PredictDisturbLearner'
defcfg._describe('attempts', instanceof=numbers.Integral, default=5,
                 docstring='number of potential solution to try')
defcfg._branch('fwd')


class PredictDisturbLearner(disturb.DisturbLearner):
    """"""
    defcfg = defcfg

    def __init__(self, *args, **kwargs):
        super(PredictDisturbLearner, self).__init__(*args, **kwargs)

        self.cfg.fwd.m_channels = self.m_channels
        self.cfg.fwd.s_channels = self.s_channels
        self.fwd = self.create(self.cfg.fwd)

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        assert self.cfg.attempts >= 1

        min_dist = float('inf')
        s_vector = tools.to_vector(s_signal, self.s_channels)
        ms_best  = None

        for i in range(self.cfg.attempts):
            m_signal = self._disturb(s_signal, self.m_disturb)
            s_pred   = self.fwd.predict(m_signal)
            s_pred_v = tools.to_vector(s_pred, self.s_channels)
            s_dist   = tools.dist_sq(s_vector, s_pred_v)
            if s_dist < min_dist:
                min_dist = s_dist
                m_best   = m_signal

        return m_best

    def _update(self, m_signal, s_signal, uuid=None):
        self.fwd.update(m_signal, s_signal, uuid=uuid)
        super(PredictDisturbLearner, self)._update(m_signal, s_signal, uuid=uuid)

