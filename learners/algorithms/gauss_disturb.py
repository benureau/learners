from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

from .. import tools
from . import disturb


defcfg = disturb.DisturbLearner.defcfg._copy(deep=True)
defcfg.classname = 'learners.GaussDisturbLearner'

class GaussDisturbLearner(disturb.DisturbLearner):
    """"""
    defcfg = defcfg

    def _perturbation(self, m_vector):
        m_disturbed = []
        for v_i, d_i, c_i in zip(m_vector, self.m_disturb, self._m_channels):
            sigma = max(d_i, v_i - c_i.bounds[0], c_i.bounds[1] - v_i) # adjusting sigma near boundaries
            m_i = random.gauss(v_i, sigma/4)
            m_disturbed.append(max(min(m_i, c_i.bounds[1]), c_i.bounds[0]))
        return m_disturbed
