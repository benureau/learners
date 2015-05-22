from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

from .. import tools
from . import disturb


defcfg = disturb.DisturbLearner.defcfg._copy(deep=True)
defcfg.classname = 'learners.DisturbClampLearner'

class DisturbClampLearner(disturb.DisturbLearner):
    """"""
    defcfg = defcfg

    def _disturb(self, s_signal, perturb):
        """"""
        if len(self.nnset) == 0:
            return None
        s_v = tools.to_vector(s_signal, self.s_channels)
        dists, s_idx = self.nnset.nn_y(s_v, k = 1)
        m_nn = self.nnset.xs[s_idx[0]]

        # we draw the perturbation inside legal values, rather than clamp it afterward
        m_disturbed = [random.uniform(v_i - d_i, v_i + d_i)
                       for v_i, d_i in zip(m_nn, self.m_disturb)]
        m_disturbed = [min(max(v_i, c_i.bounds[0]), c_i.bounds[1])
                       for v_i, c_i in zip(m_disturbed, self._uni_m_channels)]

        return tools.to_signal(m_disturbed, self._uni_m_channels)
