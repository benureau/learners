from __future__ import absolute_import, division, print_function
import random

from . import disturb


defcfg = disturb.DisturbLearner.defcfg._deepcopy()
defcfg.classname = 'learners.DisturbTwoStepLearner'

class DisturbTwoStepLearner(disturb.DisturbLearner):
    """"""
    defcfg = defcfg

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        perturb = [random.uniform(0, d_i) for d_i in self.m_disturb]
        return self._disturb(s_signal, perturb)
