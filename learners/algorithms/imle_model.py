from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

import numpy as np
try:
    import imle
except ImportError:
    print('To use the ImleLearner, you need to install imle first')
    raise

from .. import learner
from .. import tools


defcfg = learner.Learner.defcfg._copy(deep=True)
defcfg.classname = 'learners.ImleLearner'
defcfg._describe('m_disturb', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Maximum distance of disturbance along each dimension.')

class ImleLearner(learner.Learner):
    """"""
    defcfg = defcfg

    def __init__(self, cfg):
        super(ImleLearner, self).__init__(cfg)
        self.imle = imle.Imle(d=len(self.m_channels), D=len(self.s_channels))

    def _predict(self, data):
        """Predict the effect of an order"""
        return None

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        y = np.array(tools.to_vector(s_signal, self.s_channels))

        res = self.imle.predict_inverse(y, weight=True)
        print(res)

        print(res['prediction'][np.argmax(res['weight'])])

        return tools.to_signal(res['prediction'][np.argmax(res['weight'])],
                               self._uni_m_channels)

    def _update(self, m_signal, s_signal, uuid=None):
        m_v = np.array(tools.to_vector(m_signal, self._uni_m_channels))
        s_v = np.array(tools.to_vector(s_signal, self.s_channels))
        self.imle.update(m_v, s_v)
