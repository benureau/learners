from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

import numpy as np
imle = None # the imle package will be imported at the first class instanciation.


from .. import learner
from .. import tools


defcfg = learner.Learner.defcfg._deepcopy()
defcfg.classname = 'learners.ImleLearner'
defcfg._describe('sigma0', instanceof=(numbers.Real), default=1.0/30,
                 docstring='sigma0 parameter for imle')
defcfg._describe('psi0', instanceof=(numbers.Real, collections.Iterable), default=(1.0/30)**2,
                 docstring='psi0 parameter for imle')


class ImleLearner(learner.Learner):
    """"""
    defcfg = defcfg

    def __init__(self, cfg):
        super(ImleLearner, self).__init__(cfg)
        self._dynamic_import()
        self.sigma0 = self.cfg.sigma0
        self.psi0   = self.cfg.psi0
        if isinstance(self.psi0, numbers.Real):
            self.psi0 = np.array([self.psi0 for _ in range(len(self.s_channels))])
        self.imle = imle.Imle(d=len(self.m_channels), D=len(self.s_channels), sigma0=self.sigma0, psi0=self.psi0)

    @classmethod
    def _dynamic_import(cls):
        global imle
        if imle is None:
            try:
                import imle
            except ImportError:
                print('To use the ImleLearner, you need to install imle first')
                raise

    def _predict(self, data):
        """Predict the effect of an order"""
        return None

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        y = np.array(tools.to_vector(s_signal, self.s_channels))

        res = self.imle.predict_inverse(y, weight=True)
        return tools.to_signal(res['prediction'][np.argmax(res['weight'])],
                               self._m_channels)

    def _update(self, m_signal, s_signal, uuid=None):
        m_v = np.array(tools.to_vector(m_signal, self._m_channels))
        s_v = np.array(tools.to_vector(s_signal, self.s_channels))
        self.imle.update(m_v, s_v)
