## Replacing classes by faster `fastlearners` implementation if available

import math

try:
    import fastlearners
except ImportError:
    pass

from .. import tools
from .lwlr import LWLRLearner

class cLWLRLearner(LWLRLearner):

    def __init__(self, cfg):
        """LWLR based on the C++ implementation.

        @param sigma    sigma for the guassian distance.
        @param nn       the number of nearest neighbors to consider for regression.
        """
        super(cLWLRLearner, self).__init__(cfg)
        self.lwlr = fastlearners.cLWLR(self.dim_x, self.dim_y, sigma=math.sqrt(self.sigma_sq), k=self.k)

    @property
    def sigma(self):
        return self.lwlr.sigma

    @sigma.setter
    def sigma(self, sigma):
        self.lwlr.sigma = sigma

    def _update(self, m_signal, s_signal, uuid=None):
        m_v = tools.to_vector(m_signal, self._m_channels)
        s_v = tools.to_vector(s_signal, self.s_channels)
        self.lwlr.add_xy(m_v, s_v)

    ### LWR regression
    def _predict(self, m_signal):
        m_vector = tools.to_vector(m_signal, self.m_channels)
        s_vector = self.lwlr.predict(m_vector)
        return tools.to_signal(s_vector, self.s_channels)


class cESLWLRLearner(cLWLRLearner):
    """ES-LWLR : LWLR with estimated sigma, on a query basis, as the mean distance.
    Based on C++ implementation.
    """
    def __init__(self, *args, **kwargs):
        cLWLRLearner.__init__(self, *args, **kwargs)
        self.es = True
