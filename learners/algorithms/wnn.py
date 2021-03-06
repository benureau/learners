from __future__ import absolute_import, division, print_function

from .. import learner
from .. import nn


defcfg = learner.Learner.defcfg._deepcopy()
defcfg.classname = 'learners.WeightedNNLearner'
defcfg._describe('m_k', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Number of neighbors to average from with motor signal')
defcfg._describe('s_k', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Number of neighbors to average from with sensory signal')

class WeightedNNLearner(learner.Learner):
    """"""

    defcfg = defcfg

    def __init__(self, cfg, nnset=None):
        super(NNLearner, self).__init__(cfg)
        self.nnset = nnset if nnset is not None else nn.NNSet()

    def _predict(self, data):
        """Predict the effect of an order"""
        m_v = tool.to_vector(m_signal)
        dists, m_idx = self.nnset.nn_x(m_v, k=self.cfg.m_k)
        s_vector = self.nnset.ys[m_idx[0]]
        return tools.to_signal(s_vector, self.s_channels)

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        s_v = tool.to_vector(s_signal)
        dists, s_idx = self.nnset.nn_y(s_v, k=self.cfg.s_k)
        m_vector = self.nnset.xs[s_idx[0]]
        return tools.to_signal(m_vector, self._m_channels)

    def _update(self, m_signal, s_signal, uuid=None):
        m_v = tools.to_vector(m_signal)
        s_v = tools.to_vector(s_signal)
        nnset.add(m_v, s_v, uuid=None)
