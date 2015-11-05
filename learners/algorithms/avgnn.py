from __future__ import absolute_import, division, print_function

from . import nn


defcfg = learner.Learner.defcfg._deepcopy()
defcfg.classname = 'learners.AvgNNLearner'
defcfg._describe('m_k', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Number of neighbors to average from to predict')
defcfg._describe('s_k', instanceof=(numbers.Real, collections.Iterable),
                 docstring='Number of neighbors to average from to infer')

class AvgNNLearner(nn.NNLearner):
    """"""

    defcfg = defcfg

    def __init__(self, cfg, nnset=None):
        super(AvgNNLearner, self).__init__(cfg)

    def _predict(self, data):
        """Predict the effect of an order"""
        m_v = tool.to_vector(m_signal)
        dists, m_idx = self.nnset.nn_x(m_v, k=self.cfg.m_k)
        s_vector = np.average([self.nnset.ys[idx_i] for idx_i in m_idx])
        return tools.to_signal(s_vector, self.s_channels)

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        s_v = tool.to_vector(s_signal)
        dists, s_idx = self.nnset.nn_y(s_v, k=1)
        dists, m_idx = self.nnset.nn_x(self.nnset.xs[s_idx[0]], k=self.cfg.s_k)
        m_vector = np.average([self.nnset.xs[idx_i] for idx_i in m_idx])
        return tools.to_signal(m_vector, self._m_channels)
