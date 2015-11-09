from __future__ import absolute_import, division, print_function
import random
import numbers
import collections

from .. import tools
from . import nn

from .. import operators

defcfg = nn.NNLearner.defcfg._deepcopy()
defcfg.classname = 'learners.MutateNNLearner'
defcfg._branch('operator', value=operators.get('uniform').defcfg)


class MutateNNLearner(nn.NNLearner):
    """\
    Nearest-Neighbor Mutation Inverse Learner.

    Take the one input whose output is closest to the goal, and
    create a mutation of it.
    """
    defcfg = defcfg

    def __init__(self, cfg, nnset=None):
        super(MutateNNLearner, self).__init__(cfg)

        op_class = operators.get(self.cfg.operator.name)
        self._operator = op_class(self.cfg.operator, self.m_channels)

    def _predict(self, m_signal):
        """Predict the effect of an order"""
        return None

    def _infer(self, s_signal):
        """Infer the motor command to obtain an effect"""
        if len(self.nnset) == 0:
            return None
        s_v = tools.to_vector(s_signal, self.s_channels)
        dists, s_idx = self.nnset.nn_y(s_v, k = 1)
        #print('nn_idx={}'.format(s_idx[0]))
        m_nn = self.nnset.xs[s_idx[0]]

        m_mutated = self._operator.mutate(m_nn)
        return tools.to_signal(m_mutated, self._m_channels)
