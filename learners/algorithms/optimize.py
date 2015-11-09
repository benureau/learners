import numbers

import numpy as np
import scipy.optimize

from .. import learner
from .. import tools
from . import lwlr


defcfg = learner.Learner.defcfg._deepcopy()
defcfg.classname = 'leaners.OptimizeLearner'
defcfg._branch('fwd', value=lwlr.ESLWLRLearner.defcfg._deepcopy())
defcfg._describe('algo', instanceof=str, default='L-BFGS-B')
defcfg._branch('options', strict=False)
# some default values
defcfg.options._describe('disp', instanceof=bool, default=False)
defcfg.options._describe('maxiter', instanceof=numbers.Integral, default=500)
#defcfg.options._describe('ftol', instanceof=numbers.Real, default=1e-5)
#defcfg.options._describe('gtol', instanceof=numbers.Real, default=1e-3)

class OptimizeLearner(learner.Learner):
    """
    An inverse model class using optimization class of scipy (e.g. gradient descent, BFGS),
    on an error function computed from the forward model.
    """

    defcfg = defcfg._deepcopy()

    def __init__(self, cfg):
        super(OptimizeLearner, self).__init__(cfg)
        self.cfg.fwd.m_channels = self.m_channels
        self.cfg.fwd.s_channels = self.s_channels
        self.cfg.fwd.m_uniformize = self.cfg.m_uniformize
        self.fwd = self.create(self.cfg.fwd)
        try:
            self.nnset = self.fwd.nnset
        except AttributeError:
            self.nnset = nn_set.NNSet()
        self.bounds = [c.bounds for c in self.m_channels]

    def _update(self, m_signal, s_signal, uuid=None):
        m_v = tools.to_vector(m_signal, self._m_channels)
        s_v = tools.to_vector(s_signal, self.s_channels)
        self.fwd._update(m_signal, s_signal, uuid=uuid)
        try:
            if self.nnset is not self.fwd.nnset:
                self.nnset.add(m_v, s_v, uuid=uuid)
        except AttributeError:
            self.nnset.add(m_v, s_v, uuid=uuid)

    def _initialize(self, y_goal, **kwargs):
        dists, index = self.nnset.nn_y(y_goal, k=1)
        return self.nnset.xs[index[0]]

    def _predict(self, m_signal):
        return self.fwd._predict(m_signal)

    def _infer(self, s_signal):
        y_goal = tools.to_vector(s_signal, self.s_channels)

        x_guess = self._initialize(y_goal)

        def error_fun(x):
            y_pred = self.fwd._predict_v(x)
            err_v  = y_pred - y_goal
            error = sum(e*e for e in err_v)
            return error

        res = scipy.optimize.minimize(error_fun, x_guess,
                                      args        = (),
                                      method      = self.cfg.algo,
                                      bounds      = self.bounds,
                                      options     = {k:v for k, v in self.cfg.options._items()}
                                     )

        return tools.to_signal(self._enforce_bounds(res.x), self.m_channels)

    def _enforce_bounds(self, x):
        """"Enforce the bounds on x if only infinitesimal violations occurs"""
        assert len(x) == len(self.bounds)
        x_enforced = []
        for x_i, (lb, ub) in zip(x, self.bounds):
            if x_i < lb:
                if x_i > lb - (ub-lb)/1e10:
                    x_enforced.append(lb)
                else:
                    x_enforced.append(x_i)
            elif x_i > ub:
                if x_i < ub + (ub-lb)/1e10:
                    x_enforced.append(ub)
                else:
                    x_enforced.append(x_i)
            else:
                x_enforced.append(x_i)
        return np.array(x_enforced)

# class COBYLAInverseModel(ScipyInverseModel):
#     """Class that takes specialized COBYLA options"""
#
#     name = 'COBYLA'
#     desc = 'COBYLA'
#     algo = 'COBYLA'
#
#     def __init__(self, dim_x, dim_y, constraints = (),
#                  maxiter =  500,
#                  rhoend  = 1e-3,
#                  rhobeg  =  1.0,
#                  disp    = False,
#                  **kwargs):
#         """
#         * COBYLA options (from scipy doc):
#         COBYLA options:
#         rhobeg : float
#             Reasonable initial changes to the variables.
#         rhoend : float
#             Final accuracy in the optimization (not precisely guaranteed).
#             This is a lower bound on the size of the trust region.
#         maxfev : int
#             Maximum number of function evaluations.
#         """
#         OptimizedInverseModel.__init__(self, dim_x, dim_y, constraints = constraints, **kwargs)
#         self.bounds = constraints
#         self.conf   = {'maxiter': maxiter,
#                        'tol'    : rhoend,
#                        'rhobeg' : rhobeg,
#                        'disp'    : disp,
#                       }
