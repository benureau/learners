# -*- coding: utf-8 -*-

"""

Locally Weigthed Regression (LWR) python implementation.

References :
    1. C. G. Atkeson, A. W. Moore, S. Schaal, "Locally Weighted Learning for Control",
         "Springer Netherlands", 75-117, vol 11, issue 1, 1997/02, 10.1023/A:1006511328852
    For a video lecture :
    2. http://www.cosmolearning.com/video-lectures/locally-weighted-regression-probabilistic-interpretation-logistic-regression/

Pseudo Code :

    Input X matrix of inputs:  X[k] [i] = i'th component of k'th input point.
    Input Y matrix of outputs: Y[k] = k'th output value.
    Input xq = query input.    Input kwidth.

    WXTWX = empty (D+1) x (D+1) matrix
    WXTWY = empty (D+1) x 1     matrix

    for k in range(N):

        /* Compute weight of kth point  */
        wk = weight_function( distance( xq , X[k] ) / kwidth )

        /* Add to (WX) ^T (WX) matrix */
        for ( i = 0 ; i <= D ; i = i + 1 )
            for ( j = 0 ; j <= D ; j = j + 1 )
                if ( i == 0 )
                    xki = 1 else xki = X[k] [i]
                if ( j == 0 )
                    xkj = 1 else xkj = X[k] [j]
                WXTWX [i] [j] = WXTWX [i] [j] + wk * wk * xki * xkj

        /*  Add to (WX) ^T (WY) vector */
        for ( i = 0 ; i <= D ; i = i + 1 )
            if ( i == 0 )
                xki = 1 else xki = X[k] [i]
            WXTWY [i] = WXTWY [i] + wk * wk * xki * Y[k]

    /* Compute the local beta.  Call your favorite linear equation solver.
       Recommend Cholesky Decomposition for speed.
       Recommend Singular Val Decomp for Robustness. */

    Beta = (WXTWX)^{-1}(WXTWY)

    Output ypredict = beta[0] + beta[1]*xq[1] + beta[2]*xq[2] + … beta[D]*x q[D]

"""
import numbers

import numpy as np

from .. import tools
from . import nn




def gaussian_kernel(d, sigma_sq):
    """Compute the guassian kernel function of a given distance
    @param d         the euclidean distance
    @param sigma_sq  sigma of the guassian, squared.
    """
    return np.exp(-(d*d)/(2*sigma_sq))


defcfg = nn.NNLearner.defcfg._deepcopy()
defcfg.classname = 'learners.LWLRLearner'
defcfg._describe('sigma', instanceof=numbers.Real, default=1.0)
defcfg._freeze(True)

class LWLRLearner(nn.NNLearner):
    """Locally Weighted Linear Regression Forward Model"""

    name = 'LWLR'
    desc = 'LWLR, Locally Weighted Linear Regression'
    defcfg = defcfg

    def __init__(self, cfg):
        """Create the forward model

        @param sigma    sigma for the guassian distance.
        @param nn       the number of nearest neighbors to consider for regression.
        """
        super(LWLRLearner, self).__init__(cfg)
        self.dim_x = len(self.cfg.m_channels)
        self.dim_y = len(self.cfg.s_channels)
        self.k = max(3, int(1.1*self.dim_x+1))
        self.sigma_sq = self.cfg.sigma*self.cfg.sigma

    @property
    def sigma(self):
        return self.cfg['sigma']

    @sigma.setter
    def sigma(self, sigma):
        self.sigma_sq = sigma*sigma
        self.cfg['sigma'] = sigma

    ### LWR regression
    def _predict(self, m_signal):
        m_vector = tools.to_vector(m_signal, self.m_channels)
        s_vector = self._predict_v(m_vector)
        return tools.to_signal(s_vector, self.s_channels)

    def _predict_v(self, xq):
        dists, index = self.nnset.nn_x(xq, k=self.k)

        w = self._weights(dists, self.sigma_sq)

        Xq  = np.array(np.append([1.0], xq), ndmin = 2)
        X   = np.array([np.append([1.0], self.nnset.xs[i]) for i in index])
        Y   = np.array([self.nnset.ys[i] for i in index])

        # from tools import gfx
        # samples = [(d_i, w_i, tuple(self.dataset.get_x(i)), tuple(self.dataset.get_y(i))) for d_i, i, w_i in zip(dists, index, w)]
        # for d_i, w_i, x_i, y_i in sorted(samples):
        #     print('{}{:7.5f}/{:7.5f}:  {} -> {}{}'.format(gfx.cyan, d_i, w_i, gfx.ppv(x_i, fmt=' 5.2f'), gfx.ppv(y_i, fmt=' 5.2f'), gfx.end))
        # print('')

        W   = np.diag(w)
        WX  = np.dot(W, X)
        WXT = WX.T

        B   = np.dot(np.linalg.pinv(np.dot(WXT, WX)),WXT)

        self.mat = np.dot(B, np.dot(W, Y))
        Yq  = np.dot(Xq, self.mat)

        return Yq.ravel()

    def _weights(self, dists, sigma_sq):
        #print('sigma: {}'.format(sigma_sq))

        w = np.fromiter((gaussian_kernel(d, sigma_sq)
                         for d in dists), np.float, len(dists))

        wsum = w.sum()
        if wsum == 0:
            return 1.0/len(dists)*np.ones((len(dists),))
        else:
            eps = wsum * 1e-10 / self.dim_x
            return np.fromiter((w_i/wsum if w_i > eps else 0.0 for w_i in w), np.float)


defcfg_es = LWLRLearner.defcfg._deepcopy()
defcfg_es.classname = 'learners.ESLWLRLearner'
defcfg._freeze(True)

class ESLWLRLearner(LWLRLearner):
    """ES-LWLR : LWLR with estimated sigma, on a query basis, as the mean distance."""

    name = 'ES-LWLR'
    defcfg = defcfg_es

    def _weights(self, dists, sigma_sq=None):
        sigma_sq=(dists**2).sum()/len(dists)/2
        return super(ESLWLRLearner, self)._weights(dists, sigma_sq)
