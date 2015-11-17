import time
import random

import numpy as np

import dotdot
import learners

random.seed(0)
np.random.seed(0)

def random_linear(n, m):
    """Create a random linear function from R^n -> R^m"""
    M = np.random.rand(n,m)
    return lambda x : np.dot(np.array(x), M).ravel()

def bench_lwlr_linear_batch(cpp, K=1000):
    """Test LWLR on random linear models of dimensions from 1 to 20.
     It should return exact results, give of take floating point imprecisions."""
    random.seed(0)
    np.random.seed(0)

    if cpp:
        learners.enable_fastlearners(silent_fail=False)
    else:
        learners.disable_fastlearners()

    for i in range(K):
        n = random.randint(1, 20)
        m = random.randint(1, 5)
        f = random_linear(n, m)
        cfg = {'m_channels'  : [learners.Channel('x_{}'.format(i), (0.0, 1.0))
                                for i in range(n)],
               's_channels'  : [learners.Channel('y_{}'.format(i), (0.0, 1.0))
                                for i in range(m)],
               'm_uniformize': True,
               'sigma'       : 1.0}

        for learner in [learners.LWLRLearner(cfg), learners.ESLWLRLearner(cfg)]:

            for i in range(500):
                x = np.random.rand(n)
                y = f(x)
                learner.update(learners.tools.to_signal(x, cfg['m_channels']),
                               learners.tools.to_signal(y, cfg['s_channels']))

            for i in range(50):
                x = np.random.rand(n).ravel()
                y = f(x)
                yp = learner.predict(learners.tools.to_signal(x, cfg['m_channels']))
                yp = learners.tools.to_vector(yp, cfg['s_channels'])
                assert np.allclose(y, yp, rtol = 1e-5, atol = 1e-5)


def bench_lwlr_linear_online(cpp, K=1000):
    """Test LWLR on random linear models of dimensions from 1 to 20.
     It should return exact results, give of take floating point imprecisions."""
    random.seed(0)
    np.random.seed(0)

    if cpp:
        learners.enable_fastlearners(silent_fail=False)
    else:
        learners.disable_fastlearners()

    for i in range(K):
        n = random.randint(1, 20)
        m = random.randint(1, 5)
        f = random_linear(n, m)
        cfg = {'m_channels'  : [learners.Channel('x_{}'.format(i), (0.0, 1.0))
                                for i in range(n)],
               's_channels'  : [learners.Channel('y_{}'.format(i), (0.0, 1.0))
                                for i in range(m)],
               'm_uniformize': True,
               'sigma'       : 1.0}

        for learner in [learners.LWLRLearner(cfg), learners.ESLWLRLearner(cfg)]:

            for i in range(50):
                x = np.random.rand(n)
                y = f(x)
                learner.update(learners.tools.to_signal(x, cfg['m_channels']),
                               learners.tools.to_signal(y, cfg['s_channels']))

            for i in range(950):
                x = np.random.rand(n)
                y = f(x)
                learner.update(learners.tools.to_signal(x, cfg['m_channels']),
                               learners.tools.to_signal(y, cfg['s_channels']))

                x = np.random.rand(n).ravel()
                y = f(x)
                yp = learner.predict(learners.tools.to_signal(x, cfg['m_channels']))
                yp = learners.tools.to_vector(yp, cfg['s_channels'])
                assert np.allclose(y, yp, rtol = 1e-5, atol = 1e-5)



K = 50

start = time.time()
bench_lwlr_linear_batch(True, K=K)
print('lwlr_cpp  [batch]: {:3.1f}s'.format(time.time()-start))

start = time.time()
bench_lwlr_linear_batch(False, K=K)
print('lwlr      [batch]: {:3.1f}s'.format(time.time()-start))

K = 10

start = time.time()
bench_lwlr_linear_online(True, K=K)
print('lwlr_cpp [online]: {:3.1f}s'.format(time.time()-start))

start = time.time()
bench_lwlr_linear_online(False, K=K)
print('lwlr     [online]: {:3.1f}s'.format(time.time()-start))
