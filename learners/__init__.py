from __future__ import absolute_import

from .learner import Learner
from .channels import Channel
from .nn_set import NNSet
from . import tools

from .algorithms.rand           import RandomLearner
from .algorithms.nn             import NNLearner
from .algorithms.mutate_nn      import MutateNNLearner
from .algorithms.lwlr           import LWLRLearner
from .algorithms.lwlr           import ESLWLRLearner
from .algorithms.predict_mutate import PredictMutateNNLearner
from .algorithms.optimize       import OptimizeLearner

from .algorithms.imle_model     import ImleLearner

# versioneer
from ._version import get_versions
__version__ = get_versions()["version"]
__commit__ = get_versions()["full-revisionid"]
__dirty__ = get_versions()["dirty"]
del get_versions

# fastlearners
from .algorithms import lwlr
from .algorithms import lwlr_cpp

def enable_fastlearners(silent_fail=False):
    global LWLRLearner, ESLWLRLearner
    try:
        import fastlearners
        LWLRLearner = lwlr_cpp.cLWLRLearner
        ESLWLRLearner = lwlr_cpp.cESLWLRLearner

    except ImportError:
        if not silent_fail:
            print('warning: `fastlearners` could not be imported, defaulting to (slower) python implementations for LWLR.')

def disable_fastlearners():
    global LWLRLearner, ESLWLRLearner
    LWLRLearner = lwlr.LWLRLearner
    ESLWLRLearner = lwlr.ESLWLRLearner

enable_fastlearners(silent_fail=True)
