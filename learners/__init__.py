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

from .algorithms.models_wrap    import ModelLearner
from .algorithms.imle_model     import ImleLearner

# versioneer
from ._version import get_versions
__version__ = get_versions()["version"]
__commit__ = get_versions()["full-revisionid"]
__dirty__ = get_versions()["dirty"]
del get_versions
