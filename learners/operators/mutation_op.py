import random
import abc
import numbers
import collections

import scicfg

from .. import tools


def create(cfg, **kwargs):
    class_ = tools._load_class(cfg.classname)
    return class_(cfg, **kwargs)


_mutators = {}
def register(name, cls):
    _mutators[name] = cls

def get(name):
    return _mutators[name]


defcfg = scicfg.SciConfig()
defcfg._describe('classname', instanceof=str)

class MutationOperator(object):

    __metaclass__ = abc.ABCMeta
    defcfg = defcfg

    def __init__(self, cfg, channels):
        self.cfg = cfg

    def mutate(self, vector):
        """Create a mutation of the vector v"""
        raise NotImplemented


ro_cfg = defcfg._deepcopy()
ro_cfg.classname = 'learners.operators.UniformOperator'
ro_cfg._describe('name', instanceof=str, default='uniform',
                 docstring='Name of the operator.')
ro_cfg._describe('d', instanceof=(numbers.Real, collections.Iterable), default=0.05,
                 docstring='Perturbation factor for the mutation.')
ro_cfg._describe('proba', instanceof=(numbers.Real, collections.Iterable), default=1.0,
                 docstring='Probability of a mutation.')
ro_cfg._describe('relative', instanceof=bool, default=True,
                 docstring='If True, d is considered relative to the bounds of the vector channels.')
ro_cfg._freeze(True)

class UniformOperator(MutationOperator):

    defcfg = ro_cfg

    def __init__(self, cfg, channels):
        self.cfg = cfg
        self.cfg._update(self.defcfg, overwrite=False)
        self.channels = channels

        self.d = self.cfg.d
        if isinstance(self.d, numbers.Real):
            self.d = tuple(self.d for c in self.channels)
        self.proba = self.cfg.proba
        if isinstance(self.proba, numbers.Real):
            self.proba = tuple(self.proba for c in self.channels)

    def mutate(self, vector):
        """Return a perturbation of a motor vector"""
        # we draw the perturbation inside legal values, rather than clamp it afterward
        mutated = []
        for v_i, p_i, d_i, c_i in zip(vector, self.proba, self.d, self.channels):
            if p_i < random.random():
                mutated.append(mutate_vi(v_i, d_i, c_i))
            mutated.append(v_i)
        return mutated

    def mutate_vi(self, v_i, d_i, c_i):
        return random.uniform(max(v_i - d_i, c_i.bounds[0]),
                              min(v_i + d_i, c_i.bounds[1]))

register('uniform', UniformOperator)


ro_clamp_cfg = ro_cfg._deepcopy()
ro_clamp_cfg.classname = 'learners.operators.RandomClampOperator'
ro_clamp_cfg._freeze(True)

class UniformClampOperator(UniformOperator):
    defcfg = ro_clamp_cfg
    def mutate_vi(self, v_i, d_i, c_i):
        vm_i = random.uniform(v_i - d_i, v_i + d_i)
        return min(c_i.bounds[1], max(c_i.bounds[0], vm_i))

register('uniformclamp', UniformClampOperator)


ro2_cfg = ro_cfg._deepcopy()
ro2_cfg.classname = 'learners.operators.UniformSyncOperator'
ro2_cfg._freeze(True)

class UniformSyncOperator(UniformOperator):

    defcfg = ro2_cfg

    def __init__(self, cfg, channels):
        super(UniformSyncOperator, self).__init__(cfg, channels)
        assert all(self.d[0] == d_i for d_i in self.d)

    def mutate(self, vector):
        """Return a perturbation of a motor vector"""
        # we draw the perturbation inside legal values, rather than clamp it afterward
        mutated = []

        for v_i, p_i, d_i, c_i in zip(vector, self.proba, self.d, self.channels):
            if p_i < random.random():
                mutated.append(mutate_vi(v_i, d_i, c_i))
            mutated.append(v_i)
        return mutated

register('uniformsync', UniformSyncOperator)


gauss_cfg = ro_cfg._deepcopy()
gauss_cfg.classname = 'learners.operators.GaussOperator'
gauss_cfg._freeze(True)

class GaussOperator(UniformOperator):
    defcfg = gauss_cfg
    def mutate_vi(self, v_i, d_i, c_i):
        m_i = random.gauss(v_i, d_i)
        return min(c_i.bounds[1], max(c_i.bounds[0], vm_i))

register('gauss', GaussOperator)
