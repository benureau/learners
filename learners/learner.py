from __future__ import absolute_import, division, print_function
import random
import collections
import abc

import forest

from . import tools
from .channels import Channel


defcfg = forest.Tree()
defcfg._describe('m_channels', instanceof=collections.Iterable,
                 docstring='Motor channels to generate random order of')
defcfg._describe('s_channels', instanceof=collections.Iterable,
                 docstring='Sensory channels to generate random goal from')
defcfg._describe('classname', instanceof=collections.Iterable,
                 docstring='The name of the learner class. Only used with the create() class method.')
defcfg._describe('m_uniformize', instanceof=bool, default=True,
                 docstring='If True, motor signal will be uniformized as a preprocessing step')


class Learner(object):
    """"""
    __metaclass__ = abc.ABCMeta

    defcfg = defcfg

    @classmethod
    def create(cls, cfg, **kwargs):
        class_ = tools._load_class(cfg.classname)
        return class_(cfg, **kwargs)

    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = forest.Tree(cfg)
        self.cfg = cfg
        self.cfg._update(self.defcfg, overwrite=False)
        self.s_channels = cfg.s_channels
        self.m_channels = cfg.m_channels
        if self.cfg.m_uniformize:
            self.uni_m_channels = [Channel(c.name, bounds=(0., 1.), fixed=c.fixed) for c in self.m_channels]
        else:
            self.uni_m_channels = self.m_channels

        self.s_names    = set(c.name for c in self.s_channels)
        self.m_names    = set(c.name for c in self.m_channels)
        self.uuids = set()
        self._uuid_offset = 0

    def predict(self, m_signal):
        """ Predict the effect of an order.
            Inherited classes should override `_predict`.
        """
        if self.m_names == set(m_signal.keys()):
            if self.cfg.m_uniformize:
                m_signal = tools.uniformize_signal(m_signal, self.m_channels)
            return self._predict(m_signal)


    def infer(self, s_signal):
        """ Infer a motor command to obtain the given sensory signal.
            Inherited classes should override `_infer`.
        """
        if self.s_names == set(s_signal.keys()):
            m_signal = self._infer(s_signal)
            if m_signal is None:
                return None
            if self.cfg.m_uniformize:
                m_signal = tools.restore_signal(m_signal, self.m_channels)

            return m_signal


    def update(self, m_signal, s_signal, uuid=None):
        """ Update the model with a (motor_signal, sensory_signal) observation.
            Inherited classes should override `_update`.
        """
        if uuid is None or uuid not in self.uuids:
            # verify that this observation is compatible with current channels.
            if (self.s_names <= set(s_signal.keys()) and
                self.m_names <= set(m_signal.keys())):
                # register the uuid so that we don't update the same obs twice.
                self.uuids.add(uuid)
                if uuid is None:
                    self._uuid_offset += 1

                # uniformize the motor signal if enabled
                if self.cfg.m_uniformize:
                    m_signal = tools.uniformize_signal(m_signal, self.m_channels)

                self._update(m_signal, s_signal, uuid=uuid)


    def fwd_request(self, request):
        m_signal   = request['m_signal']
        s_channels = request['s_channels']

        if set(c.name for c in s_channels) == self.s_names:
            return self.predict(m_signal)

    def inv_request(self, request):
        s_signal   = request['s_goal']
        m_channels = request['m_channels']

        if set(c.name for c in m_channels) == self.m_names:
            return self.infer(s_signal)

    def update_request(self, request):
        uuid     = request['uuid']
        m_signal = request['m_signal']
        s_signal = request['s_signal']

        self.update(m_signal, s_signal, uuid=uuid)

    @abc.abstractmethod
    def _infer(self, s_signal):
        """Should return a m_signal or None"""
        return None

    @abc.abstractmethod
    def _predict(self, m_signal):
        """Should return a s_signal or None"""
        return None

    @abc.abstractmethod
    def _update(self, m_signal, s_signal):
        pass

    def __len__(self):
        return len(self.uuids) + self._uuid_offset
