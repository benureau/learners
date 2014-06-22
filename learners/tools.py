"""A collections of """
from __future__ import absolute_import, division

import importlib
import collections
import random

def _load_class(classname):
    """Load a class from a string"""
    module_name, class_name = classname.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def uniformize_signal(signal, channels):
    """ Uniformize a signal between 0.0 and 1.0
        Requires every channel to have bounds, or be discretized.
    """
    uni_signal = {}
    for c in channels:
        factor = 1.0
        if c.bounds[0] != c.bounds[1]:
            assert c.bounds[0] < c.bounds[1]
            factor = c.bounds[1] - c.bounds[0]
        uni_signal[c.name] = (signal[c.name]-c.bounds[0])/factor
    return uni_signal

def restore_signal(uni_signal, channels):
    """ Uniformize a signal between 0.0 and 1.0
        Requires every channel to have bounds, or be discretized.
    """
    signal = {}
    for c in channels:
        factor = 1.0
        if c.bounds[0] != c.bounds[1]:
            assert c.bounds[0] < c.bounds[1]
            factor = c.bounds[1] - c.bounds[0]
        signal[c.name] = uni_signal[c.name]*factor + c.bounds[0]
    return signal

def to_vector(signal, channels=None):
    """Convert a signal to a vector"""
    if channels is None:
        # we need consistent ordering
        assert isinstance(signal, collections.OrderedDict)
        return tuple(signal.values())
    else:
        return tuple(signal[c.name] for c in channels)

def to_signal(vector, channels):
    """Convert a vector to a signal"""
    assert len(vector) == len(channels), 'the vector length is {}, but there are {} channels'.format(len(vector), len(channels))
    return {c_i.name: v_i for c_i, v_i in zip(channels, vector)}

def clip_signal(signal, channels):
    """Clip a signal to the bounds of the channels"""
    c_signal = {}
    for k, v in signal:
        if k in channels:
            c_signal[k] = min(max(signal[k], channels.bounds[0]),
                              channels.bounds[1])
        else:
            c_signal[k] = v
    return c_signal

def clip_vector(vector, channels):
    """Clip a signal to the bounds of the channels"""
    assert len(vector) == len(channels), 'the vector length is {}, but there are {} channels'.format(len(vector), len(channels))
    return tuple(min(max(v_i, c_i.bounds[0]),
                     c_i.bounds[1]) for v_i, c_i in zip(vector, channels))

def random_signal(channels, bounds=None):
    if bounds is None:
        return {c.name: c.fixed if c.fixed is not None else random.uniform(*c.bounds)
                for c in channels}
    else:
        return {c.name: c.fixed if c.fixed is not None else random.uniform(*b)
                for c, b in zip(channels, bounds)}
