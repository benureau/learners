"""A collections of """

def uniformize_signal(signal, channels):
    """ Uniformize a signal between 0.0 and 1.0
        Requires every channel to have bounds, or be discretized.
    """
    uni_signal = []
    for c in channels:
        factor = 1.0
        if c.bounds[0] != c.bounds[1]:
            assert c.bounds[0] < c.bounds[1]
            factor = c.bounds[1] - c.bounds[0]
        uni_signal.append((signal[c.name]-c.bounds[0])/factor)
    return uni_signal

def restore_signal(uni_signal, channels):
    """ Uniformize a signal between 0.0 and 1.0
        Requires every channel to have bounds, or be discretized.
    """
    signal = collections.OrderedDict()
    for c in channels:
        factor = 1.0
        if c.bounds[0] != c.bounds[1]:
            assert c.bounds[0] < c.bounds[1]
            factor = c.bounds[1] - c.bounds[0]
        signal[c.name] = uni_signal[c.name]*factor + c.bounds[0]
    return signal

def _load_class(classname):
    """Load a class from a string"""
    module_name, class_name = classname.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
