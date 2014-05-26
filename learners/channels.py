from __future__ import absolute_import, division, print_function

class Channel(object):

    def __init__(self, name, bounds=(float('-inf'), float('+inf')), fixed=None):
        self.name = name
        self.bounds = bounds
        self.fixed = fixed

    def __repr__(self):
        return 'Channel({}, {})'.format(self.name, self.bounds)

    def __eq__(self, channel):
        return self.name == channel.name and self.bounds == channel.bounds

try:
    # We replace the Channels class by the one of the module environments
    # if present. Probably not a good idea.
    # TODO abstract class Channel
    import environments
    # TODO check version for compatibility
    Channels = environments.Channels
except (ImportError, AttributeError):
    pass
