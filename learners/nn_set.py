from __future__ import absolute_import, division

import numpy as np
import sklearn.neighbors

class NNSet(object):
    """Hold observations an provide nearest neighbors facilities"""

    def __init__(self):
        self.uuids = set()
        self.shape = None
        self._uuid_offset = 0

    def __len__(self):
        return len(self.uuids) + self._uuid_offset

    def add(self, x, y=None, uuid=None):
        if uuid is not None:
            if uuid in self.uuids:
                return
            self.uuids.add(uuid)
        else:
            self._uuid_offset += 1

        obs = [x] if y is None else [x, y]

        if self.shape is None:
            self.shape    = tuple(len(v_i) for v_i in obs)
            self._data    = [[] for _ in self.shape]
            self._nn_tree = [None  for _ in self.shape]
            self._nn_tree = [False for _ in self.shape]

        assert len(self.shape) == len(obs)
        assert all(len(v_i) == s_i for v_i, s_i in zip(obs, self.shape))
        assert all(len(d_i) == len(self._data[0]) for d_i in self._data)

        for i, obs_i in enumerate(obs):
            self._data[i].append(np.array(obs_i))
        self._nn_ready = [False]*len(self.shape)


    @property
    def xs(self):
        return self._data[0]

    @property
    def ys(self):
        return self._data[1]


    def nn(self, x, k=1):
        return self.nn_x(x, k=k)

    def nn_x(self, x, k=1):
        return self._nn(0, x, k=k)

    def nn_y(self, y, k=1):
        return self._nn(1, y, k=k)

    def _nn(self, side, v, k=1):
        """ Compute the k nearest neighbors of v in the observed data,
            :arg side:  if equal to _DATA_X, search among input data.
                    if equal to _DATA_Y, search among output data.
            :return:    distance and indexes of found nearest neighbors.
        """
        self._update_tree(side)
        dists, idxes = self._nn_tree[side].kneighbors(v, n_neighbors = k)
        return dists[0], idxes[0]

    def _update_tree(self, side):
        """Build the NNTree for the observed data"""
        if self.shape is None:
            raise IndexError('no data added to the structure')
        if not self._nn_ready[side]:
            self._nn_tree[side]  = sklearn.neighbors.NearestNeighbors(algorithm='auto')
            self._nn_tree[side].fit(self._data[side])
            self._nn_ready[side] = True
