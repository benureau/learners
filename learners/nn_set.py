# -*- coding: utf8 -*-
from __future__ import absolute_import, division

import numpy as np
import sklearn.neighbors


class BruteForceNNSet(object):
    """\
    Naïve implementation, as a API documentation,
    and to verify the correctness of the other implementations.
    """
    def __init__(self):
        self.uuids = set()
        self._size = 0
        self.shape = None

    def __len__(self):
        return self._size

    def _check_uuid(self, uuid):
        if uuid is not None:
            if uuid in self.uuids:
                return False
            self.uuids.add(uuid)
        return True

    def _check_obs(self, obs):
        assert len(self.shape) == len(obs)
        assert all(len(v_i) == s_i for v_i, s_i in zip(obs, self.shape))
        assert all(len(d_i) == len(self._data[0]) for d_i in self._data)

    def add(self, x, y=None, uuid=None):
        if not self._check_uuid(uuid):
            return
        self._size += 1

        obs = [x] if y is None else [x, y]
        if self.shape is None:
            self.shape    = tuple(len(v_i) for v_i in obs)
            self._data    = [[] for _ in self.shape]

        self._check_obs(obs)

        for i, obs_i in enumerate(obs):
            self._data[i].append(np.array(obs_i))

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
        assert len(v) == self.shape[side]
        v = np.array(v)
        data = self._data[side]

        results = []
        for i, u in enumerate(data):
            results.append((np.linalg.norm(u-v), i))
        results.sort()

        return (np.array([results[i][0] for i in range(k)]),
                   tuple([results[i][1] for i in range(k)]))

class BatchNNSet(BruteForceNNSet):
    """Hold observations an provide nearest neighbors facilities"""

    def add(self, x, y=None, uuid=None):
        if not self._check_uuid(uuid):
            return
        self._size += 1

        obs = [x] if y is None else [x, y]

        if self.shape is None:
            self.shape    = tuple(len(v_i) for v_i in obs)
            self._data    = [[] for _ in self.shape]
            self._nn_tree = [None for _ in self.shape]

        self._check_obs(obs)

        for i, obs_i in enumerate(obs):
            self._data[i].append(np.array(obs_i))
        self._nn_ready = [False]*len(self.shape)

    def _nn(self, side, v, k=1):
        """ Compute the k nearest neighbors of v in the observed data,
            :arg side:  if equal to _DATA_X, search among input data.
                        if equal to _DATA_Y, search among output data.
            :return:    distance and indexes of found nearest neighbors.
        """
        self._update_tree(side)
        dists, idxes = self._nn_tree[side].kneighbors(v, n_neighbors=k)
        return np.array(dists[0]), tuple(idxes[0])

    def _update_tree(self, side):
        """Build the NNSciConfig for the observed data"""
        if self.shape is None:
            raise IndexError('no data added to the structure')
        if not self._nn_ready[side]:
            self._nn_tree[side]  = sklearn.neighbors.NearestNeighbors(algorithm='auto')
            self._nn_tree[side].fit(self._data[side])
            self._nn_ready[side] = True


class NNSet(BatchNNSet):

    def __init__(self, poolsize=100):
        super(NNSet, self).__init__()
        self.poolsize = poolsize

    def add(self, x, y=None, uuid=None):
        if not self._check_uuid(uuid):
            return
        self._size += 1

        obs = [x] if y is None else [x, y]

        if self.shape is None:
            self.shape      = tuple(len(v_i) for v_i in obs)
            self._data       = [[]    for _ in self.shape]
            self._nn_tree    = [None  for _ in self.shape]
            self._nn_sizes   = [0     for _ in self.shape]
            self._pool_sizes = [0     for _ in self.shape]
            self._pool_tree  = [None  for _ in self.shape]
            self._pool_ready = [False for _ in self.shape]

        self._check_obs(obs)

        for i, obs_i in enumerate(obs):
            self._data[i].append(np.array(obs_i))
            self._pool_sizes[i] += 1
            self._pool_ready[i] = False

    def _nn(self, side, v, k=1):
        self._update_tree(side)
        k = min(k, len(self))

        if self._pool_sizes[side] == 0:
            t_dists, t_idxes = self._nn_tree[side].kneighbors(v, n_neighbors=k)
            return np.array(t_dists[0]), tuple(t_idxes[0])
        elif len(self._data[side]) <= self.poolsize:
            offset = len(self._data[side]) - self._pool_sizes[side]
            p_dists, p_idxes = self._pool_tree[side].kneighbors(v, n_neighbors=k)
            return np.array(p_dists[0]), tuple(p_i + offset for p_i in p_idxes[0])
        else:
            t_dists, t_idxes = self._nn_tree[side].kneighbors(v, n_neighbors=min(k, self._nn_sizes[side]))
            t_dists, t_idxes = t_dists[0], t_idxes[0]
            p_dists, p_idxes = self._pool_tree[side].kneighbors(v, n_neighbors=min(k, self._pool_sizes[side]))
            p_dists, p_idxes = p_dists[0], p_idxes[0]
            # merge results
            offset = len(self._data[side]) - self._pool_sizes[side]
            dists, idxes = [], []
            t_i, p_i = 0, 0
            for _ in range(k):
                if p_i >= len(p_dists):
                    dists.append(t_dists[t_i])
                    idxes.append(t_idxes[t_i])
                    t_i += 1
                elif t_i >= len(t_dists):
                    dists.append(p_dists[p_i])
                    idxes.append(p_idxes[p_i] + offset)
                    p_i += 1
                elif p_dists[p_i] <= t_dists[t_i]:
                    dists.append(p_dists[p_i])
                    idxes.append(p_idxes[p_i] + offset)
                    p_i += 1
                else:
                    dists.append(t_dists[t_i])
                    idxes.append(t_idxes[t_i])
                    t_i += 1
            return np.array(dists), tuple(idxes)

    def _update_tree(self, side):
        if self._pool_sizes[side] >= self.poolsize:
            self._nn_tree[side]   = sklearn.neighbors.NearestNeighbors(algorithm='auto')
            self._nn_tree[side].fit(self._data[side])
            self._nn_sizes[side] = len(self._data[side])
            self._pool_sizes[side] = 0
            self._pool_ready[side] = True
        else:
            if not self._pool_ready[side]:
                data = self._data[side][-self._pool_sizes[side]:]
                self._pool_tree[side]  = sklearn.neighbors.NearestNeighbors(algorithm='brute')
                self._pool_tree[side].fit(data)
                self._pool_ready[side] = True
