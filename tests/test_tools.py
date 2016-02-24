from __future__ import absolute_import, division, print_function
import unittest
import random

import dotdot
from learners import tools, Channel


random.seed(0)

class TestTools(unittest.TestCase):

    def test_uniformize(self):
        ch = [Channel('x', (-1, 1))]

        for _ in range(100):
            signal = tools.random_signal(ch)
            uni_signal = tools.uniformize_signal(signal, ch)
            restored   = tools.restore_signal(uni_signal, ch)
            self.assertEqual(signal, restored)


if __name__ == '__main__':
    unittest.main()
