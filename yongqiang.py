#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_directory = os.path.dirname(os.path.abspath(__file__))

print(16 * "++--")
print("current_directory:", current_directory)
print(16 * "++--")

value = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]

split0, split1, split2 = tf.split(value, 3, axis=0)
split3, split4, split5, split6 = tf.split(value, num_or_size_splits=4, axis=1)

with tf.Session() as sess:
    print("split0:\n", sess.run(split0))
    print('-' * 32)
    print("split1:\n", sess.run(split1))
    print('-' * 32)
    print("split2:\n", sess.run(split2))
    print('-' * 32)
    print("split3:\n", sess.run(split3))
    print('-' * 32)
    print("split4:\n", sess.run(split4))
    print('-' * 32)
    print("split5:\n", sess.run(split5))
    print('-' * 32)
    print("split6:\n", sess.run(split6))
