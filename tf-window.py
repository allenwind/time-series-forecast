import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()



window_size = 5
label_size = 1
rolling_size = window_size + label_size


def make_labels(x):
	return x[:-1], x[-1]


# tensorflow 实现的 滑动窗口

ts = tf.data.Dataset.range(30)
windows = ts.window(rolling_size, shift=1)
windows = windows.flat_map(lambda x: x.batch(rolling_size, drop_remainder=True))
windows = windows.map(make_labels)

for w, label in windows:
	print(w.numpy(), "=>", label.numpy())


# dot multilines
windows = tf.data.Dataset.range(30) \
		                 .window(rolling_size, shift=1) \
		                 .flat_map(lambda x: x.batch(rolling_size, drop_remainder=True)) \
		                 .map(make_labels)

for w, label in windows:
	print(w.numpy(), "=>", label.numpy())