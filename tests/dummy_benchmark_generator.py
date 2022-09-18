import numpy as np
import tensorflow as tf

from gn_patches.utils_tf import data_dicts_to_graphs_tuple

SERIES_A = tf.convert_to_tensor([[1., 1.5, 1.2, 1.1, 1.]], dtype=tf.float32)
SERIES_B = tf.convert_to_tensor([[0.5, 0.3, 0.2, 0.4, 0.3]], dtype=tf.float32)
SERIES_C = tf.convert_to_tensor([[0.7, 0.8, 1., 0.9, 1.]], dtype=tf.float32)
SERIES_D = tf.convert_to_tensor([[1.2, 0.7, 0.6, 0.9, 1.]], dtype=tf.float32)


class DummyTimeSeries():
    """
    Generates a batch of time series to use in tests for SimpleRNN.
    """

    def __init__(self):
        self._time_series = tf.concat([SERIES_A, SERIES_B, SERIES_C, SERIES_D], 0)

    def __call__(self):
        return self._time_series + tf.random.truncated_normal([4, 5], stddev=0.1)

class DummyBenchmarkGenerator():
    """
    Generates 2 small GraphsTuple samples (one as input, one as target)
    to use as a benchmark during tests.
    """

    def __init__(self):
        senders = tf.reshape([0, 0, 0, 1], [4,1])
        receivers = tf.reshape([1, 2, 3, 3], [4,1])
        temp_graph = [{
            "nodes" : tf.concat([SERIES_A, SERIES_B, (SERIES_C+SERIES_D)/2, 2*SERIES_B], 0) + \
                                                tf.random.truncated_normal([4, 5], stddev=0.1),
            "edges" : tf.concat([(SERIES_A+SERIES_B)/10, (SERIES_A+SERIES_C)/10, SERIES_B/5, \
                                SERIES_B/5], 0) + tf.random.truncated_normal([4, 5], stddev=0.02),
            "senders" : tf.repeat(senders, [5], axis=1),
            "receivers" : tf.repeat(receivers, [5], axis=1),
            "globals" : tf.convert_to_tensor([0., 0., 0., 0., 0.], dtype=tf.float32)
            }]
        slided_temp_graph = [{
            "nodes" : tf.roll(temp_graph[0]["nodes"], shift=-1, axis=1),
            "edges" : tf.roll(temp_graph[0]["edges"], shift=-1, axis=1),
            "senders" : temp_graph[0]["senders"],
            "receivers" : temp_graph[0]["receivers"],
            "globals" : temp_graph[0]["globals"]
            }]
        self._temp_graph = data_dicts_to_graphs_tuple(temp_graph)
        self._target_temp_graph = data_dicts_to_graphs_tuple(slided_temp_graph)

    def __call__(self):
        return [self._temp_graph], [self._target_temp_graph]
