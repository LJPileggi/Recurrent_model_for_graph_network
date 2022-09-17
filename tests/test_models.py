import sys
sys.path.append('../')
import unittest
import numpy as np
import tensorflow as tf
import sonnet as snt
import inspect

from models.rec_graph_net import MultiTimeSeriesRNN, RecGraphNetwork
from models.simple_rnn import SimpleRNN
from tests.dummy_benchmark_generator import DummyTimeSeries, DummyBenchmarkGenerator


class RecModelsTest(tf.test.TestCase):
    """
    Tensorflow based unittest for models used.
    """
    def setUp(self):
        super(RecModelsTest, self).setUp()
        tf.random.set_seed(1)

    def tearDown(self):
        pass

    def _get_model(self, *args):
        pass

    def _test_loaded_model_weights(self, model, weights):
        for m_w, w in zip(model.trainable_weights(), weights):
            self.assertAllEqual(m_w, w)

    def test_loaded_model_weights(self):
        pass

    def _test_correct_shape(self, inputs, *args):
        model = self._get_model(*args)
        outputs = model(inputs)
        self.assertAllEqual(inputs.shape.as_list(),
                            outputs.shape.as_list())


class SimpleRNNTest(RecModelsTest):

    def _get_model(self, loaded_model):
        return SimpleRNN(saved_model=loaded_model)

    def test_loaded_model_weights(self):
        rec_model = snt.GRU(1)
        _ = rec_model(tf.ones((4, 1), dtype=tf.float32),
                        tf.zeros((4, 1), dtype=tf.float32))
        weights = [rec_model.input_to_hidden, rec_model.hidden_to_hidden]
        model = self._get_model(rec_model)
        _ = model(tf.ones((4, 1), dtype=tf.float32))
        self._test_loaded_model_weights(model, weights)

    def test_correct_shape(self):
        inputs = DummyTimeSeries()
        self._test_correct_shape(inputs(), None)


class MultiTimeSeriesRNNTest(RecModelsTest):

    def _get_model(self, input_dim, rec_model, enc_model):
        return MultiTimeSeriesRNN(input_dim, loaded_rec_model=rec_model,
                                                loaded_encoder=enc_model)

    def test_loaded_model_weights(self):
        rec_model, enc_model = snt.GRU(1), snt.nets.MLP([1, 1])
        _ = rec_model(enc_model(tf.ones((4, 1), dtype=tf.float32)),
                                tf.zeros((4, 1), dtype=tf.float32))
        weights = [rec_model.input_to_hidden, rec_model.hidden_to_hidden] + \
                                        list(enc_model.trainable_variables)
        model = self._get_model(1, rec_model, enc_model)
        _ = model(tf.ones((4, 1), dtype=tf.float32))
        self._test_loaded_model_weights(model, weights)

    def test_correct_shape(self):
        inputs = DummyTimeSeries()
        self._test_correct_shape(inputs(), 1, None, None)


if __name__ == "__main__":
    tf.test.main()
