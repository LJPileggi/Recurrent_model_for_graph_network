import sys
sys.path.append('../')

import tensorflow as tf
import sonnet as snt

from models.rec_graph_net import MultiTimeSeriesRNN, RecGraphNetwork
from models.simple_rnn import SimpleRNN
from tests.dummy_benchmark_generator import DummyTimeSeries, DummyBenchmarkGenerator


class RecModelsTest(tf.test.TestCase):
    """
    Base class for model unittests. Based on tensorflow unittests.
    """
    def setUp(self):
        super(RecModelsTest, self).setUp()
        tf.random.set_seed(1)

    def tearDown(self):
        pass

    def _get_model(self, *args):
        """
        Returns initialised model to test.
        """
        pass

    def _test_loaded_model_weights(self, model, weights):
        """
        Tests a model has been uploaded correctly by comparing
        weights.
        """
        for m_w, w in zip(model.trainable_weights(), weights):
            self.assertAllEqual(m_w, w)

    def test_loaded_model_weights(self):
        """
        Tests a model has been uploaded correctly; calls
        self._test_loaded_model_weights to check weights
        are the expected ones.
        """
        pass

    def _test_correct_shape(self, inputs, *args):
        """
        Tests model output has the correct shape.
        """
        model = self._get_model(*args)
        outputs = model(inputs)
        self.assertAllEqual(inputs.shape.as_list(),
                            outputs.shape.as_list())


class SimpleRNNTest(RecModelsTest):
    """
    Class for SimpleRNN unittests.
    """

    def _get_model(self, loaded_model):
        """
        Returns initialised SimpleRNN.
        """
        return SimpleRNN(saved_model=loaded_model)

    def test_loaded_model_weights(self):
        """
        Tests a model has been uploaded correctly; calls
        self._test_loaded_model_weights to check weights
        are the expected ones.
        """
        rec_model = snt.GRU(1)
        _ = rec_model(tf.ones((4, 1), dtype=tf.float32),
                        tf.zeros((4, 1), dtype=tf.float32))
        weights = [rec_model.input_to_hidden, rec_model.hidden_to_hidden]
        model = self._get_model(rec_model)
        _ = model(tf.ones((4, 1), dtype=tf.float32))
        self._test_loaded_model_weights(model, weights)

    def test_correct_shape(self):
        """
        Tests model output has the correct shape.
        """
        inputs = DummyTimeSeries()
        self._test_correct_shape(inputs(), None)


class MultiTimeSeriesRNNTest(RecModelsTest):
    """
    Class for MultiTimeSeriesRNN unittests.
    """

    def _get_model(self, input_dim, rec_model, enc_model):
        """
        Returns initialised MultiTimeSeriesRNN.
        """
        return MultiTimeSeriesRNN(input_dim, loaded_rec_model=rec_model,
                                                loaded_encoder=enc_model)

    def test_loaded_model_weights(self):
        """
        Tests a model has been uploaded correctly; calls
        self._test_loaded_model_weights to check weights
        are the expected ones.
        """
        rec_model, enc_model = snt.GRU(1), snt.nets.MLP([1, 1])
        _ = rec_model(enc_model(tf.ones((4, 1), dtype=tf.float32)),
                                tf.zeros((4, 1), dtype=tf.float32))
        weights = [rec_model.input_to_hidden, rec_model.hidden_to_hidden] + \
                                        list(enc_model.trainable_variables)
        model = self._get_model(1, rec_model, enc_model)
        _ = model(tf.ones((4, 1), dtype=tf.float32))
        self._test_loaded_model_weights(model, weights)

    def test_correct_shape(self):
        """
        Tests model output has the correct shape.
        """
        inputs = DummyTimeSeries()
        self._test_correct_shape(inputs(), 1, None, None)


if __name__ == "__main__":
    tf.test.main()
