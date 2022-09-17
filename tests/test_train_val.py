import os
import sys
sys.path.append('../')
import unittest
import numpy as np
import tensorflow as tf
import sonnet as snt
import inspect

from config import Config
from tests.dummy_benchmark_generator import DummyTimeSeries, DummyBenchmarkGenerator
from utils.training_routine import training_routine_SimpleRNN, training_routine_RecGraphNet
from models.simple_rnn import *
from models.rec_graph_net import RecGraphNetwork
from models.graph_nets_deploy import *

CONFIG_SIMPLE_RNN = [20, 20, 1e-4, 1e-4, "n", 1]
CONFIG_REC_GRAPH_NET = [20, 20, 1e-4, 1e-4, "y", 2]

class TrainValTest(tf.test.TestCase):
    """
    Tensorflow based unittest for models' training and validation.
    """
    def setUp(self):
        super(TrainValTest, self).setUp()
        tf.random.set_seed(1)

    def tearDown(self):
        pass

    def _benchmark(self):
        benchmark = DummyBenchmarkGenerator()
        return benchmark()

    def _create_config_file(self, name, ep_n, ep_e, l_n, l_e, model, pool_dim):
        filename = f"{name}_temp.ini"
        content = f"[Epochs]\n" \
            f"ep_nodes = {ep_n}\n" \
            f"ep_edges = {ep_e}\n\n" \
            f"[Learning_rates]\n" \
            f"l_nodes = {l_n}\n" \
            f"l_edges = {l_e}\n\n" \
            f"[Model_options]\n" \
            f"graph_nets = {model}\n" \
            f"pool_dim = {pool_dim}"
        #_ = self.create_tempfile(os.path.join("config", filename))
        with open(os.path.join("config", filename), 'w') as f:
            f.write(content)
        return filename

    def _read_config(self, *args):
        config_file = self._create_config_file(*args)
        return Config(config_file), config_file

    def _model(self):
        pass

    def _training_routine(self):
        pass

    def test_training_routine(self, *args):
        pass

    def _test_training(self):
        pass

    def _test_validation(self, saved_model):
        pass

    def _test_train_val(self):
        pass

    def test_all(self):
        saved_model = self._test_training()
        self._test_validation(saved_model)
        self._test_train_val()


class SimpleRNNTrainValTest(TrainValTest):
    """
    Tensorflow based unittest for SimpleRNN's training and validation.
    """
    def _model(self):
        return SimpleRNN()

    def _training_routine(self, *args):
        return training_routine_SimpleRNN(*args)

    def test_training_routine(self, *args):
        data = DummyTimeSeries()
        inputs, target = data()[:,:-1], data()[:,1:]
        model = self._model()
        loss = tf.losses.MeanSquaredError()
        self._training_routine(model, loss, 1e-4, 10, inputs, target, "N/A")

    def _test_training(self):
        config, _ = self._read_config("rnn", *CONFIG_SIMPLE_RNN)
        input_tr, target_tr = self._benchmark()
        input_vl, target_vl = self._benchmark()
        save_model = "test_rnn"
        train_rnn(input_tr, target_tr, input_vl, target_vl,
                        config, save_model, load_model=None)
        return save_model

    def _test_validation(self, save_model):
        input_ts, target_ts = self._benchmark()
        validate_rnn(input_ts, target_ts, save_model, test=True)
        os.remove(os.path.join("saved_models", f"{save_model}_nodes.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_edges.obj"))

    def _test_train_val(self):
        config, config_file = self._read_config("rnn", *CONFIG_SIMPLE_RNN)
        input_tr, target_tr = self._benchmark()
        input_vl, target_vl = self._benchmark()
        input_ts, target_ts = self._benchmark()
        data = [input_tr, target_tr, input_vl, target_vl, input_ts, target_ts]
        save_model = "test_rnn"
        train_test_rnn(config, save_model,
                         *data, test=True)
        os.remove(os.path.join("saved_models", f"{save_model}_nodes.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_edges.obj"))
        os.remove(os.path.join("config", config_file))


class RecGraphNetTrainValTest(TrainValTest):
    """
    Tensorflow based unittest for RecGraphNetwork's training and validation.
    """
    def _model(self, pool_dim):
        return RecGraphNetwork(pool_dim,
                None, None, None, None)

    def _training_routine(self, *args):
        return training_routine_RecGraphNet(*args)

    def test_training_routine(self, *args):
        config, _ = self._read_config("rgn", *CONFIG_REC_GRAPH_NET)
        inputs, target = self._benchmark()
        model = self._model(config.pool_dim)
        loss = tf.losses.MeanSquaredError()
        self._training_routine(model, loss, config.l_rate_nodes, config.l_rate_edges,
                            config.epochs_nodes, config.epochs_edges, inputs, target)

    def _test_training(self):
        config, _ = self._read_config("rgn", *CONFIG_REC_GRAPH_NET)
        input_tr, target_tr = self._benchmark()
        input_vl, target_vl = self._benchmark()
        save_model = "test_rgn"
        train_graph_nets(input_tr, target_tr, input_vl, target_vl,
                            config, save_model, load_model=None)
        return save_model

    def _test_validation(self, save_model):
        config, _ = self._read_config("rgn", *CONFIG_REC_GRAPH_NET)
        input_ts, target_ts = self._benchmark()
        validate_graph_nets(input_ts, target_ts, config.pool_dim,
                                            save_model, test=True)
        os.remove(os.path.join("saved_models", f"{save_model}_nodes_rec.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_nodes_enc.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_edges_rec.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_edges_enc.obj"))

    def _test_train_val(self):
        config, config_file = self._read_config("rnn", *CONFIG_REC_GRAPH_NET)
        input_tr, target_tr = self._benchmark()
        input_vl, target_vl = self._benchmark()
        input_ts, target_ts = self._benchmark()
        data = [input_tr, target_tr, input_vl, target_vl, input_ts, target_ts]
        save_model = "test_model"
        train_test_graph_nets(config, save_model,
                         None, *data, test=True)
        os.remove(os.path.join("saved_models", f"{save_model}_nodes_rec.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_nodes_enc.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_edges_rec.obj"))
        os.remove(os.path.join("saved_models", f"{save_model}_edges_enc.obj"))
        os.remove(os.path.join("config", config_file))

