import os
import pickle

import numpy as np
import sonnet as snt
import tensorflow as tf

from utils.plot_series import plot_series
from utils.training_routine import training_routine_SimpleRNN

class SimpleRNN(snt.Module):
    """
    RNN container to aptly reshape input and output for prediction.
    Uses GRU as recurrent model.

    Args:
      - saved_model: upload already trained model from file.
    """
    def __init__(self, saved_model=None, name="SimpleRNN"):
        super(SimpleRNN, self).__init__(name=name)
        tf.random.set_seed(1)
        self.model = snt.GRU(1) if not saved_model \
                else saved_model

    def trainable_weights(self):
        """
        Returns model trainable weights for training.
        """
        return [self.model.input_to_hidden,
                self.model.hidden_to_hidden]

    def __call__(self, inputs):
        """
        Reshapes inputs to (n_nodes/edges, n_t_steps, 1),
        applies GRU and finally normalises outputs so that
        their sum equals 1 within each timestep.

        Args:
          - inputs: list of tf.Tensor of shape (n_nodes/edges, n_t_steps);

        Returns:
          - output: tf.Tensor of shape (n_nodes/edges, n_t_steps).
        """
        inputs = tf.reshape(inputs, [inputs.shape.as_list(
                        )[0], inputs.shape.as_list()[1], 1])
        init_state = tf.zeros((inputs.shape.as_list(
                )[0], inputs.shape.as_list()[2]), tf.float32)
        output = []
        for i in range(inputs.shape.as_list()[1]):
            out, init_state = self.model(inputs[:,i,:], init_state)
            output.append(out)
        output = tf.stack(tf.reshape(output,
                np.shape(output)[:-1]), axis=-1)
        tots = tf.math.reduce_sum(output, axis=0)
        output = tf.multiply(output, tf.pow(tots, -1))
        return output


def train_rnn(input_tr, target_tr, input_vl, target_vl,
            config, save_model_as, load_model=None):
    """
    Trains SimpleRNN with input data. Can optionally load trained
    model from file. Optimiser used: Adam. Loss function: mean squared error.

    Args:
      - input_tr: input temporal graph for training; GraphsTuple object;
      - target_tr: target temporal graph for training; GraphsTuple object;
      - input_vl: input temporal graph for validation; GraphsTuple object;
      - target_vl: target temporal graph for validation; GraphsTuple object;
      - config: Config object from config file;
      - save_model_as: name of model files to save trained models in. If none
        is provided, no model is saved;
      - load_model: name ofload trained model from previously trained file. If
        none is provided, no model is loaded.

    Various GraphsTuple.nodes/edges arguments must have shape (n_nodes/n_edges, n_t_steps);

    Returns:
      - save_model_as: same as homonymous arg. Goes as input in validate_rnn
        function in train_test_rnn func.
    """

    loss = tf.losses.MeanSquaredError()

    load_node_model = os.path.join("saved_models", f"{load_model}_nodes.obj") if load_model else None
    load_edge_model = os.path.join("saved_models", f"{load_model}_edges.obj") if load_model else None

    if not load_model:
        node_model, edge_model = SimpleRNN(), SimpleRNN()
    else:
        with open(f"{load_node_model}.obj", 'rb') as f_n:
            node_model = SimpleRNN(pickle.load(f_n))
        with open(f"{load_edge_model}.obj", 'rb') as f_e:
            edge_model = SimpleRNN(pickle.load(f_e))

    training_routine_SimpleRNN(node_model, loss, config.l_rate_nodes, config.epochs_nodes,
                                            input_tr[0].nodes, target_tr[0].nodes, "nodes")
    training_routine_SimpleRNN(edge_model, loss, config.l_rate_edges, config.epochs_edges,
                                            input_tr[0].edges, target_tr[0].edges, "edges")

    output_vl_nodes = node_model(input_vl[0].nodes)
    output_vl_edges = edge_model(input_vl[0].edges)
    loss_nodes_vl = loss(target_vl[0].nodes, output_vl_nodes)
    loss_edges_vl = loss(target_vl[0].edges, output_vl_edges)

    print(f"MSE nodes : {loss_nodes_vl:.10f}\n"
          f"MSE edges : {loss_edges_vl:.10f}"
          )

    if not save_model_as == None:
        with open(os.path.join("saved_models", f"{save_model_as}_nodes.obj"), 'wb') as f:
            pickle.dump(node_model.model, f)
        with open(os.path.join("saved_models", f"{save_model_as}_edges.obj"), 'wb') as f:
            pickle.dump(edge_model.model, f)

    return save_model_as


def validate_rnn(input_ts, target_ts, load_model, test=False):
    """
    Evaluates performances of trained model on a test set. Plots of the various
    nodes' series are plotted at the end.

    Args:
      - input_ts: input temporal graph for testing; GraphsTuple object;
      - target_ts: target temporal graph for testing; GraphsTuple object;
      - pool_dim: dimension of pool operator output in graph model;
      - load_model: name of load trained model from previously trained file;
      - test: default to False, deactivates the creation of plots if in testing
        mode (True).

    Various GraphsTuple.nodes/edges arguments must have shape (n_nodes/n_edges, n_t_steps);
    """
    with open(os.path.join("saved_models", f"{load_model}_nodes.obj"), 'rb') as f_n:
        node_model = SimpleRNN(pickle.load(f_n))
    with open(os.path.join("saved_models", f"{load_model}_edges.obj"), 'rb') as f_e:
        edge_model = SimpleRNN(pickle.load(f_e))

    loss = tf.losses.MeanSquaredError()

    output_ts_nodes = node_model(input_ts[0].nodes)
    output_ts_edges = edge_model(input_ts[0].edges)
    loss_nodes_ts = loss(target_ts[0].nodes, output_ts_nodes)
    loss_edges_ts = loss(target_ts[0].edges, output_ts_edges)

    print(f"MSE nodes : {loss_nodes_ts:.10f}\n"
          f"MSE edges : {loss_edges_ts:.10f}"
          )

    if not test:
        for i in range(output_ts_nodes.shape[0]//10):
            indices = range(i*10, (i+1)*10)
            targets = [target_ts[0].nodes[index] for index in indices]
            outputs = [output_ts_nodes[index] for index in indices]
            plot_series(targets, outputs, indices, "simple", i)

def train_test_rnn(config, save_model_as,
                     *data, load_model=None, test=False):
    """
    Performs a training and then a validation on given training/test sets.

    Args:
      - config: Config object;
      - save_model_as: name of file to save trained model in;
      - load_model: name of already trained model to load before training;
      - data: list containing input_tr, target_tr, input_vl, target_vl,
        input_ts, target_ts, to go into respective functions;
      - test: default to False, deactivates the creation of plots if in testing
        mode (True).
    """

    validate_rnn(*data[-2:], train_rnn(*data[:-2],
                config, save_model_as, load_model), test=test)
