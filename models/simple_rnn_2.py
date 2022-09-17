import sys
import os
import json
import pickle
import time

from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

from utils.plot_series import plot_series

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

    def training_routine(model, loss, l_rate, epochs, input_tr, target_tr, feat):
        optimizer = tf.optimizers.Adam(learning_rate=l_rate)
        for i in range(epochs):
            start = time.time()
            with tf.GradientTape() as tape:
                output_tr = model(input_tr)
                loss_tr = loss(target_tr, output_tr)
            params = model.trainable_weights()
            grads = tape.gradient(loss_tr, params)
            optimizer.apply_gradients(list(zip(grads, params)))
            end = time.time() - start
            if i%10 == 0:
                print(f"Epoch {i} :\n"
                  f"MSE {feat} : {loss_tr:.10f}\n"
                  f"time taken: {end:.2f} s"
                  )


    loss = tf.losses.MeanSquaredError()

    load_node_model = os.path.join("saved_models",load_model+"_nodes.obj") if load_model else None
    load_edge_model = os.path.join("saved_models", load_model+"_edges.obj") if load_model else None

    optimizer_nodes = tf.optimizers.Adam(learning_rate=config.l_rate_nodes)
    optimizer_edges = tf.optimizers.Adam(learning_rate=config.l_rate_edges)

    if not load_model:
        node_model, edge_model = SimpleRNN(), SimpleRNN()
    else:
        f1 = open(load_node_model+".obj", 'rb')
        f2 = open(load_edge_model+".obj", 'rb')
        node_model, edge_model = SimpleRNN(
            pickle.load(f1)), \
                                 SimpleRNN(
            pickle.load(f2))
        f1.close()
        f2.close()

    for i in range(config.epochs_nodes):
        start = time.time()
        with tf.GradientTape() as tape:
            output_tr = node_model(input_tr[0].nodes)
            loss_nodes_tr = loss(target_tr[0].nodes, output_tr)
        params_nodes = node_model.trainable_weights()
        grads_nodes = tape.gradient(loss_nodes_tr, params_nodes)
        optimizer_nodes.apply_gradients(list(zip(grads_nodes, params_nodes)))
        end = time.time() - start
        if i%10 == 0:
            print(f"Epoch {i} :\n"
              f"MSE nodes : {loss_nodes_tr:.10f}\n"
              f"time taken: {end:.2f} s"
              )

    for i in range(config.epochs_edges):
        start = time.time()
        with tf.GradientTape() as tape:
            output_tr = edge_model(input_tr[0].edges)
            loss_edges_tr = loss(target_tr[0].edges, output_tr)
        params_edges = edge_model.trainable_weights()
        grads_edges = tape.gradient(loss_edges_tr, params_edges)
        optimizer_edges.apply_gradients(list(zip(grads_edges, params_edges)))
        end = time.time() - start
        if i%10 == 0:
            print(f"Epoch {i} :\n"
              f"MSE edges : {loss_edges_tr:.10f}\n"
              f"time taken: {end:.2f} s"
              )

    output_vl_nodes = node_model(input_vl[0].nodes)
    output_vl_edges = edge_model(input_vl[0].edges)
    loss_nodes_vl = loss(target_vl[0].nodes, output_vl_nodes)
    loss_edges_vl = loss(target_vl[0].edges, output_vl_edges)

    print(f"MSE nodes : {loss_nodes_vl:.10f}\n"
          f"MSE edges : {loss_edges_vl:.10f}"
          )

    if not save_model_as == None:
        with open(os.path.join("saved_models",save_model_as+"_nodes.obj"), 'wb') as f:
            pickle.dump(node_model.model, f)
        with open(os.path.join("saved_models", save_model_as+"_edges.obj"), 'wb') as f:
            pickle.dump(edge_model.model, f)

    return save_model_as


def validate_rnn(input_ts, target_ts, load_model):
    """
    Evaluates performances of trained model on a test set. A plot of a few
    nodes' series are plotted at the end.

    Args:
      - input_ts: input temporal graph for testing; GraphsTuple object;
      - target_ts: target temporal graph for testing; GraphsTuple object;
      - pool_dim: dimension of pool operator output in graph model;
      - load_model: name of load trained model from previously trained file.
    
    Various GraphsTuple.nodes/edges arguments must have shape (n_nodes/n_edges, n_t_steps);
    """
    f1 = open(os.path.join("saved_models", load_model+"_nodes.obj"), 'rb')
    f2 = open(os.path.join("saved_models", load_model+"_edges.obj"), 'rb')
    node_model, edge_model = SimpleRNN(
        pickle.load(f1)), \
                             SimpleRNN(
        pickle.load(f2))
    f1.close()
    f2.close()

    loss = tf.losses.MeanSquaredError()
    node_normalizer = tf.keras.layers.BatchNormalization(axis=0)
    edge_normalizer = tf.keras.layers.BatchNormalization(axis=0)

    output_ts_nodes = node_model(input_ts[0].nodes)
    output_ts_edges = edge_model(input_ts[0].edges)
    loss_nodes_ts = loss(target_ts[0].nodes, output_ts_nodes)
    loss_edges_ts = loss(target_ts[0].edges, output_ts_edges)

    print(f"MSE nodes : {loss_nodes_ts:.10f}\n"
          f"MSE edges : {loss_edges_ts:.10f}"
          )

    for i in range(output_ts_nodes.shape[0]//10):
        indices = range(i*10, (i+1)*10)
        targets = [target_ts[0].nodes[index] for index in indices]
        outputs = [output_ts_nodes[index] for index in indices]
        plot_series(targets, outputs, indices, "simple", i)

def train_test_rnn(config, save_model_as,
                     load_model, *data):
    """
    Performs a training and then a validation on given training/test sets.

    Args:
      - config: Config object;
      - save_model_as: name of file to save trained model in;
      - load_model: name of already trained model to load before training;
      - data: list containing input_tr, target_tr, input_vl, target_vl,
        input_ts, target_ts, to go into respective functions.
    """

    validate_rnn(*data[-2:], train_rnn(*data[:-2],
                config, save_model_as, load_model))
