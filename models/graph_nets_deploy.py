from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import pickle
import time

from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

from .rec_graph_net import RecGraphNetwork
from utils.plot_series import plot_series

"""
Container for training and validation functions for RecGraphNetwork.

"""

def train_graph_nets(input_tr, target_tr, input_vl, target_vl,
                     config, save_model_as, load_model=None):
    """
    Trains RecGraphNetwork's edge and node blocks' models with input data.
    Can optionally load trained model from file. Optimiser used: Adam.
    Loss function: mean squared error.

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
      - save_model_as: same as homonymous arg. Goes as input in validate_graph_nets
        function in train_test_graph_nets func.
    """

    load_model_nodes = os.path.join("saved_models",load_model+"_nodes") if load_model else None
    load_model_edges = os.path.join("saved_models", load_model+"_edges") if load_model else None

    model = RecGraphNetwork(config.pool_dim, load_model_nodes,
                                    load_model_edges)

    loss = tf.losses.MeanSquaredError()

    optimizer_nodes = tf.optimizers.Adam(learning_rate=config.l_rate_nodes)
    optimizer_edges = tf.optimizers.Adam(learning_rate=config.l_rate_edges)

    for i in range(max((config.epochs_nodes, config.epochs_edges))):
        start = time.time()
        with tf.GradientTape() as tape:
            output_tr = model(input_tr)
            loss_nodes_tr = loss(target_tr[0].nodes, output_tr.nodes)
        params_nodes = model.trainable_weights_nodes
        grads_nodes = tape.gradient(loss_nodes_tr, params_nodes)
        optimizer_nodes.apply_gradients(list(zip(grads_nodes, params_nodes)))

        with tf.GradientTape() as tape:
            output_tr = model(input_tr)
            loss_edges_tr = loss(target_tr[0].edges, output_tr.edges)
        params_edges = model.trainable_weights_edges
        grads_edges = tape.gradient(loss_edges_tr, params_edges)
        optimizer_edges.apply_gradients(list(zip(grads_edges, params_edges)))

        end = time.time() - start
        print(f"Epoch {i} :\n"
              f"MSE nodes : {loss_nodes_tr:.10f}\n"
              f"MSE edges : {loss_edges_tr:.10f}\n"
              f"time taken: {end:.0f} s"
              )

    output_vl = model(input_vl)
    loss_nodes_vl = loss(target_vl[0].nodes, output_vl.nodes)
    loss_edges_vl = loss(target_vl[0].edges, output_vl.edges)

    print(f"MSE nodes : {loss_nodes_vl:.10f}\n"
          f"MSE edges : {loss_edges_vl:.10f}"
          )

    if not save_model_as == None:
        f1 = open(os.path.join("saved_models", save_model_as+"_nodes_rec.obj"), 'wb')
        f2 = open(os.path.join("saved_models", save_model_as+"_nodes_enc.obj"), 'wb')
        f3 = open(os.path.join("saved_models", save_model_as+"_edges_rec.obj"), 'wb')
        f4 = open(os.path.join("saved_models", save_model_as+"_edges_enc.obj"), 'wb')
        pickle.dump(model.node_model.rec_model, f1)
        pickle.dump(model.node_model.encoder, f2)
        pickle.dump(model.edge_model.rec_model, f3)
        pickle.dump(model.edge_model.encoder, f4)
        f1.close()
        f2.close()
        f3.close()
        f4.close()

    return save_model_as


def validate_graph_nets(input_ts, target_ts, pool_dim,
                    load_model):
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

    model = RecGraphNetwork(pool_dim, os.path.join("saved_models",
        load_model+"_nodes"), os.path.join("saved_models", load_model+"_edges"))

    loss = tf.losses.MeanSquaredError()

    output_ts = model(input_ts)
    loss_nodes_ts = loss(target_ts[0].nodes, output_ts.nodes)
    loss_edges_ts = loss(target_ts[0].edges, output_ts.edges)

    print(f"MSE nodes : {loss_nodes_ts:.10f}\n"
          f"MSE edges : {loss_edges_ts:.10f}"
          )

    for i in range(output_ts.nodes.shape[0]//10):
        indices = range(i*10, (i+1)*10)
        targets = [target_ts[0].nodes[index] for index in indices]
        outputs = [output_ts.nodes[index] for index in indices]
        plot_series(targets, outputs, indices, "multi", i)

def train_test_graph_nets(config,save_model_as,
                            load_model, *data):
    """
    Performs a training and then a validation on given training/test sets.
    Deprecated.

    Args:
      - config: Config object;
      - save_model_as: name of file to save trained model in;
      - load_model: name of already trained model to load before training;
      - data: list containing input_tr, target_tr, input_vl, target_vl,
        input_ts, target_ts, to go into respective functions.
    """

    print('Deprecated method. Raises error when calling validate_graph_nets.'
          ' Execute train_graph_nets and validate_graph_nets separately '
          'instead.')
    validate_graph_nets(*data[-2:], config.pool_dim, train_graph_nets(
                        *data[:-2], config, save_model_as, load_model))
