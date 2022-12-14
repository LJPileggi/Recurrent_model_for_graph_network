import os
import pickle

import tensorflow as tf

from .rec_graph_net import RecGraphNetwork
from utils.plot_series import plot_series
from utils.training_routine import training_routine_RecGraphNet

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

    if load_model:
        with open(os.path.join("saved_models", f"{load_model}_nodes_rec.obj"
                                                        ), 'rb') as f_n_rec:
            loaded_nodes_rec = pickle.load(f_n_rec)
        with open(os.path.join("saved_models", f"{load_model}_nodes_enc.obj"
                                                        ), 'rb') as f_n_enc:
            loaded_nodes_enc = pickle.load(f_n_enc)
        with open(os.path.join("saved_models", f"{load_model}_edges_rec.obj"
                                                        ), 'rb') as f_e_rec:
            loaded_edges_rec = pickle.load(f_e_rec)
        with open(os.path.join("saved_models", f"{load_model}_edges_enc.obj"
                                                        ), 'rb') as f_e_enc:
            loaded_edges_enc = pickle.load(f_e_enc)
    else:
        loaded_nodes_rec, loaded_nodes_enc, loaded_edges_rec, loaded_edges_enc = \
                                                            None, None, None, None

    model = RecGraphNetwork(config.pool_dim, loaded_nodes_rec, loaded_nodes_enc,
                                            loaded_edges_rec, loaded_edges_enc)

    loss = tf.losses.MeanSquaredError()

    training_routine_RecGraphNet(model, loss, config.l_rate_nodes, config.l_rate_edges,
                        config.epochs_nodes, config.epochs_edges, input_tr, target_tr)
    output_vl = model(input_vl)
    loss_nodes_vl = loss(target_vl[0].nodes, output_vl.nodes)
    loss_edges_vl = loss(target_vl[0].edges, output_vl.edges)

    print(f"MSE nodes : {loss_nodes_vl:.10f}\n"
          f"MSE edges : {loss_edges_vl:.10f}"
          )

    if not save_model_as == None:
        with open(os.path.join("saved_models", f"{save_model_as}_nodes_rec.obj"), 'wb') as f_n_rec:
            pickle.dump(model.node_model.rec_model, f_n_rec)
        with open(os.path.join("saved_models", f"{save_model_as}_nodes_enc.obj"), 'wb') as f_n_enc:
            pickle.dump(model.node_model.encoder, f_n_enc)
        with open(os.path.join("saved_models", f"{save_model_as}_edges_rec.obj"), 'wb') as f_e_rec:
            pickle.dump(model.edge_model.rec_model, f_e_rec)
        with open(os.path.join("saved_models", f"{save_model_as}_edges_enc.obj"), 'wb') as f_e_enc:
            pickle.dump(model.edge_model.encoder, f_e_enc)

    return save_model_as


def validate_graph_nets(input_ts, target_ts, pool_dim,
                                load_model, test=False):
    """
    Evaluates performances of trained model on a test set. A plot of a few
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

    with open(os.path.join("saved_models", load_model+"_nodes"+"_rec.obj"), 'rb') as f_n_rec:
        loaded_nodes_rec = pickle.load(f_n_rec)
    with open(os.path.join("saved_models", load_model+"_nodes"+"_enc.obj"), 'rb') as f_n_enc:
        loaded_nodes_enc = pickle.load(f_n_enc)
    with open(os.path.join("saved_models", load_model+"_edges"+"_rec.obj"), 'rb') as f_e_rec:
        loaded_edges_rec = pickle.load(f_e_rec)
    with open(os.path.join("saved_models", load_model+"_edges"+"_enc.obj"), 'rb') as f_e_enc:
        loaded_edges_enc = pickle.load(f_e_enc)

    model = RecGraphNetwork(pool_dim, loaded_nodes_rec, loaded_nodes_enc,
                                    loaded_edges_rec, loaded_edges_enc)

    loss = tf.losses.MeanSquaredError()

    output_ts = model(input_ts)
    loss_nodes_ts = loss(target_ts[0].nodes, output_ts.nodes)
    loss_edges_ts = loss(target_ts[0].edges, output_ts.edges)

    print(f"MSE nodes : {loss_nodes_ts:.10f}\n"
          f"MSE edges : {loss_edges_ts:.10f}"
          )

    if not test:
        for i in range(output_ts.nodes.shape[0]//10):
            indices = range(i*10, (i+1)*10)
            targets = [target_ts[0].nodes[index] for index in indices]
            outputs = [output_ts.nodes[index] for index in indices]
            plot_series(targets, outputs, indices, "multi", i)

def train_test_graph_nets(config, save_model_as,
                load_model, *data, test=False):
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

    validate_graph_nets(*data[-2:], config.pool_dim, train_graph_nets(
            *data[:-2], config, save_model_as, load_model), test=test)
