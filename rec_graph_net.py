import pickle

from graph_nets import utils_tf
from gn_patches.modules import GraphNetwork
import numpy as np
import tensorflow as tf
import sonnet as snt

"""
Options dictionaries for GraphNetwork blocks.
"""
edge_block_opt = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": False,
}

node_block_opt = {
    "use_received_edges": True,
    "use_sent_edges": True,
    "use_nodes": True,
    "use_globals": False,
}

global_block_opt = {
    "use_edges": False,
    "use_nodes": False,
    "use_globals": True,
}

class MultiTimeSeriesRNN(snt.Module):
    """
    Model for forecasting off of different time series. Made of
    MLP layer, to return output with correct shape, and GRU
    recurrent layer. Used both in Edge and Node Block of GraphNetwork.

    Args:
      - input_dim: dimension of dataset;
      - loaded_rec_model: name of previously saved model file; if none,
        brand new recurrent and readout layer are initialised.
    """
    def __init__(self, input_dim, loaded_rec_model=None,
            loaded_encoder=None, name="MultiTimeSeriesRNN"):
        super(MultiTimeSeriesRNN, self).__init__(name=name)
        tf.random.set_seed(1)
        if not loaded_rec_model:
            self.rec_model = snt.GRU(1)
        else:
            with open(loaded_rec_model, 'rb') as f:
                self.rec_model = pickle.load(f)
        if not loaded_encoder:
            self.encoder = snt.nets.MLP([1, 1])
        else:
            with open(loaded_encoder, 'rb') as f:
                self.encoder = pickle.load(f)
        self._input_dim = input_dim
        
    def trainable_weights(self):
        """
        Returns weights of recurrent and readout layer to use during model
        training.
        """
        return [
        self.rec_model.input_to_hidden,
        self.rec_model.hidden_to_hidden] + \
        list(self.encoder.trainable_variables)

    def __call__(self, inputs):
        """
        Forward step of model. Inputs are reshaped as (n_nodes/edges, n_t_steps,
        n_reduced_features), and then fed to the 2 layers iterating along the
        temporal steps' axis; finally, all outputs are normalised to make their
        sum equal to 1 within each time step.

        Args:
          - inputs: list of tf.Tensor of shape (n_nodes/edges, n_t_steps *
          n_reduced_features);

        Returns:
          - output: tf.Tensor of shape (n_nodes/edges, n_t_steps).
        """
        old_lists = tf.reshape(inputs, [inputs.shape.as_list()[0],
            self._input_dim, inputs.shape.as_list()[1]//self._input_dim])
        lists = [tf.stack(old, axis=-1) for old in old_lists]
        lists = tf.convert_to_tensor(lists, dtype=tf.float32)
        init_state = tf.zeros((lists.shape.as_list()[0], 1), tf.float32)
        output = []
        for i in range(lists.shape.as_list()[1]):
            hidden = self.encoder(lists[:,i,:])
            out, init_state = self.rec_model(hidden, init_state)
            output.append(out)
        output = tf.stack(tf.reshape(output, np.shape(output)[:-1]), axis=-1)
        tots = tf.math.reduce_sum(output, axis=0)
        output = tf.multiply(output, tf.pow(tots, -1))
        return output

def global_model_fn():
    """
    Dummy model for global variables; we are not using it in our model,
    so it is just for making the whole thing work.
    """

    def identity(x):
        return x

    return lambda x: identity(x)

reduced_sum = tf.math.unsorted_segment_sum

def reducer(pool_dim, feats, indices, num_items):
    """
    Reducer function to "pool" data coming from different GraphsTuple's features,
    in order to implement message passing. For each node/edge, it operates pool_dim
    reduced sums over the i indices such that i mod pool_dim = k for k in
    range(pool_dim); such pool_dim features will be used as additional features
    inside each block's model.

    Args:
      - pool_dim : dimension of reduced features vector;
      - feats: tensor of features to reduce;
      - indices: indices of sender/receiver nodes;
      - num_items: number of nodes/edges.

    Returns reduced tensor.
    """
    feats = tf.stack(feats, axis=-1)
    indices = tf.stack(indices, axis=-1)
    reduced_list = []
    for mod in range(pool_dim):
        reduced = []
        for feats_t, indices_t in zip(feats, indices):
            mask = tf.convert_to_tensor([1. if i%pool_dim == mod else 0.
                        for i, index in enumerate(indices_t)], dtype=tf.float32)
            reduced.append(reduced_sum(tf.multiply(feats_t, mask), indices_t, num_items))
        reduced_list.append(tf.stack(reduced, axis=-1))
    return tf.concat(reduced_list, axis=-1)

class RecGraphNetwork(snt.Module):
    """
    Graph module for analysing temporal graphs with RNN model.
    Deploys the MultiTimeSeriesRNN in EdgeBlock and NodeBlock.
    Essentially a GraphNetwork with MultiTimeSeriesRNN as a model.
    Global features are disregarded in this framework.
    """
    def __init__(self, reducer_pool_dim, loaded_model_nodes,
                   loaded_model_edges, name="RecGraphNetwork"):
        super(RecGraphNetwork, self).__init__(name=name)
        if (not loaded_model_nodes) & (not loaded_model_edges):
            edge_model_fn = lambda : MultiTimeSeriesRNN(3)
            node_model_fn = lambda : MultiTimeSeriesRNN(2*reducer_pool_dim+1)
        else:
            edge_model_fn = lambda : MultiTimeSeriesRNN(3,
                loaded_rec_model=loaded_model_edges+"_rec.obj",
                  loaded_encoder=loaded_model_edges+"_enc.obj")
            node_model_fn = lambda : MultiTimeSeriesRNN(2*reducer_pool_dim+1,
                                loaded_rec_model=loaded_model_nodes+"_rec.obj",
                                  loaded_encoder=loaded_model_nodes+"_enc.obj")
        self._network = GraphNetwork(
               edge_model_fn,
               node_model_fn,
               global_model_fn,
               reducer = lambda *args : reducer(reducer_pool_dim, *args),
               edge_block_opt=edge_block_opt,
               node_block_opt=node_block_opt,
               global_block_opt=global_block_opt,
               name="graph_network")

    @property
    def trainable_weights_nodes(self):
        """
        Returns trainable weights of node model for training.
        """
        return self._network.node_block.node_model.trainable_weights()

    @property
    def trainable_weights_edges(self):
        """
        Returns trainable weights of edge model for training.
        """
        return self._network.edge_block.edge_model.trainable_weights()

    @property
    def node_model(self):
        """
        Returns node model.
        """
        return self._network.node_block.node_model

    @property
    def edge_model(self):
        """
        Returns edge model.
        """
        return self._network.edge_block.edge_model

    def __call__(self, inputs):
        return self._network(inputs)
