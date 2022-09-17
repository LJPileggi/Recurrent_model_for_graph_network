# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import _base
from . import blocks
import tensorflow as tf

_DEFAULT_EDGE_BLOCK_OPT = {
    "use_edges": True,
    "use_receiver_nodes": True,
    "use_sender_nodes": True,
    "use_globals": True,
}

_DEFAULT_NODE_BLOCK_OPT = {
    "use_received_edges": True,
    "use_sent_edges": False,
    "use_nodes": True,
    "use_globals": True,
}

_DEFAULT_GLOBAL_BLOCK_OPT = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}


def _make_default_edge_block_opt(edge_block_opt):
  """Default options to be used in the EdgeBlock of a generic GraphNetwork."""
  edge_block_opt = dict(edge_block_opt.items()) if edge_block_opt else {}
  for k, v in _DEFAULT_EDGE_BLOCK_OPT.items():
    edge_block_opt[k] = edge_block_opt.get(k, v)
  return edge_block_opt


def _make_default_node_block_opt(node_block_opt, default_reducer):
  """Default options to be used in the NodeBlock of a generic GraphNetwork."""
  node_block_opt = dict(node_block_opt.items()) if node_block_opt else {}
  for k, v in _DEFAULT_NODE_BLOCK_OPT.items():
    node_block_opt[k] = node_block_opt.get(k, v)
  for key in ["received_edges_reducer", "sent_edges_reducer"]:
    node_block_opt[key] = node_block_opt.get(key, default_reducer)
  return node_block_opt


def _make_default_global_block_opt(global_block_opt, default_reducer):
  """Default options to be used in the GlobalBlock of a generic GraphNetwork."""
  global_block_opt = dict(global_block_opt.items()) if global_block_opt else {}
  for k, v in _DEFAULT_GLOBAL_BLOCK_OPT.items():
    global_block_opt[k] = global_block_opt.get(k, v)
  for key in ["edges_reducer", "nodes_reducer"]:
    global_block_opt[key] = global_block_opt.get(key, default_reducer)
  return global_block_opt


class GraphNetwork(_base.AbstractModule):
  """Implementation of a Graph Network.

  See https://arxiv.org/abs/1806.01261 for more details.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               global_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               edge_block_opt=None,
               node_block_opt=None,
               global_block_opt=None,
               name="graph_network"):
    """Initializes the GraphNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to EdgeBlock to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see EdgeBlock for details).
      node_model_fn: A callable that will be passed to NodeBlock to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see NodeBlock for details).
      global_model_fn: A callable that will be passed to GlobalBlock to perform
        per-global computations. The callable must return a Sonnet module (or
        equivalent; see GlobalBlock for details).
      reducer: Reducer to be used by NodeBlock and GlobalBlock to aggregate
        nodes and edges. Defaults to tf.math.unsorted_segment_sum. This will be
        overridden by the reducers specified in `node_block_opt` and
        `global_block_opt`, if any.
      edge_block_opt: Additional options to be passed to the EdgeBlock. Can
        contain keys `use_edges`, `use_receiver_nodes`, `use_sender_nodes`,
        `use_globals`. By default, these are all True.
      node_block_opt: Additional options to be passed to the NodeBlock. Can
        contain the keys `use_received_edges`, `use_nodes`, `use_globals` (all
        set to True by default), `use_sent_edges` (defaults to False), and
        `received_edges_reducer`, `sent_edges_reducer` (default to `reducer`).
      global_block_opt: Additional options to be passed to the GlobalBlock. Can
        contain the keys `use_edges`, `use_nodes`, `use_globals` (all set to
        True by default), and `edges_reducer`, `nodes_reducer` (defaults to
        `reducer`).
      name: The module name.
    """
    super(GraphNetwork, self).__init__(name=name)
    edge_block_opt = _make_default_edge_block_opt(edge_block_opt)
    node_block_opt = _make_default_node_block_opt(node_block_opt, reducer)
    global_block_opt = _make_default_global_block_opt(global_block_opt, reducer)

    with self._enter_variable_scope():
      self._edge_block = blocks.EdgeBlock(
          edge_model_fn=edge_model_fn, **edge_block_opt)
      self._node_block = blocks.NodeBlock(
          node_model_fn=node_model_fn, **node_block_opt)
      self._global_block = blocks.GlobalBlock(
          global_model_fn=global_model_fn, **global_block_opt)

  @property
  def edge_block(self):
    return self._edge_block

  @property
  def node_block(self):
    return self._node_block

  @property
  def global_block(self):
    return self._global_block

  def _build(self, graph):
    """Connects the GraphNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. Depending on the block
        options, `graph` may contain `None` fields; but with the default
        configuration, no `None` field is allowed. Moreover, when using the
        default configuration, the features of each nodes, edges and globals of
        `graph` should be concatenable on the last dimension.

    Returns:
      An output `graphs.GraphsTuple` with updated edges, nodes and globals.
    """
    return self._global_block(self._node_block(self._edge_block(graph)))
