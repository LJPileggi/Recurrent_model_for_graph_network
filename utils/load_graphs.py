import os
import json

from graph_nets.utils_tf import make_runnable_in_session
import tensorflow as tf

from gn_patches.utils_tf import data_dicts_to_graphs_tuple

"""
Container for functions to read graph from data file and convert it into GraphsTuple
format.
"""

def make_all_runnable_in_session(*args):
    """
    Apply make_runnable_in_session to an iterable of graphs.
    """
    return [make_runnable_in_session(a) for a in args]

def create_temp_graph(nodes_l, edges_l, senders_l, receivers_l, globals_l):
    """
    Takes lists of shape (n_t_steps, n_nodes/edges/senders/receivers/globals),
    reshapes them into (n_nodes/edges/senders/receivers/globals, n_t_steps) and
    creates GraphsTuple object off of them.

    Args:
      - nodes/edges/senders/receivers/globals_l: list of shape
        (n_t_steps, n_nodes/edges/senders/receivers/globals) read
        from data file.

    Returns:
      - temp_graph: GraphsTuple object.
    """
    nodes_l = tf.stack(nodes_l, axis=-1)
    edges_l = tf.stack(edges_l, axis=-1)
    senders_l = tf.stack(senders_l, axis=-1)
    receivers_l = tf.stack(receivers_l, axis=-1)
    globals_l = tf.stack(globals_l, axis=-1)
    temp_graph = [{
        "nodes" : nodes_l,
        "edges" : edges_l,
        "senders" : senders_l,
        "receivers" : receivers_l,
        "globals" : globals_l
        }]
    temp_graph = data_dicts_to_graphs_tuple(temp_graph)
    return temp_graph

def graph_loader(graph_file, task):
    """
    Loads graph data from file, converts it into GraphsTuple, makes it
    runnable in session and returns sets required for given task.

    Args:
      - graph_file: name of graph data file;
      - task: name of task at hand; must be 'tr', 'vl' or 'ts'.

    Returns:
      - input_graph-s and target_graphs-s relative to tasks (tr and vl for
        training, ts for validation, all six of them for both).
    """
    filename = os.path.join("final_data", graph_file)
    with open(filename, 'r') as f:
        data_f = f.read()
    data = json.loads(data_f)

    nodes_l = [graph['nodes'] for graph in data]
    edges_l = [graph['edges'] for graph in data]
    senders_l = [graph['senders'] for graph in data]
    receivers_l = [graph['receivers'] for graph in data]
    globals_l = [0. for graph in data]

    input_graph_tr = create_temp_graph(nodes_l[:10], edges_l[:10],
                  senders_l[:10], receivers_l[:10], globals_l[:10])
    target_graph_tr = create_temp_graph(nodes_l[1:11], edges_l[1:11],
                  senders_l[1:11], receivers_l[1:11], globals_l[1:11])
    input_graph_vl = create_temp_graph(nodes_l[10:15], edges_l[10:15],
                  senders_l[10:15], receivers_l[10:15], globals_l[10:15])
    target_graph_vl = create_temp_graph(nodes_l[11:16], edges_l[11:16],
                  senders_l[11:16], receivers_l[11:16], globals_l[11:16])
    input_graph_ts = create_temp_graph(nodes_l[15:-1], edges_l[15:-1],
                  senders_l[15:-1], receivers_l[15:-1], globals_l[15:-1])
    target_graph_ts = create_temp_graph(nodes_l[16:], edges_l[16:],
                  senders_l[16:], receivers_l[16:], globals_l[16:])

    input_graph_tr = make_all_runnable_in_session(input_graph_tr)
    target_graph_tr = make_all_runnable_in_session(target_graph_tr)
    input_graph_vl = make_all_runnable_in_session(input_graph_vl)
    target_graph_vl = make_all_runnable_in_session(target_graph_vl)
    input_graph_ts = make_all_runnable_in_session(input_graph_ts)
    target_graph_ts = make_all_runnable_in_session(target_graph_ts)

    if task == 'tr':
        return (input_graph_tr, target_graph_tr,
                input_graph_vl, target_graph_vl)
    if task == 'ts':
        return input_graph_ts, target_graph_ts
    if task == 'trts':
        return (input_graph_tr, target_graph_tr,
                input_graph_vl, target_graph_vl,
                input_graph_ts, target_graph_ts)
    raise ValueError(f"ValueError: invalid argument for task: {task}")
