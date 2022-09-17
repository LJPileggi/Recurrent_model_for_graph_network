import time

import tensorflow as tf

def training_routine_SimpleRNN(model, loss, l_rate, epochs, input_tr, target_tr, feat):
    """
    Training routine to update weights of SimpleRNN through gradient descent.

    Args:
      - model: model to train;
      - loss: type of loss function;
      - l_rate: learning rate;
      - epochs: n. of epochs of training;
      - input_tr: training dataset;
      - target_tr: target values for training;
      - feat: name of feature to train; either "nodes" or "edges".
    """
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

def training_routine_RecGraphNet(model, loss, l_rate_n, l_rate_e, epochs_n, epochs_e, input_tr, target_tr):
    """
    Training routine to update weights of RecGraphNetwork through gradient descent.

    Args:
      - model: model to train;
      - loss: type of loss function;
      - l_rate_n: learning rate for nodes;
      - l_rate_e: learning rate for edges;
      - epochs_n: n. of epochs of training for nodes;
      - epochs_e: n. of epochs of training for edges;
      - input_tr: training dataset;
      - target_tr: target values for training.
    """

    optimizer_nodes = tf.optimizers.Adam(learning_rate=l_rate_n)
    optimizer_edges = tf.optimizers.Adam(learning_rate=l_rate_e)

    for i in range(max((epochs_n, epochs_e))):
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

