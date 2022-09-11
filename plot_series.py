import os
import numpy as np
import matplotlib.pyplot as plt

def plot_series(targets, outputs, indices, model, iter_i):
    """
    Plots a handful of series from test set, both the target and the predicted one.

    Args:
      - targets: list of a few nodes' target test series;
      - outputs_ list of a few nodes' predicted test series;
      - indices: indices by alphabetic order of the countries
        relative to the series, see ./preprocessing/dicts.py;
        to appear in legends;
      - model: name of model used; to appear in plot's file
    """
    colors = [np.random.uniform(0, 1, 3) for _ in range(len(indices))]
    xx = np.linspace(2018 - len(outputs[0]), 2018, len(outputs[0]))
    for target, output, index, color in zip(targets, outputs, indices, colors):
        plt.plot(xx, target, marker='.', linestyle='-', c=color,
                        label=f'{index}: target')
        plt.plot(xx, output, marker='x', linestyle='-', c=color,
                        label=f'{index}: output')
    plt.yscale('log')
    plt.legend(loc='right')
    folder = 'outputs'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, f'{model}_output_{iter_i}.png'))
    plt.clf()
