import os
import csv
import json
import time

import numpy as np
from numpyencoder import NumpyEncoder
import pycuda
from pycuda.compiler import SourceModule

from dicts import *
from matsum import matsum_gpu

def build_adjacency(country_dicts):
    """
    Builds list of dictionaries with node, edge, sender and
    receiver lists of yearly data to fit graph_nets framework.
    Yearly trades under 1,000,000 USD are disregarded. Figures
    from import and export are averaged through gpu matrix average.

    Args:
      - country_dicts: dictionary yielded by polish_raw_data.py;

    Returns:
      - data_dict_list: list of dictionaries compliant with graph_nets'
        framework.
    """
    tot_ctries = len(countries_names)
    adj_imp_raw = np.zeros((len(years), tot_ctries, tot_ctries), dtype=np.int64)
    adj_exp_raw = np.zeros((len(years), tot_ctries, tot_ctries), dtype=np.int64)
    for c_dict in country_dicts:
        for ctry, ctry_dict in c_dict.items():
            for i, year in enumerate(years):
                for part_imp, exch in ctry_dict[str(year)]["Import"].items():
                    adj_imp_raw[i][countries_names[ctry][1]][countries_names[part_imp][1]] = int(exch//1000) if exch > 1000000 else 0
                for part_exp, exch in ctry_dict[str(year)]["Export"].items():
                    adj_exp_raw[i][countries_names[ctry][1]][countries_names[part_exp][1]] = int(exch//1000) if exch > 1000000 else 0
    adj, signif = matsum_gpu(adj_imp_raw, adj_exp_raw)
    nodes = np.zeros((len(years), tot_ctries), dtype=float)
    for c_dict in country_dicts:
        for ctry, ctry_dict in c_dict.items():
            for i, year in enumerate(years):
                nodes[i][countries_names[ctry][1]] = ctry_dict[str(year)]["gdp"]
    edges = []
    senders = []
    receivers = []
    for y in range(len(years)):
        edges.append([])
        senders.append([])
        receivers.append([])
        for i in range(tot_ctries):
            for j in range(tot_ctries):
                if signif[i][j] >= 15:
                    edges[-1].append([adj[y][i][j]])
                    senders[-1].append(i)
                    receivers[-1].append(j)
    data_dict_list = [{
        "nodes" : nodes[y],
        "edges" : edges[y],
        "senders" : senders[y],
        "receivers": receivers[y]
        } for y in range(len(years))
        ]
    return data_dict_list

def main():
    """
    Creates apt data dictionary for graph_nets framework
    and dumps it in json file.
    """
    filename = os.path.join("..", "final_data", "data.json")
    with open(filename, 'r') as f:
        data = f.read()
    country_dicts = json.loads(data)
    start = time.time()
    data_dict_list = build_adjacency(country_dicts)
    print(f'time taken: {(time.time() - start):.1f} s')
    folder = os.path.join("..", "final_data")
    if not os.path.exists(folder):
        os.makedirs(folder)
    data_json = os.path.join(folder, "data_dict_list.json")
    with open(data_json, 'w') as f:
        json.dump(data_dict_list, f, cls=NumpyEncoder)

if __name__ == '__main__':
    main()
