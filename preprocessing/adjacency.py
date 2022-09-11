import os
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpyencoder import NumpyEncoder

from dicts import *

def build_adjacency(country_dicts):
    """
    Builds dictionary with node, edge, sender and receiver
    lists of yearly data to fit graph_nets framework. Yearly
    trades under 1,000,000 USD are disregarded. Figures from
    import and export are averaged through normal matrix average.

    Args:
      - country_dicts: dictionary yielded by polish_raw_data.py;

    Returns:
      - data_dict_list: list of dictionaries compliant with graph_nets'
        framework.
    """
    tot_ctries = len(countries_names)
    adj = np.zeros((len(years), tot_ctries, tot_ctries), dtype=np.float32)
    signif = np.zeros((tot_ctries, tot_ctries), dtype=np.int32)
    adj_imp_raw = np.zeros((len(years), tot_ctries, tot_ctries), dtype=np.int64)
    adj_exp_raw = np.zeros((len(years), tot_ctries, tot_ctries), dtype=np.int64)
    tot_exchange = np.zeros(len(years), dtype=np.float32)
    for c_dict in country_dicts:
        for ctry, ctry_dict in c_dict.items():
            for i, year in enumerate(years):
                for part_imp, exch in ctry_dict[str(year)]["Import"].items():
                    adj_imp_raw[i][countries_names[ctry][1]][countries_names[part_imp][1]] = exch//1000 if exch > 1000000 else 0
                    tot_exchange[i] += adj_imp_raw[i][countries_names[ctry][1]][countries_names[part_imp][1]]/2
                for part_exp, exch in ctry_dict[str(year)]["Export"].items():
                    adj_exp_raw[i][countries_names[ctry][1]][countries_names[part_exp][1]] = exch//1000 if exch > 1000000 else 0
                    tot_exchange[i] += adj_exp_raw[i][countries_names[ctry][1]][countries_names[part_exp][1]]/2
    for y in range(len(years)):
        for i in range(tot_ctries):
            for j in range(tot_ctries):
                adj[y][i][j] = float(int((adj_imp_raw[y][i][j] + adj_exp_raw[y][j][i])/2/tot_exchange[y])) \
                if not (adj_imp_raw[y][i][j] != 0 ^ adj_exp_raw[y][j][i] != 0) \
                else float(adj_imp_raw[y][i][j] + adj_exp_raw[y][j][i])/tot_exchange[y]
                signif[i][j] += 1 if adj[y][i][j] != 0 else 0
    nodes = np.zeros((len(years), tot_ctries), dtype=np.float32)
    tot_gdp = np.zeros(len(years), dtype=np.float32)
    for c_dict in country_dicts:
        for i, year in enumerate(years):
            for ctry, ctry_dict in c_dict.items():
                nodes[i][countries_names[ctry][1]] = float(ctry_dict[str(year)]["gdp"])
            tot_gdp[i] = nodes[i,:].sum()
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
                    edges[-1].append(adj[y][i][j])
                    senders[-1].append(i)
                    receivers[-1].append(j)
    data_dict_list = [{
        "nodes" : nodes[y]/tot_gdp[y],
        "edges" : edges[y],
        "senders" : senders[y],
        "receivers": receivers[y]
        } for y in range(len(years))#enumerate(years)
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
