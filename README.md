# Recurrent model for graph network
Provides the graph_nets library with a framework to handle temporal graph data. It uses the GraphNetwork module with a recurrent model for the EdgeBlock and NodeBlock.

The newly constructed framework is then applied to the World Trade Network dataset, comprised of yearly trade data between countries in the 1997-2018 period, with countries being characterised by their annual GDP, to make 1-step-ahead predictions on the GDP trends, exploiting the feature of message-passing deriving from graph_nets.

The results of this framework are then compared to the ones obtained by an RNN applied to the GDP data only, without message-passing, similarly to what is done in the GraphIndependent module of graph_nets, to assess whether message-passing itself yields any improvement to the results.

For more information:
  - Peter W. Battaglia et al., Relational inductive biases, deep learning, and graph networks: https://arxiv.org/abs/1806.01261;
  - graph_nets library: https://github.com/deepmind/graph_nets;
  - World Trade Network data: https://comtrade.un.org/;

### Preprocessing
Raw data csv-s are contained in the ./data folder. It contains 190 folders named by the name of the corresponding country; each of them contains csv files referring to the commercial exchanges of a certain year, and a csv for the yearly GDP data. The polish_raw_data.py file in the ./preprocessing folder creates a first raw adjacency list dumped into the ./final_data/data.json file, and then the adjacency.py/adjacency_gpu.py files creates the final ./final_data/data_dict_list.json file containing data suited for the graph_nets modules.
