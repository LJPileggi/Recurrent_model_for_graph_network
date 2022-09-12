# Recurrent model for graph network
Provides the graph_nets library with a framework to handle temporal graph data. It uses the GraphNetwork module with a recurrent model for the EdgeBlock and NodeBlock.

The newly constructed framework is then applied to the World Trade Network dataset, comprised of yearly trade data between countries in the 1997-2018 period, with countries being characterised by their annual GDP, to make 1-step-ahead predictions on the GDP trends, exploiting the feature of message-passing deriving from graph_nets.

The results of this framework are then compared to the ones obtained by an RNN applied to the GDP data only, without message-passing, similarly to what is done in the GraphIndependent module of graph_nets, to assess whether message-passing itself yields any improvement to the results.

For more information:
  - Peter W. Battaglia et al., Relational inductive biases, deep learning, and graph networks: https://arxiv.org/abs/1806.01261;
  - graph_nets library: https://github.com/deepmind/graph_nets;
  - World Trade Network data: https://comtrade.un.org/;

### Preprocessing
Raw data csv-s are contained in the ./data folder. It contains 190 folders named by the name of the corresponding country; each of them contains csv files referring to the commercial exchanges of a certain year, and a csv for the yearly GDP data. The polish_raw_data.py file in the ./preprocessing folder creates a first raw adjacency list dumped into the ./final_data/data.json file, and then the adjacency.py or adjacency_gpu.py files creates the final ./final_data/data_dict_list.json file containing data suited for the graph_nets modules.

### Training and validation
Both the Recurrent Graph Network and a simple RNN for node data only are implemented and can be both trained and validated. It is also possible to save trained models and reload them when needed.

The main.py file can be launched along with the following arguments:
  - --graph_nets: use graph_nets (y) or recurrent model (n);
  - --train_test: perform training only (tr), testing only (ts) or both (trts). Default to both;
  - --config_file: file for configurating training, contained in the ./config folder;
  - --pool_dim: feature dimension of nodes in EdgesToNodesAggregator function. Mandatory when doing validation only;
  - --save_model_as: name of file to save model's parameters in. If None, no model is saved;
  - --load_model_as: name of file to load saved model's parameters from. Mandatory if testing only;
  - --graph_file: name of graph's file. Mandatory argument.

The config files contain the n. of epochs and the learning rate for the node and edge block model for training, the model to use and (for RecGraphNetwork) the pool dimension for the reducer function inside the module.
