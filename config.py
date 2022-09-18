import os
import configparser

class Config():
    """
    Config object. Reads config options from config files.
    Options are divided into:
      - epochs:
        - ep_nodes: for node training;
        - ep_edges: for edge training;
      - learning rates:
        - l_nodes: for node training;
        - l_edges: for edge training;
      - model options:
        - graph nets: use graph net (y) or a simple rnn (n);
        - pool dim: dimension of the pooling of model aggregators.
    """
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(os.path.join("config", config_file))
        except:
            print(f'Failed to read {config_file}.')
        if conf.has_option("Epochs", "ep_nodes"):
            self.epochs_nodes = int(conf.get("Epochs", "ep_nodes"))
        else:
            self.epochs_nodes = 10
        if conf.has_option("Epochs", "ep_edges"):
            self.epochs_edges = int(conf.get("Epochs", "ep_edges"))
        else:
            self.epochs_edges = 10
        if conf.has_option("Learning_rates", "l_nodes"):
            self.l_rate_nodes = float(conf.get("Learning_rates", "l_nodes"))
        else:
            self.l_rate_nodes = 0.001
        if conf.has_option("Learning_rates", "l_edges"):
            self.l_rate_edges = float(conf.get("Learning_rates", "l_edges"))
        else:
            self.l_rate_edges = 0.001
        if conf.has_option("Model_options", "graph_nets"):
            self.graph_nets = conf.get("Model_options", "graph_nets")
            if self.graph_nets not in ["y", "n"]:
                raise ValueError('Invalid config argument to '
                                            '\'model\'.')
        else:
            raise ValueError('Missing mandatory argument in'
                            ' config file: \'model\'.')
        if conf.has_option("Model_options", "pool_dim"):
            self.pool_dim = int(conf.get("Model_options", "pool_dim"))
        else:
            self.pool_dim = 3 if self.graph_nets == 'y' else None
