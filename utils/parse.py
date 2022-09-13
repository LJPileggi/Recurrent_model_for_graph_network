import argparse

def parser():
    parser = argparse.ArgumentParser(description='Graph Networks.')
    parser.add_argument('--graph_nets', type=str, dest='graph_nets',
            help='use graph_nets (y) or recurrent model (n).')
    parser.add_argument('--train_test', type=str, dest='train_test',
            help='perform training only (tr), testing only (ts) '
            'or both (trts). Default to both.')
    parser.add_argument('--config_file', type=str, dest='config_file',
            help='file for configurating training')
    parser.add_argument('--pool_dim', type=int, dest='pool_dim',
            help='feature dimension of nodes in EdgesToNodesAggregator '
            'function. Mandatory when doing validation only.')
    parser.add_argument('--save_model_as', type=str, dest='save',
            help='name of file to save model\'s parameters in. '
            'If None, no model is saved.')
    parser.add_argument('--load_model_as', type=str, dest='load',
            help='name of file to load saved model\'s parameters'
            ' from. Mandatory if testing only.')
    parser.add_argument('--graph_file', type=str, dest='graph_file',
            help='name of graph\'s file. Mandatory argument.')
    parser.set_defaults(graph_nets='y')
    parser.set_defaults(train_test='trts')
    parser.set_defaults(config_file=None)
    parser.set_defaults(pool_dim=None)
    parser.set_defaults(graph_nets='n')
    parser.set_defaults(save=None)
    parser.set_defaults(load=None)
    parser.set_defaults(graph_file=None)
    args = parser.parse_args()
    if args.graph_nets not in ['y', 'n']:
        raise ValueError('Invalid argument passed to \'--graph_nets\'.')
    if args.train_test not in ['tr', 'ts', 'trts']:
        raise ValueError('Invalid argument passed to \'--train_test\'.')
    if (args.train_test == 'ts') & (args.load == None):
        raise ValueError('Invalid argument passed to \'--load_model_as\':'
                ' missing loaded model for testing.')
    if (args.graph_nets == 'y') & (args.train_test == 'ts') & (args.pool_dim == None):
        raise ValueError('Missing \'pool_dim\' argument for validation.')
    if (args.train_test in ['tr', 'trts']) & (not args.config_file):
        raise ValueError('Missing configuration file for training.')
    if args.graph_file == None:
        raise ValueError('Invalid argument passed to \'--graph_file\': '
                'no graph data provided.')
    return args
