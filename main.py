from utils.parse import parser
from config import Config
from utils.load_graphs import graph_loader
from models.graph_nets_deploy import train_graph_nets, validate_graph_nets, train_test_graph_nets
from models.simple_rnn import train_rnn, validate_rnn, train_test_rnn

def main():
    args = parser()
    graph_data = graph_loader(args.graph_file, args.train_test)
    if args.train_test == "tr":
        config = Config(args.config_file)
        if args.graph_nets == "y":
            train_graph_nets(*graph_data, config,
                    args.save, args.load)
        else:
            train_rnn(*graph_data, config,
                    args.save, args.load)
    elif args.train_test == "ts":
        if args.graph_nets == "y":
            validate_graph_nets(*graph_data, args.pool_dim,
                            args.load)
        else:
            validate_rnn(*graph_data,
                args.load)
    else:
        config = Config(args.config_file)
        if args.graph_nets == "y":
            train_test_graph_nets(config, args.save,
                             args.load, *graph_data)
        else:
            train_test_rnn(config, args.save,
                        *graph_data, load_model=args.load)

if __name__ == '__main__':
    main()
