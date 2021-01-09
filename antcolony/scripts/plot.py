import configargparse
import matplotlib.pyplot as plt
import networkx as nx

from antcolony.parsing import parse_sndlib_weighted


def parse_args():
    argparser = configargparse.ArgParser(prog=__package__,
                                         description="{} - graph plotter".format(__package__))
    argparser.add_argument("graph", type=str,
                           help='path to graph definition file')
    return argparser.parse_args()


def get_graph(path):
    with open(path) as file:
        return parse_sndlib_weighted(file.read())


if __name__ == '__main__':
    args = parse_args()
    graph = get_graph(args.graph)
    nx.draw(graph)
    plt.show()
