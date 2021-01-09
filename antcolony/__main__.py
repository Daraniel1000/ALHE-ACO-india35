import importlib.resources as pkg_resources
import logging

import configargparse
import networkx as nx

from antcolony.optimization import AntColonyPathFinder, MultiPathFinder
from antcolony.parsing import parse_sndlib_weighted


def parse_args():
    with pkg_resources.path("antcolony.resources", "config.yaml") as config_path:
        argparser = configargparse.ArgParser(prog=__package__,
                                             description="{} - ant colony optimization".format(__package__),
                                             default_config_files=[str(config_path)])
        argparser.add_argument("graph", type=str,
                               help='path to graph definition file')
        argparser.add_argument("start_node", type=str,
                               help='starting node')
        argparser.add_argument("end_node", type=str,
                               help='ending node')
        argparser.add_argument("--n", type=int, default=1,
                               help='how many paths to find')
        argparser.add_argument('--config', is_config_file=True,
                               help='config file path')
        argparser.add_argument('--n_ants', type=int, default=50,
                               help='number of ants')
        argparser.add_argument('--n_iter', type=int, default=10,
                               help='number of iterations')
        argparser.add_argument('--max_steps', type=int, default=1000,
                               help='max steps for single ant in one iteration')
        argparser.add_argument('--alpha', type=float, default=0.5,
                               help='alpha coefficient')
        argparser.add_argument('--beta', type=float, default=1.2,
                               help='beta coefficient')
        argparser.add_argument('--ro', type=float, default=0.6,
                               help='pheromone evaporation coefficient')
        argparser.add_argument('--q', type=float, default=10,
                               help='q coefficient')
    return argparser.parse_args()


def config_logging():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')


def get_graph(path, weight_label='weight'):
    with open(path) as file:
        return parse_sndlib_weighted(file.read(), weight_label)


def nx_shortest_paths(graph, start_node, end_node, k, weight_label='weight'):
    return [path for _, path in
            zip(range(k), nx.shortest_simple_paths(graph, start_node, end_node, weight=weight_label))]


if __name__ == '__main__':
    args = parse_args()
    config_logging()
    logger = logging.getLogger("antcolony")

    weight_label = 'weight'

    graph = get_graph(args.graph, weight_label)
    optimizer = AntColonyPathFinder(args.n_ants, args.n_iter, args.max_steps,
                                    args.alpha, args.beta, args.ro, args.q)

    start, end, n = args.start_node, args.end_node, args.n

    logger.info("Starting search")
    paths = MultiPathFinder(optimizer, n).find(graph, start, end)
    logger.info(f"Shortest paths according to ant colony: {paths}")

    logger.info(f"Shortest paths according to NetworkX: {nx_shortest_paths(graph, start, end, n, weight_label)}")
