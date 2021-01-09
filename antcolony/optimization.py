import itertools
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor

import networkx as nx
import numpy as np


class PathFinder(ABC):
    @abstractmethod
    def find(self, graph, start_node, end_node):
        return NotImplemented


class DijkstraPathFinder(PathFinder):
    """Shortest path finder using NetworkX builtin Dijkstra algorithm"""

    def find(self, graph, start_node, end_node, weight_label='pheromone'):
        return nx.shortest_path(graph, start_node, end_node, weight=weight_label, method='dijkstra')


class AntColonyPathFinder(PathFinder):
    """Shortest path finder using ant colony optimization"""

    class Ant(PathFinder):
        def __init__(self, max_steps, alpha, beta, pheromone_label='pheromone', weight_label='pheromone'):
            super().__init__()
            self.max_steps = max_steps
            self.alpha = alpha
            self.beta = beta
            self.pheromone_label = pheromone_label
            self.weight_label = weight_label

        def _pick_next(self, graph, node):
            def safe_inverse(a):
                return np.divide(np.repeat(1.0, len(a)).astype(np.float), a.astype(np.float),
                                 out=np.zeros_like(a, dtype=a.dtype), where=a != 0)

            def sum_to_one(a):
                return a / a.sum() if a.any() else np.repeat(1 / len(a), len(a))

            # choose from neighbours based on pheromones and weights
            pheromones = np.array([n[self.pheromone_label] for n in graph[node].values()])
            weights = np.array([n[self.weight_label] for n in graph[node].values()])
            probabilities = sum_to_one(pheromones ** self.alpha * (safe_inverse(weights) ** self.beta))
            return np.random.choice(list(graph[node].keys()), p=probabilities)

        def find(self, graph, start_node, end_node):
            current_node, path = start_node, [start_node]
            for _ in range(self.max_steps):
                current_node = self._pick_next(graph, current_node)
                path.append(current_node)
                if current_node == end_node:
                    return path
            return []  # end node not reached, return empty path

    def __init__(self, n_ants, n_iter, max_steps, alpha, beta, ro, q):
        super().__init__()
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta
        self.ro = ro
        self.q = q
        self.pheromone_label = 'pheromone'

    def _init_pheromones(self, graph):
        nx.set_edge_attributes(graph, 0.0, name=self.pheromone_label)  # initial pheromone level on each edge

    def _decay_pheromones(self, graph):
        for e in graph.edges():
            graph[e[0]][e[1]][self.pheromone_label] = (1 - self.ro) * graph[e[0]][e[1]][self.pheromone_label]

    def _leave_pheromones(self, graph, paths):
        for path in paths:
            for a, b in zip(path, path[1:]):
                graph[a][b][self.pheromone_label] = graph[a][b][self.pheromone_label] + self.q / len(path)

    @staticmethod
    def _let_out_ants(ants, graph, start_node, end_node):
        # each ant on its own thread
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(ant.find, graph, start_node, end_node) for ant in ants]
        return [f.result() for f in futures]

    def find(self, graph, start_node, end_node):
        self._init_pheromones(graph)
        ants = [self.Ant(self.max_steps, self.alpha, self.beta, self.pheromone_label) for _ in range(self.n_ants)]
        shortest_path = []
        for _ in range(self.n_iter):
            paths = self._let_out_ants(ants, graph, start_node, end_node)
            self._decay_pheromones(graph)
            self._leave_pheromones(graph, paths)
            iteration_shortest_path = min(paths, key=len)
            if not shortest_path or len(iteration_shortest_path) < len(shortest_path):
                shortest_path = iteration_shortest_path
        return shortest_path


class MultiPathFinder:
    """K-shortest paths finder using the simplest possible method and any singular path finder"""

    def __init__(self, path_finder, k):
        super().__init__()
        self.path_finder = path_finder
        self.k = k

    @staticmethod
    def _edge_tuples(paths):
        if not paths:
            return [[]]
        edge_paths = [set(frozenset(x) for x in zip(path, path[1:])) for path in paths]
        edge_tuples_sets = [x for x in itertools.product(*edge_paths) if len(set(x)) == len(paths)]
        return [[tuple(edge) for edge in pair] for pair in edge_tuples_sets]

    @staticmethod
    def _without_edges(graph, edges):
        graph = graph.copy()
        graph.remove_edges_from(edges)
        return graph

    def find(self, graph, start_node, end_node):
        best_paths = []
        for _ in range(self.k):
            # find shortest path blocking already chosen paths
            # block path by removing one edge from each path for all possible tuples of edges
            candidate_paths = [self.path_finder.find(self._without_edges(graph, edges), start_node, end_node)
                               for edges in self._edge_tuples(best_paths)]
            non_empty_paths = [path for path in candidate_paths if path]
            if not non_empty_paths:  # can't search further
                return best_paths
            best_paths.append(min(non_empty_paths, key=len))
        return best_paths
