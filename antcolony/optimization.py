import itertools
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor

import networkx as nx
import numpy as np


class Path:
    def __init__(self, nodes, graph, weight_label='weight'):
        self.graph = graph
        self.nodes = nodes
        self.len = None if nodes else float('inf')
        self.weight_label = weight_label

    def nodes(self):
        return self.nodes

    def edges(self):
        return zip(self.nodes, self.nodes[1:])

    def length(self):
        if self.len is None:
            self.len = sum(self.graph[a][b][self.weight_label] for a, b in self.edges())
        return self.len

    def __repr__(self):
        return str((self.length(), self.nodes))

    def __bool__(self):
        return bool(self.nodes)


class PathFinder(ABC):
    @abstractmethod
    def find(self, graph, start_node, end_node):
        return NotImplemented


class DijkstraPathFinder(PathFinder):
    """Shortest path finder using NetworkX builtin Dijkstra algorithm"""

    def __init__(self, weight_label='weight'):
        self.weight_label = weight_label

    def find(self, graph, start_node, end_node):
        return Path(nx.shortest_path(graph, start_node, end_node, weight=self.weight_label, method='dijkstra'),
                    graph,
                    self.weight_label)


class AntColonyPathFinder(PathFinder):
    """Shortest path finder using ant colony optimization"""

    class Ant(PathFinder):
        def __init__(self, max_steps, alpha, beta, weight_label='weight', pheromone_label='pheromone'):
            super().__init__()
            self.max_steps = max_steps
            self.alpha = alpha
            self.beta = beta
            self.weight_label = weight_label
            self.pheromone_label = pheromone_label

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
                    return Path(path, graph, self.weight_label)
            return Path([], graph, self.weight_label)  # end node not reached, return empty path

    def __init__(self, n_ants, n_iter, max_steps, alpha, beta, ro, q, weight_label='weight'):
        super().__init__()
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta
        self.ro = ro
        self.q = q
        self.weight_label = weight_label
        self.pheromone_label = 'pheromone'

    def _init_pheromones(self, graph):
        nx.set_edge_attributes(graph, 0.0, name=self.pheromone_label)  # initial pheromone level on each edge

    def _decay_pheromones(self, graph):
        for a, b in graph.edges():
            graph[a][b][self.pheromone_label] = max(0, (1 - self.ro) * graph[a][b][self.pheromone_label])

    def _leave_pheromones(self, graph, paths):
        for path in paths:
            for a, b in path.edges():
                graph[a][b][self.pheromone_label] = graph[a][b][self.pheromone_label] + self.q / path.length()

    @staticmethod
    def _let_out_ants(ants, graph, start_node, end_node):
        # each ant on its own thread
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(ant.find, graph, start_node, end_node) for ant in ants]
        return [f.result() for f in futures]

    def find(self, graph, start_node, end_node):
        self._init_pheromones(graph)
        ants = [self.Ant(self.max_steps, self.alpha, self.beta, self.weight_label, self.pheromone_label)
                for _ in range(self.n_ants)]
        shortest_path = []
        for _ in range(self.n_iter):
            paths = self._let_out_ants(ants, graph, start_node, end_node)
            self._decay_pheromones(graph)
            self._leave_pheromones(graph, paths)
            iteration_shortest_path = min(paths, key=lambda p: p.length())
            if not shortest_path or iteration_shortest_path.length() < shortest_path.length():
                shortest_path = iteration_shortest_path
        return shortest_path


class MultiPathFinder(ABC):
    @abstractmethod
    def find(self, graph, start_node, end_node, k):
        return NotImplemented


class SimpleMultiPathFinder(MultiPathFinder):
    """K-shortest paths finder using the simplest possible method and any singular path finder"""

    def __init__(self, path_finder):
        super().__init__()
        self.path_finder = path_finder

    @staticmethod
    def _edge_tuples(paths):
        if not paths:
            return [[]]
        edge_paths = [set(frozenset(edge) for edge in path.edges()) for path in paths]
        edge_tuples_sets = [x for x in itertools.product(*edge_paths) if len(set(x)) == len(paths)]
        return [[tuple(edge) for edge in pair] for pair in edge_tuples_sets]

    @staticmethod
    def _without_edges(graph, edges):
        graph = graph.copy()
        graph.remove_edges_from(edges)
        return graph

    def find(self, graph, start_node, end_node, k):
        best_paths = []
        for _ in range(k):
            # find shortest path blocking already chosen paths
            # block path by removing one edge from each path for all possible tuples of edges
            candidate_paths = [self.path_finder.find(self._without_edges(graph, edges), start_node, end_node)
                               for edges in self._edge_tuples(best_paths)]
            non_empty_paths = [path for path in candidate_paths if path]
            if not non_empty_paths:  # can't search further
                return best_paths
            best_paths.append(min(non_empty_paths, key=lambda p: p.length()))
        return best_paths


class NxMultiPathFinder(MultiPathFinder):
    """K-shortest paths finder using NetworkX builtin algorithm"""

    def __init__(self, weight_label='weight'):
        super().__init__()
        self.weight_label = weight_label

    def find(self, graph, start_node, end_node, k):
        return [Path(path, graph, self.weight_label) for _, path in
                zip(range(k), nx.shortest_simple_paths(graph, start_node, end_node, weight=self.weight_label))]
