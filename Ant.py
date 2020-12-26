import numpy as np
import parameters as par
import networkx as nx

graph = nx.Graph(shortest=float('inf'))
all_time_shortest_path = []


class Ant:
    ant_id = 0

    def __init__(self, init_node, term_node):
        self.init_node = init_node  # wierzchołek startowy mrówki
        self.term_node = term_node  # wierzchołek końcowy mrówki
        self.location = init_node  # obecny wierzchołek
        self.vi_nodes = []  # lista odwiedzonych wierzchołków
        self.path_length = 0  # długość (suma wag) przebytych krawędzi
        self.retsize = 1  # mnożnik feromonów na najkrótszej ścieżce
        self.is_returning = 0  # flaga powrotu

        self.ant_id = Ant.ant_id
        Ant.ant_id += 1

    def pick_move(self):
        vi_ctr = 0
        possible_nodes = [n for n in graph[self.location]]  # wczytaj sasiadow

        for n in possible_nodes:  # zlicz sasiadow odwiedzonych
            if n in self.vi_nodes:
                vi_ctr += 1

        if vi_ctr < len(possible_nodes):  # jesli nie wszyscy odwiedzeni, usun odwiedzonych
            for n in self.vi_nodes:
                if n in possible_nodes:
                    possible_nodes.remove(n)

        row = np.array([graph[self.location][node]['pheromone'] for node in possible_nodes])
        dist = np.array([graph[self.location][node]['distance'] for node in possible_nodes])

        row = row ** par.ALPHA * ((1.0 / dist) ** par.BETA)  # liczymy tablicę prawdopodobieństw
        if row.sum() == 0:
            print("row.sum() = 0. mrówka:", self.ant_id, "wierzcholek:", self.location, possible_nodes, row,
                  self.vi_nodes)
            row += 1
        row = row / row.sum()
        # print("ant:", self.ant_id, "node:", self.location, "choices:", possible_nodes, "probs:", row)
        return np.random.choice(possible_nodes, 1, p=row)[0]  # i wybieramy nr wierzchołka następnego

    def step(self):
        next_node = self.init_node

        if self.is_returning == 1:  # powrót po odwiedzonych i pozostawienie feromonów
            next_node = self.vi_nodes.pop()
            new_pheromone = graph[next_node][self.location]['pheromone'] + self.retsize / self.path_length * \
                            graph[next_node][self.location]['distance']
            graph[next_node][self.location]['pheromone'] = min(par.MAX_PHER, new_pheromone)
        else:  # wybór nowego wierzchołka
            try:
                next_node = self.pick_move()
                self.vi_nodes.append(self.location)
                self.path_length += graph[self.location][next_node]['distance']
            except:
                exit()

        # zmiana wierzchołka i aktualizacja zmiennych
        self.location = next_node

        if self.location == self.term_node:
            self.is_returning = 1
            self.retsize = par.PHER_CONSTANT
            if self.path_length <= graph.graph['shortest']:
                self.retsize *= par.BIAS
                if self.path_length < graph.graph['shortest']:
                    graph.graph['shortest'] = self.path_length
                    global all_time_shortest_path
                    all_time_shortest_path = np.copy(self.vi_nodes)
        # print("ant", self.ant_id, "path: ", self.path, self.path_length)
        elif self.location == self.init_node:
            self.path_length = 0
            self.vi_nodes.clear()
            self.is_returning = 0
