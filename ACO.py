import networkx as nx
from threading import Thread
import numpy as np
from readGraph import readGraph
import Ant
import parameters as par

ants = []


def run(ant: Ant):
    n_steps = 0
    while ant.is_returning != 1:
        ant.step()
        n_steps += 1
        if n_steps > par.MAX_STEPS:
            return


def init(startNode, endNode):
    graph = Ant.graph
    graph.update(readGraph())
    for k in range(par.NUM_OF_ANTS):
        ants.append(Ant.Ant(startNode, endNode))
    for i in range(par.N_ITER):
        threads = []
        for ant in ants:
            it = Thread(target=run, args=(ant,))
            threads.append(it)
            it.start()
        for thread in threads:
            thread.join()
        print('################ ALL THREADS JOINED ################')
        for u, v, p in graph.edges.data('pheromone'):
            graph[u][v]['pheromone'] = max(par.MIN_PHER, p*par.DECAY)
        for ant in ants:
            #update best route
            if graph.graph['shortest'] > ant.path_length:
                graph.graph['shortest'] = ant.path_length
                Ant.all_time_shortest_path = np.copy(ant.vi_nodes)
            #update pheromones
            prev = -1
            for v in ant.vi_nodes:
                if prev == -1:
                    prev = v
                    continue
                print('prev: ' + str(prev) + ' v: ' + str(v))
                graph[v][prev]['pheromone'] = min(par.MAX_PHER, graph[v][prev]['pheromone'] +
                                                      par.PHER_CONSTANT / ant.path_length)
                prev = v
            #reset ant
            ant.reset()
    print('path length: ' + str(graph.graph['shortest']))
    print('path: ' + str(Ant.all_time_shortest_path))

    #now for n of iterations run ants asynchroneously until they find their solutions, update the graph and go again
    #DONE-ish? modify Ant to work that way instead of the old pheromone update system
    #p much DONE implement async running of ants
    #TODO test this mess
    #TODO optimize parameters and find the right amounts for pheromones
    #TODO implement Yen's algorithm
