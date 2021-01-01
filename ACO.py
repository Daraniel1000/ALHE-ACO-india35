from readGraph import readGraph
import networkx as nx
import Ant
import parameters as par
from threading import Thread

ants = []


def run(ant: Ant):
    print('start ' + str(ant.ant_id))
    x = 1
    for i in range(1000):
        x *= 2
    print('finish ' + str(ant.ant_id))
    return
    #TODO test this
    n_steps = 0
    while ant.is_returning != 1:
        ant.step()
        n_steps += 1
        if n_steps > par.MAX_STEPS:
            return


def init(startNode, endNode):
    Ant.graph.update(readGraph())
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
        #then update pheromones, update best route, reset ants
        for ant in ants:
            #update pheromones, update best route
            ant.reset()

    #now for n of iterations run ants asynchroneously until they find their solutions, update the graph and go again
    #TODO modify Ant to work that way instead of the old pheromone update system
    #TODO implement async running of ants
    #TODO implement Yen's algorithm
