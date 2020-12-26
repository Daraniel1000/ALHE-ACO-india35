from readGraph import readGraph
import networkx as nx
import Ant
import parameters as par

ants = []

def init(startNode, endNode):
    Ant.graph.update(readGraph())
    for k in range(par.NUM_OF_ANTS):
        ants.append(Ant.Ant(startNode, endNode))
    #now for n of iterations run ants asynchroneously until they find their solutions, update the graph and go again
    #TODO modify Ant to work that way instead of the old pheromone update system
    #TODO implement async running of ants
    #TODO implement Yen's algorithm
