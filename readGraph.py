import networkx as nx
import parameters as par


def getedge(s: str):
    i = 0
    while s[i] != '(':
        i += 1
    i += 2
    first = ''
    while s[i] != ' ':
        first += s[i]
        i += 1
    i += 1
    second = ''
    while s[i] != ' ':
        second += s[i]
        i += 1
    weight = ''
    while s[i] != '(':
        i += 1
    i += 2
    while s[i] != ' ':
        i += 1
    i += 1
    while s[i] != ' ':
        weight += s[i]
        i += 1
    return int(first), int(second), float(weight)


def getnode(s: str):
    i = 2
    first = ''
    while s[i] != ' ':
        first += s[i]
        i += 1
    return int(first)


def readGraph():
    G = nx.Graph()
    with open('nodes', 'r') as file:
        s = file.readline()
        while s:
            G.add_node(getnode(s))
            s = file.readline()
    with open('edges', 'r') as file:
        s = file.readline()
        while s:
            edge = getedge(s)
            G.add_edge(edge[0], edge[1], distance=edge[2], pheromone=par.MIN_PHER)
            s = file.readline()
    return G
