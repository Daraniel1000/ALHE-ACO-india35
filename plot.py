import networkx as nx
import matplotlib.pyplot as plt

def getEdge(s:str):
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
	
def getNode(s:str):
	i = 2
	first = ''
	while s[i] != ' ':
		first += s[i]
		i += 1
	return int(first)

G = nx.Graph()
with open('nodes','r') as file:
    s = file.readline()
    while s:
        G.add_node(getNode(s))
        s = file.readline()
with open('edges','r') as file:
	s = file.readline()
	while s:
		edge = getEdge(s)
		G.add_edge(edge[0], edge[1], weight=edge[2])
		s = file.readline()
		
nx.draw(G)
plt.show()
