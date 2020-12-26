import networkx as nx
import matplotlib.pyplot as plt
from readGraph import readGraph

G = readGraph()
nx.draw(G)
plt.show()
