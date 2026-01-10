import networkx as nx
import numpy as np
from full_gcn import full_gcn_color
from full_gcn import loss

n = 200
d = 20

k = 1
while not 2*k*np.log(k) > d:
	k += 1
k += 1

g = nx.erdos_renyi_graph(n, d/(n-1))
colors = full_gcn_color(g, k)
print(f'Loss: {loss(g, colors)}')