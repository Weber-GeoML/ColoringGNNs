import networkx as nx
from full_and_triple_fast import triple_color
from full_and_triple_fast import loss

g = nx.erdos_renyi_graph(200, 20/199)
colors = triple_color(g, 7)
print(loss(g, colors))
