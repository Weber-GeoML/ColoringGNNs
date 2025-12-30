
import networkx as nx
import random

def loss(g, colors):
	return len([edge for edge in g.edges() if colors[edge[0]] == colors[edge[1]]])

def random_color(g, k):
	return random.choices(range(k), k=g.order())

def discrete_color_(g, k, starting_colors):
	order = g.order()
	colors = starting_colors.copy()
	neighbor_colors = [{c:0 for c in range(k)} for i in range(order)]
	for node, color in enumerate(colors):
		for neighbor in nx.neighbors(g, node):
			neighbor_colors[neighbor][color] += 1
	improvements = [neighbor_colors[node][colors[node]] - min(neighbor_colors[node].values()) for node in range(order)]
	while True:
		max_improvement = max(improvements)
		if max_improvement == 0:
			return colors
		max_improvement_nodes = [node for node, improvement in enumerate(improvements) if improvement == max_improvement]
		update_node = random.choice(max_improvement_nodes)
		min_neighbor_color_frequency = min(neighbor_colors[update_node].values())
		max_improvement_colors = [color for color in range(k) if neighbor_colors[update_node][color] == min_neighbor_color_frequency]
		update_color = random.choice(max_improvement_colors)
		old_color = colors[update_node]
		colors[update_node] = update_color
		improvements[update_node] = 0
		for neighbor in nx.neighbors(g, update_node):
			neighbor_colors[neighbor][old_color] -= 1
			neighbor_colors[neighbor][update_color] += 1
			improvements[neighbor] = neighbor_colors[neighbor][colors[neighbor]] - min(neighbor_colors[neighbor].values())

def triple_color_(g, k, k_start, starting_colors):
	if k_start == k:
		return starting_colors
	final_colors_1 = triple_color_(g, k, k_start+1, discrete_color_(g, k_start+1, starting_colors))
	final_colors_2 = triple_color_(g, k, k_start+1, discrete_color_(g, k_start+1, starting_colors))
	final_colors_3 = triple_color_(g, k, k_start+1, discrete_color_(g, k_start+1, starting_colors))
	return min((final_colors_1, final_colors_2, final_colors_3), key = lambda colors: loss(g, colors))

def discrete_color(g, k):
	return discrete_color_(g, k, starting_colors=random_color(g, k))

def full_color(g, k):
	if k == 1:
		return random_color(g, 1)
	return discrete_color_(g, k, full_color(g, k - 1))

def triple_color(g, k):
	return triple_color_(g, k, 1, random_color(g, 1))
