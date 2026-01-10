import networkx as nx
import random

def loss(g, colors):
	return len([edge for edge in g.edges() if colors[edge[0]] == colors[edge[1]]])

class Coloring:

	def __init__(self, adj_lst, k, colors, neighbor_color_frequencies, conflicts, loss):
		self.adj_lst = adj_lst
		self.k = k
		self.colors = colors
		self.neighbor_color_frequencies = neighbor_color_frequencies
		self.conflicts = conflicts
		self.loss = loss

	def improve(self):
		k = self.k + 1
		colors = self.colors.copy()
		neighbor_color_frequencies = [row.copy() for row in self.neighbor_color_frequencies]
		neighbor_color_frequencies.append([0 for color in self.colors])
		conflicts = self.conflicts.copy()
		improvements = self.conflicts.copy()
		loss = self.loss
		while True:
			node_choices = []
			max_improvement = improvements[0]
			for i, improvement in enumerate(improvements):
				if improvement > max_improvement:
					node_choices = []
					max_improvement = improvement
				if improvement == max_improvement:
					node_choices.append(i)
			if max_improvement == 0:
				break
			update_node = random.choice(node_choices)
			loss -= max_improvement
			color_choices = []
			min_frequency = neighbor_color_frequencies[0][update_node]
			for i in range(k):
				if neighbor_color_frequencies[i][update_node] < min_frequency:
					color_choices = []
					min_frequency = neighbor_color_frequencies[i][update_node]
				if neighbor_color_frequencies[i][update_node] == min_frequency:
					color_choices.append(i)
			update_color = random.choice(color_choices)
			for neighbor in self.adj_lst[update_node]:
				if neighbor_color_frequencies[colors[neighbor]][neighbor] - neighbor_color_frequencies[colors[update_node]][neighbor] == improvements[neighbor]:
					improvements[neighbor] += 1
				if neighbor_color_frequencies[colors[neighbor]][neighbor] - neighbor_color_frequencies[update_color][neighbor] == improvements[neighbor]:
					for i in range(k):
						if neighbor_color_frequencies[colors[neighbor]][neighbor] - neighbor_color_frequencies[i][neighbor] + (1 if i == colors[update_node] else 0) == improvements[neighbor] and i != update_color:
							break
					else:
						improvements[neighbor] -= 1
				if colors[update_node] == colors[neighbor]:
					conflicts[neighbor] -= 1
					improvements[neighbor] -= 1
				if update_color == colors[neighbor]:
					conflicts[neighbor] += 1
					improvements[neighbor] += 1
				neighbor_color_frequencies[colors[update_node]][neighbor] -= 1
				neighbor_color_frequencies[update_color][neighbor] += 1
			conflicts[update_node] -= max_improvement
			improvements[update_node] = 0
			colors[update_node] = update_color
		return Coloring(self.adj_lst, k, colors, neighbor_color_frequencies, conflicts, loss)

def full_color(g, k):
	adj_lst = [list(g.neighbors(node)) for node in range(g.order())]
	colors = Coloring(adj_lst, 1, [0] * g.order(), [[len(neighbors) for neighbors in adj_lst]], [len(neighbors) for neighbors in adj_lst], g.size())
	while colors.k < k:
		colors = colors.improve()
	return colors.colors

def triple_color(g, k):
	adj_lst = [list(g.neighbors(node)) for node in range(g.order())]
	colors = Coloring(adj_lst, 1, [0] * g.order(), [[len(neighbors) for neighbors in adj_lst]], [len(neighbors) for neighbors in adj_lst], g.size())
	coloring_lst = [colors]
	final_colors = colors
	while len(coloring_lst) > 0:
		colors = coloring_lst.pop()
		for new_colors in [colors.improve(), colors.improve(), colors.improve()]:
			if new_colors.k == k:
				if new_colors.loss < final_colors.loss:
					final_colors = new_colors
			else:
				coloring_lst.append(new_colors)
	return final_colors.colors