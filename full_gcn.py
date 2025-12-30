
import networkx as nx
import random
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import numpy as np
from itertools import chain

def discrete_color(g, k, starting_colors):
	order = g.order()
	colors = starting_colors.copy()
	neighbor_colors = []
	for i in range(order):
		neighbor_colors.append({c:0 for c in range(k)})
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

class GCN(torch.nn.Module):

	def __init__(self, features):
		super().__init__()
		self.layers = torch.nn.ModuleList()
		for i in range(len(features) - 1):
			self.layers.append(GCNConv(features[i], features[i+1], add_self_loops=False))

	def forward(self, x, edges):
		for i, layer in enumerate(self.layers):
			x = layer(x, edges)
			if i + 1 < len(self.layers):
				x = torch.nn.functional.relu(x)
		return x

def loss(g, colors):
	return len([edge for edge in g.edges if colors[edge[0]] == colors[edge[1]]])

class Coloring:

	def __init__(self, g, k, colors, loss):
		self.g = g
		self.k = k
		self.colors = colors
		self.loss = loss

	def improve(self):
		g = self.g
		k = self.k + 1
		colors = self.colors
		data = from_networkx(g)
		edges = data.edge_index
		A = torch.tensor(nx.to_numpy_array(g, dtype='float32'))
		features = [200, k]
		net = GCN(features)
		q, r = torch.linalg.qr(torch.randn(200, 200))
		x = torch.nn.Parameter(q[:g.order()])
		params = chain(net.parameters(), [x])
		optimizer = torch.optim.AdamW(params)
		desired_probs = torch.zeros(g.order(), k)
		desired_probs_ = torch.zeros(g.order(), k)
		for i in range(g.order()):
			for j in range(k):
				desired_probs[i, j] = 0.55 if colors[i] == j else 0.45/(k-1)
				desired_probs_[i, j] = 0.55 if colors[i] == j else 0
		while True:
			optimizer.zero_grad()
			result = net(x, edges)
			probs = torch.nn.functional.softmax(result, 1)
			if (desired_probs_ - probs).max() < 0.05:
				break
			dif = desired_probs - probs
			l = torch.tensordot(dif, dif)
			l.backward()
			optimizer.step()
		min_hard_loss = torch.inf
		min_mod_loss = torch.inf
		old_min_mod_loss = torch.inf
		mod_losses = []
		while True:
			optimizer.zero_grad()
			result = net(x, edges)
			probs = torch.nn.functional.softmax(result, 1)
			mod_loss = torch.tensordot(torch.linalg.matrix_power(torch.diag(A @ torch.ones(len(A))), 3) @ A, probs @ probs.T)/2
			mod_loss_val = mod_loss.detach().item()
			colors = [int(probs[i].argmax()) for i in range(g.order())]
			hard_loss_val = loss(g, colors)
			if hard_loss_val < min_hard_loss:
				min_hard_loss = hard_loss_val
				best_colors = colors
			if min_hard_loss == 0:
				break
			if mod_loss_val < min_mod_loss:
				min_mod_loss = mod_loss_val
			mod_losses.append(mod_loss_val)
			if len(mod_losses) > 5000 and mod_losses[-5001] < old_min_mod_loss:
				old_min_mod_loss = mod_losses[-5001]
				if old_min_mod_loss - min_mod_loss < 0.001:
					break
			mod_loss.backward()
			optimizer.step()
		final_colors = discrete_color(g, k, best_colors)
		final_loss = loss(g, best_colors)
		return Coloring(g, k, final_colors, final_loss)

def full_gcn_color(g, k):
	colors = Coloring(g, 1, [0]*g.order(), g.size())
	while colors.k < k:
		print(f'Producing {colors.k+1}-coloring...')
		colors = colors.improve()
	return colors.colors
