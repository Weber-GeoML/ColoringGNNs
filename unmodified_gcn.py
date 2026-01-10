import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from itertools import chain

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
	return len([edge for edge in g.edges() if colors[edge[0]] == colors[edge[1]]])

def unmod_gcn_color(g, k):
	data = from_networkx(g)
	edges = data.edge_index
	A = torch.tensor(nx.to_numpy_array(g, dtype='float32'))
	features = [200, k]
	net = GCN(features)
	x = torch.nn.Parameter(torch.randn(g.order(), features[0]))
	params = chain(net.parameters(), [x])
	optimizer = torch.optim.AdamW(params)
	min_hard_loss = torch.inf
	min_soft_loss = torch.inf
	old_min_soft_loss = torch.inf
	soft_losses = []
	while True:
		optimizer.zero_grad()
		result = net(x, edges)
		probs = torch.nn.functional.softmax(result, 1)
		soft_loss = torch.tensordot(A, probs @ probs.T)/2
		soft_loss_val = soft_loss.detach().item()
		colors = [int(probs[i].argmax()) for i in range(g.order())]
		hard_loss_val = loss(g, colors)
		if hard_loss_val < min_hard_loss:
			min_hard_loss = hard_loss_val
			best_colors = colors
		if min_hard_loss == 0:
			break
		if soft_loss_val < min_soft_loss:
			min_soft_loss = soft_loss_val
		soft_losses.append(soft_loss_val)
		if len(soft_losses) > 5000 and soft_losses[-5001] < old_min_soft_loss:
			old_min_soft_loss = soft_losses[-5001]
			if old_min_soft_loss - min_soft_loss < 0.001:
				break
		soft_loss.backward()
		optimizer.step()
	return best_colors