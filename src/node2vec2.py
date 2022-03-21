import numpy as np
import networkx as nx
import random
import pdb


class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.指定的边的setup列表
		'''
		G = self.G
		p = self.p#1
		q = self.q#1

		unnormalized_probs = []#根据p，q计算可能性
		for dst_nbr in sorted(G.neighbors(dst)):#终点的邻接点
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]#计算可能性的占比

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.预处理引导随机walk的概率
		'''
		G = self.G#nx对象
		is_directed = self.is_directed

		alias_nodes = {}

		for node in G.nodes():#NodeView，根据有向边的节点出现顺序提取的NodeView:NodeView((1, 32, 22, 20, 18, 14, 13, 12, 11, 9, 8, 7, 6, 5, 4, 3, 2, 34, 33, 29, 26, 25, 17, 10, 28, 31, 15, 16, 19, 21, 23, 24, 30, 27))
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]#对某个节点，构造一个邻接节点的权重列表，但是生成的例子权重都是1,[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
			norm_const = sum(unnormalized_probs)#16,
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]#对某个节点，构造边的权重的占比列表,[0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]
			alias_nodes[node] = alias_setup(normalized_probs)#点的别名数组
        #{1: (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), 32: (array([0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1.])),
        # 22: (array([0, 0]), array([1., 1.])), 20: (array([0, 0, 0]), array([1., 1., 1.])), 18: (array([0, 0]), array([1., 1.])), 14: (array([0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1.])), 13: (array([0, 0]), array([1., 1.])),
        # 12: (array([0]), array([1.])), 11: (array([0, 0, 0]), array([1., 1., 1.])), 9: (array([0, 0, 0, 0]), array([1., 1., 1., 1.])), 8: (array([0, 0, 0, 0]), array([1., 1., 1., 1.])), 7: (array([0, 0, 0, 0]), array([1., 1., 1., 1.])),
        # 6: (array([0, 0, 0, 0]), array([1., 1., 1., 1.])), 5: (array([0, 0, 0]), array([1., 1., 1.])), 4: (array([0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1.])), 3: (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),
        # 2: (array([0, 0, 0, 0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1., 1., 1., 1.])), 34: (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), 33: (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), 29: (array([0, 0, 0]), array([1., 1., 1.])), 26: (array([0, 0, 0]), array([1., 1., 1.])), 25: (array([0, 0, 0]), array([1., 1., 1.])), 17: (array([0, 0]), array([1., 1.])), 10: (array([0, 0]), array([1., 1.])), 28: (array([0, 0, 0, 0]), array([1., 1., 1., 1.])), 31: (array([0, 0, 0]), array([1., 1., 1.])), 15: (array([0, 0]), array([1., 1.])), 16: (array([0, 0]), array([1., 1.])), 19: (array([0, 0]), array([1., 1.])), 21: (array([0, 0]), array([1., 1.])), 23: (array([0, 0]), array([1., 1.])), 24: (array([0, 0, 0, 0, 0]), array([1., 1., 1., 1., 1.])), 30: (array([0, 0, 0, 0]), array([1., 1., 1., 1.])), 27: (array([0, 0]), array([1., 1.]))}
		alias_edges = {}
		triads = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])#计算边的alias数组
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.计算来自离散分布的非均匀抽样的utility列表。
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)#16
	q = np.zeros(K)#
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []

	for kk, prob in enumerate(probs):
	    q[kk] = K*prob#1.0
	    if q[kk] < 1.0:
	        smaller.append(kk)#[]
	    else:
	        larger.append(kk)#[0,1,...15,]

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

	return J, q#27: (array([0, 0]), array([1., 1.]))

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]