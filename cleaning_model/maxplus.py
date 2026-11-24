# Description: Max-Plus message passing on a coordination graph (Guestrin, 2002; Kok & Vlassis, 2006).
# Authors:
#   Alan Oropeza Saucedo - A01661614
#   Armando Vásquez Ambrocio - A01669283
#   David Alejandro Goncalves Pestana - A01669122
#   Hugo Fernando Rodríguez Montiel - A01782589
#   Lorenz Rasbach - A01832437
# Date: 2026-11-21

import numpy as np


# Purpose: global payoff G(a) = sum_i q_i(a_i) + sum_(i,j) f_ij(a_i, a_j) of a joint action.
# Parameters: actions, chosen action per agent, q, local utilities, edges, list of (i, j), f, pairwise tables.
# Returns: float.
def globalPayoff(actions, q, edges, f):
	total = sum(q[i][actions[i]] for i in range(len(q)))
	for (i, j), fij in zip(edges, f):
		total += fij[actions[i], actions[j]]
	return float(total)


# Purpose: Max-Plus message passing. Agent i sends neighbor j
#          mu_ij(a_j) = max_ai [ q_i(a_i) + f_ij(a_i, a_j) + sum_{k in N(i) \ j} mu_ki(a_i) ]
#          and every agent picks argmax_ai [ q_i(a_i) + sum_k mu_ki(a_i) ]. Because the graph can have
#          cycles, the anytime variant is used: the best joint action seen in any iteration is kept.
# Parameters: q, list of utility arrays (one per agent), edges, list of (i, j),
#             f, list of (A_i x A_j) arrays (one per edge), maxIters, maximum iterations,
#             damping, weight of the old message (0 = no damping), tol, convergence tolerance.
# Returns: tuple (numpy array with the chosen action per agent, iterations run).
def maxPlus(q, edges, f, maxIters=30, damping=0.5, tol=1e-6):
	n = len(q)
	neighbors = [[] for _ in range(n)]
	pair = {}
	for (i, j), fij in zip(edges, f):
		neighbors[i].append(j)
		neighbors[j].append(i)
		pair[(i, j)] = fij
		pair[(j, i)] = fij.T
	mu = {(i, j): np.zeros(len(q[j])) for (i, j) in pair}

	# Purpose: q_i plus the incoming messages, optionally leaving out the one from `exclude`.
	def beliefs(i, exclude=None):
		g = q[i].copy()
		for k in neighbors[i]:
			if k != exclude:
				g += mu[(k, i)]
		return g

	# Purpose: every agent picks the action that maximizes its beliefs.
	def decide():
		return np.array([int(np.argmax(beliefs(i))) for i in range(n)])

	best = decide()
	bestValue = globalPayoff(best, q, edges, f)
	if not edges:
		return best, 0  # no neighbors: independent argmax is already optimal

	iterations = 0
	for iterations in range(1, maxIters + 1):
		newMu = {}
		delta = 0.0
		for (i, j), fij in pair.items():
			msg = np.max(beliefs(i, exclude=j)[:, None] + fij, axis=0)
			msg -= msg.max()  # normalization
			msg = damping * mu[(i, j)] + (1 - damping) * msg
			delta = max(delta, float(np.max(np.abs(msg - mu[(i, j)]))))
			newMu[(i, j)] = msg
		mu = newMu  # synchronous update

		candidate = decide()
		value = globalPayoff(candidate, q, edges, f)
		if value > bestValue:
			best, bestValue = candidate, value
		if delta < tol:
			break

	return best, iterations
