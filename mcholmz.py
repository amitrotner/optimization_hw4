import numpy as np

def modifiedChol(G):
	"""
	****** Thanks to Lior Cohen for translating the code to Python! ******

	Perform Modified Cholesky Factorization according to Michael Zibulevsky's code.

	Notes from the original algorithm in matlab:

	%  Given a symmetric matrix G, find a vector e of "small" norm and
	%  lower triangular matrix L, and vector d such that  G+diag(e) is Positive Definite, and 
	%
	%      G+diag(e) = L*diag(d)*L'
	%
	%  Also, calculate a direction pneg, such that if G is not PSD, then
	%
	%      pneg'*G*pneg < 0
	%
	%  Reference: Gill, Murray, and Wright, "Practical Optimization", p111.
	%  Author: Brian Borchers (borchers@nmt.edu)
	%  Modification (acceleration):  Michael Zibulevsky   (mzib@cs.technion.ac.il)
	%

	:param G: a symmetric Matrix to decompose.
	"""
	assert np.allclose(G, G.T, atol=1e-8)

	#	initialze variables:
	n = G.shape[0]
	eps = np.finfo(float).eps

	diagG = np.diag(G)
	C = np.diag(diagG)

	gamma = np.max(diagG)
	zi = np.max(G-C)
	nu = np.max([1, np.sqrt(n**2 - 1)])
	beta2 = np.max([gamma, zi/nu, 1e-15])

	L = np.zeros((n,n))
	d = np.zeros((n,1))
	e = np.zeros((n,1))

	theta = np.zeros((n,1))

	#	Perform for first element seperately (to aviod if statements):
	j = 0
	ee = range(1, n)
	C[ee,j] = G[ee,j]
	theta[j] = np.max(np.abs(C[ee,j]))

	d[j] = np.max(np.array([ eps, np.abs(C[j,j]), (theta[j]**2)/beta2 ]).T)
	e[j] = d[j] - C[j,j]

	ind = [(i,i) for i in range(n)]
	for pos, e_i in zip(ind[j+1:], ee):
		C[pos]=C[pos]-(1.0/d[j])*(C[e_i,j]**2)

	#	Perform for 2 <= j <= n-1
	for j in range(1, n-1):
		bb = range(j)
		ee = range(j+1, n)

		L[j, bb] = np.divide(C[j, bb], d[bb].T)
		C[ee, j] = G[ee, j] - ((C[np.ix_(ee, bb)]) @ (L[j, bb].T))
		theta[j] = np.max(np.abs(C[ee,j]))

		d[j] = np.max(np.array([ eps, np.abs(C[j,j]), (theta[j]**2)/beta2 ]).T)
		e[j] = d[j] - C[j,j]

		for pos, e_i in zip(ind[j+1:], ee):
			C[pos]=C[pos]-(1.0/d[j])*(C[e_i,j]**2)

	#	Perform for last element seperately (to aviod if statements):
	j = n-1
	bb = range(j)
	ee = range(j+1, n)

	L[j, bb] = np.divide(C[j, bb], d[bb].T)
	C[ee, j] = G[ee, j] - ((C[np.ix_(ee, bb)]) @ (L[j, bb].T))
	theta[j] = 0

	d[j] = np.max(np.array([ eps, np.abs(C[j,j]), (theta[j]**2)/beta2 ]).T)
	e[j] = d[j] - C[j,j]

	#	Add ones on the diagonal:
	for pos in ind:
		L[ pos ] = 1

	return L, d, e


if __name__ == '__main__':
	G = np.array([[2, 6, 10],
		[6, 10, 14],
		[10, 14, 18]], dtype=np.float64)

	L, d, e = modifiedChol(G)
	Err = (L @ np.diag(d.flatten()) @ L.T) - G
	print('Difference between original and decomposed matrices:\n', Err)
