import math
from mpmath import nsum, inf
import numpy as np

class BcmpNetworkOpen(object):
	def __init__(self, R, N, k, mi_matrix, p, m, types, lambdas):
		self.R = R
		self.N = N
		self.k = k
		self.mi_matrix = mi_matrix
		self.p = p
		self.m = m
		self.types = types
		self.e = self.calculate_e_ir()
		self.lambdas = lambdas
		self.lambda_matrix = self.get_lambda_matrix()


	# TODO: To jest zrobione dla sieci zamkniętej, trzeba przerobić na otwartą, ale nie umiem :c
	def calculate_e_ir(self):
		tempMatrix = np.zeros((self.N, self.N))
		row_list = []
		for i in range(0, len(self.p)):
			matList = []
			for j in range(0, len(self.p)):
				if i == j:
					matList.append(self.p[i])
				else:
					matList.append(tempMatrix)
			row = np.concatenate(matList)
			row_list.append(row)
		finishedMatrix = np.column_stack(row_list)

		a_minus = ([0] + [1] * (self.N - 1)) * (self.R)
		A = finishedMatrix.T - np.diagflat(a_minus)
		b = ([1] + [0] * (self.N - 1)) * (self.R)
		ret, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

		visit_ratios = ret.reshape(self.R, self.N).T#+ p[0] # tu jakoś dodać p_0,ir
		return visit_ratios

    # Tego tak średnio jestem pewna
	def get_lambda_matrix(self):
		lambda_r = []
		lambda_matrix = np.zeros((self.N, self.R))
		for p_we, lambda_we in zip(self.p, self.lambdas):
			A = np.zeros((p_we.shape[1] - 2, p_we.shape[1] - 2))
			np.fill_diagonal(A, 1)
			b = lambda_we * p_we[0][1:-1]

			for row_i, p_row in enumerate(p_we):
				if row_i >= 1 and row_i <= A.shape[0]:
					A[row_i - 1] = A[row_i - 1] - p_we[row_i][1:-1]

			lambda_r.append(np.linalg.solve(np.transpose(A), b))
		for i in range(self.N):
			for r in range(self.R):
				if i < self.R:
					lambda_matrix[i][r] = lambda_r[r][i]
				else:
					lambda_matrix[i][r] = lambda_r[r][i - 2]

		return lambda_matrix

	def calculate_ro_ir(self, i, r):
		if self.mi_matrix[i, r] == 0:
			return 0
		elif self.m[i] >= 1 and self.types[i] == 1:
			return self.lambda_matrix[i, r] / (self.m[i] * self.mi_matrix[i, r])
		elif self.types[i] in frozenset([2,3,4]):
			return self.lambda_matrix[i, r] / (self.mi_matrix[i, r])

	def calculate_ro_i(self, i):
		return sum([self.calculate_ro_ir(i, r) for r in range(self.R)])

	def get_ro_matrix(self):
		ro_matrix = np.zeros((self.N, self.R))
		for r in range(self.R):
			for i in range(self.N):
				ro_matrix[i, r] = self.calculate_ro_ir(i, r)
		return ro_matrix

	def calculate_Pmi(self, i):
		ro_i = self.calculate_ro_i(i)
		if self.m[i] == 0:
			return 1
		elif ro_i == 0:
			return 0

		mul = ((self.m[i] * ro_i) ** self.m[i]) / (math.factorial(self.m[i]) * (1 - ro_i))
		den1 = sum([((self.m[i] * ro_i) ** k) / math.factorial(k) for k in range(self.m[i])])
		den2 = (((self.m[i] * ro_i) ** self.m[i]) / math.factorial(self.m[i])) * (1. / (1 - ro_i))

		return mul / (den1 + den2)

	def get_K_matrix(self):
		K_matrix = np.zeros((self.N, self.R))
		for r in range(self.R):
			for i in range(self.N):
				if self.types[i] == 1:
					K_matrix[i, r] = self.m[i] * self.calculate_ro_ir(i, r) + self.calculate_ro_ir(i, r)/(1- self.calculate_ro_i(i)) * self.calculate_Pmi(i)
				elif self.types[i] == 3 and self.mi_matrix[i, r] != 0:
					K_matrix[i, r] = self.lambda_matrix[i, r] / self.mi_matrix[i, r]
				else:
					K_matrix[i, r] = 0
		return K_matrix

	# TODO : nie wiem czy chodzi o to K które się wyliczy w get_K_matrix, czy o to K ktore się podaje w konstruktorze
	def calculate_k_i(self, i):
		K_matrix = self.get_K_matrix()
		return sum([K_matrix[i, r] for r in range(self.R)])

	def get_T_matrix(self):
		T_matrix = np.zeros((self.N, self.R))
		K_matrix = self.get_K_matrix()
		for r in range(self.R):
			for i in range(self.N):
				if self.e[i, r] or self.lambda_matrix[i, r] == 0:
					T_matrix[i, r] = 0
				else:
					T_matrix[i, r] = K_matrix[i, r] / self.lambda_matrix[i, r]
		return T_matrix

	def get_W_matrix(self):
		W_matrix = np.zeros((self.N, self.R))
		T_matrix = self.get_T_matrix()
		for r in range(self.R):
			for i in range(self.N):
				if self.mi_matrix[i, r] == 0:
					W_matrix[i, r] = 0
				else:
					W_ir = T_matrix[i, r] - 1 / self.mi_matrix[i, r]
					W_matrix[i, r] = W_ir if W_ir > 0 else 0
		return W_matrix

	def get_Q_matrix(self):
		Q_matrix = np.zeros((self.N, self.R))
		W_matrix = self.get_W_matrix()
		for r in range(self.R):
			for i in range(self.N):
				Q_matrix[i, r] = self.lambda_matrix[i, r] * W_matrix[i,r]
		return Q_matrix

def main():
	p1 = np.array([[0.01, 0.99, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.01, 0.99, 1.0, 0.0, 0.0],
				   [0.0, 0.0, 0.01, 0.99, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 0.01, 0.99, 1.0],
				   [0.0, 0.0, 0.0, 0.0, 0.01, 0.99],
				   [0.0, 0.0, 0.0, 0.0, 0.99, 0.01]])

	p2 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
				   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
				   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

	p3 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
				   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
				   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

	p4 = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
				   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
	#service times
	mi = np.array([[67., 67., 67., 67.],
						[8., 8., 8., 8.],
						[60., 60., 60., 60.],
						[8.33, 8.33, 8.33, 8.33],
						[12., 12., 12., 12.],
						[0.218, 0.218, 0.218, 0.218]])
	R = 4  # liczba klas
	N = 6  # liczba systemow
	classes = [p1, p1, p1, p1]
	lambda_we_r = [1.8, 1.2, 1, 1.2]

	# node types
	# 1 - FIFO
	# 2 - PS
	# 3 - IS
	# 4 - LIFO
	types = [1, 3, 1, 1, 1, 1]
	m = [1, 1, 1, 4, 2, 66]
	K1 = [250, 144, 20, 20]  # czym to jest? czy to jest to samo co K_ir?

	bcmpNetworkOpen = BcmpNetworkOpen(
		R=R,
		N=N,
		k=K1,
		mi_matrix=mi,
		p=classes,
		m=m,
		types=types,
		lambdas = lambda_we_r,
	)

	print('lambda')
	print(bcmpNetworkOpen.get_lambda_matrix())
	print('e')
	print(bcmpNetworkOpen.e)
	print('ro')
	print(bcmpNetworkOpen.get_ro_matrix())
	print('K')
	print(bcmpNetworkOpen.get_K_matrix())
	print('Q')
	print(bcmpNetworkOpen.get_Q_matrix())
	print('W')
	print(bcmpNetworkOpen.get_W_matrix())
	print('T')
	print(bcmpNetworkOpen.get_T_matrix())


main()