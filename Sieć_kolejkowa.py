import math
from mpmath import nsum, inf
import numpy as np

class BcmpNetworkOpen(object):
	def __init__(self, R, N, k, mi_matrix, p, m, types, epsilon):
		self.R = R
		self.N = N
		self.k = k
		self.mi_matrix = mi_matrix
		self.m = m
		self.types = types
		self.epsilon = epsilon
		self.e = self.calculate_e_ir(p)
		self.lambda_r = np.array([epsilon for _ in range(self.R)])

	# TODO: To jest zrobione dla sieci zamkniętej, trzeba przerobić na otwartą, ale nie umiem :c
	def calculate_e_ir(self, p):
		tempMatrix = np.zeros((self.N, self.N))
		row_list = []
		for i in range(0, len(p)):
			matList = []
			for j in range(0, len(p)):
				if i == j:
					matList.append(p[i])
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

	# TODO :Summation method liczy lambda_r, nie jestem pewna czy można tak zrobic dla sieci otwartej
	# TODO: jeśli nie, to trzeba się dowiedzieć jak policzyć lambda_r
	def calculate_fix_ir(self, i, r):
		if self.mi_matrix[i, r] == 0:
			return 0

		if self.types[i] in frozenset([1, 2, 4]) and self.m[i] == 1:
			return (self.e[i, r] / self.mi_matrix[i, r]) / (1. - ((sum(self.k) - 1.) / sum(self.k)) * self.calculate_ro_i(i))
		elif self.types[i] == 1 and self.m[i] > 1:
			sum1 = self.e[i, r] / self.mi_matrix[i, r]
			mul1 = (self.e[i, r] / (self.m[i] * self.mi_matrix[i, r])) / (
					1. - (((sum(self.k) - self.m[i] - 1.) / (sum(self.k) - self.m[i])) * self.calculate_ro_i(i)))
			return sum1 + mul1 * self.calculate_Pmi(i)
		elif self.types[i] == 3:
			return self.e[i, r] / self.mi_matrix[i, r]

	def calculate_single_iteration(self):
		s = 0.
		for r in range(self.R):
			for i in range(self.N):
				fix_ir = self.calculate_fix_ir(i, r)
				s += fix_ir

			self.lambda_r[r] = (float(self.k[r]) / s) if s != 0 else 0

	def calculate_lambda_r_sum_method(self):
		error = self.epsilon + 1
		i = 0
		while error > self.epsilon:
			if i > 100:
				break

			old_lambda_r = np.copy(self.lambda_r)
			self.calculate_single_iteration()

			err = ((self.lambda_r - old_lambda_r) ** 2).sum()
			error = math.sqrt(err)
			i += 1

	def get_lambda_matrix(self):
		lambda_matrix = np.zeros((self.N, self.R))
		for r in range(self.R):
			for i in range(self.N):
				lambda_matrix[i, r] = self.calculate_ro_ir(i, r) * self.m[i] * self.mi_matrix[i, r]
		return lambda_matrix

	def calculate_ro_ir(self, i, r):
		if self.mi_matrix[i, r] == 0:
			return 0
		elif self.m[i] >= 1 and self.types[i] == 1:
			return self.lambda_r[r] * self.e[i, r] / (self.m[i] * self.mi_matrix[i, r])
		elif self.types[i] in frozenset([2,3,4]):
			return self.lambda_r[r] * self.e[i, r] / (self.mi_matrix[i, r])

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
		lambda_matrix = self.get_lambda_matrix()
		for r in range(self.R):
			for i in range(self.N):
				if self.types[i] == 1:
					K_matrix[i, r] = self.m[i] * self.calculate_ro_ir(i, r) + self.calculate_ro_ir(i, r)/(1- self.calculate_ro_i(i)) * self.calculate_Pmi(i)
				elif self.types[i] == 3 and self.mi_matrix[i, r] != 0:
					K_matrix[i, r] = lambda_matrix[i, r] / self.mi_matrix[i, r]
				else:
					K_matrix[i, r] = 0
		return K_matrix

	# TODO : nie wiem czy chodzi o to K które się wyliczy w get_K_matrix, czy o to K ktore się podaje w konstruktorze
	def calculate_k_i(self, i):
		K_matrix = self.get_K_matrix()
		return sum([K_matrix[i, r] for r in range(self.R)])

	def get_T_matrix(self):
		T_matrix = np.zeros((self.N, self.R))
		lambda_matrix = self.get_lambda_matrix()
		K_matrix = self.get_K_matrix()
		for r in range(self.R):
			for i in range(self.N):
				if self.e[i, r] or lambda_matrix[i, r] == 0:
					T_matrix[i, r] = 0
				else:
					T_matrix[i, r] = K_matrix[i, r] / lambda_matrix[i, r]
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
		lambda_matrix = self.get_lambda_matrix()
		for r in range(self.R):
			for i in range(self.N):
				Q_matrix[i, r] = lambda_matrix[i, r] * W_matrix[i,r]
		return Q_matrix

	#-----------POCZĄTEK: To są parametry dla pojedyńczego systemu M/M/FIFO/inf
	def no_request_prob(self, i): #czy tutaj ro to jest calculate_ro_i ?
		suma = 0
		ro = self.calculate_ro_i(i)
		for k in range(self.m[i]-1):
			suma += ((pow(ro, k) / math.factorial(k)) +
			(pow(ro, self.m[i]) / (math.factorial(self.m[i] - 1) *
								   math.factorial(math.ceil(self.m[i] - ro))))) #Tutaj jest konieczne zaokraglenie, czy tak ma byc?
		return 1 / suma

	def s_busy_prob(self, s, i):
		ro = self.calculate_ro_i(i)
		if s > self.m[i]-1 or s < 1:
			raise Exception("Invalid number of channels")
		if ro > self.m[i]:
			raise Exception("Ro cannot be higher than m")
		return self.no_request_prob(i)*(ro**s)/math.factorial(s)

	def lenght_prob(self, r, i):
		ro = self.calculate_ro_i(i)
		if r < 0:
			raise Exception("r must be > 0")
		if not ro < self.m[i]:
			raise Exception("Ro must be < m")
		return self.no_request_prob(i)*(ro**(self.m[i]+r))/(math.factorial(self.m[i])*(self.m[i]**r))

	def avrg_requests(self, i):
		ro = self.calculate_ro_i(i)
		return ro + self.no_request_prob(i)*((ro**(self.m[i]+1))/(((self.m[i]-ro)**2)*math.factorial(self.m[i]-1)))

	# -----------------------------------------------------------

	def temp_1(self, i): # big fraction
		ro = self.calculate_ro_i(i)
		return ((pow(ro, self.m[i]+1))/((pow(self.m[i]-ro,2))*math.factorial(self.m[i]-1)))/self.temp_2(i)

	def temp_2(self, i): # sum in denominator
		ro = self.calculate_ro_i(i)
		return sum(pow(ro,i)/math.factorial(i) + pow(ro, self.m[i])/(math.factorial(self.m[i]-1)*(self.m[i]-ro))for i in range(self.m[i]-1))

	def avrg_length(self, i):
		ro = self.calculate_ro_i(i)
		if not ro < self.m[i]:
			raise Exception("Ro must be < m")
		return nsum(lambda r: r*self.lenght_prob(r, i)*self.temp_1(i), [0, inf])

	# ---------------------------------------------------------
	#-----------KONIEC: To są parametry dla pojedyńczego systemu M/M/FIFO/inf

def main():
	#service times
	mi = np.array([[67., 67., 67., 67.],
						[8., 8., 8., 8.],
						[60., 60., 60., 60.],
						[8.33, 8.33, 8.33, 8.33],
						[12., 12., 12., 12.],
						[0.218, 0.218, 0.218, 0.218]])
	R = 4  # liczba klas
	N = 6  # liczba systemow
	p1 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
				   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
				   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
				   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

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

	classes = [p1, p2, p3, p4]
	epsilon = 1e-05

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
		epsilon=epsilon
	)

	#print(bcmpNetworkOpen.no_request_prob(1))# To oznacza prawdopodobieństwo p_0 dla systemu 1, czyli prawdopodobieństwo, że 0 kanały w systemie 1 sa zajęte
	#print(bcmpNetworkOpen.s_busy_prob(2, 1))# Prawdopodobieństwo, że 2 kanały w systemie 1 sa zajęte
	#print(bcmpNetworkOpen.lenght_prob(2, 1))# Prawdopodobieństwo, że długość kolejki w systemie 1 jest równa 2
	#print(bcmpNetworkOpen.avrg_requests(1))#Średnia liczba zgłoszeń w systemie 1
	#print(bcmpNetworkOpen.avrg_length(1))#Średnia długość kolejki dla systemu 1

	bcmpNetworkOpen.calculate_lambda_r_sum_method()
	print('lambda')
	print(bcmpNetworkOpen.get_lambda_matrix())
	print('lambda_r')
	print(bcmpNetworkOpen.lambda_r)
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
