from sys import path
import numpy as np
from typing import List

# 5* pity (pathological case)
def pathological_e5s_pity(n):
    ret = np.array([0])
    return int((n==1) | (n==2)) * 1/2


def pathological_solver(chain):
    # Make the chain drain out the final state
    chain = np.array(chain)
    chain[-1, -1] = 0

    n_final = [None]

    def prob(n):
        if n <= 0:
            return 0
        if n_final[0] is not None and n >= n_final[0]:
            return 0.0
        matpow = np.linalg.matrix_power(chain, n)
        if matpow[-1, 0] == 0:
            n_final[0] = n
        return matpow[-1, 0]

    return np.vectorize(prob, otypes=[np.float])


# Solve a markov chain
def markov_soln(chain: List[List[float]]):
    chain = np.array(chain)
    assert len(chain.shape) == 2
    assert chain.shape[0] == chain.shape[1]

    eig, ev = np.linalg.eig(chain)
    if np.isclose(np.linalg.det(ev), 0):
        print('PATHOLOGICAL MARKOV CHAIN')
        return pathological_solver(chain)

    coeffs = (np.linalg.pinv(ev) @ [1, 0, 0]) * ev[-1]
    def p_an(n):
        if n <= 0: return 0.0
        xp = eig**(n-1)
        return np.sum(coeffs * (eig - 1) * xp)
    return np.vectorize(p_an, otypes=[np.float])
