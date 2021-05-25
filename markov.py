from functools import cache
from sys import path
import numpy as np
from typing import List

import multi_distr

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

class MarkovSolver:
    def __init__(self, chain) -> None:
        chain = np.array(chain)
        assert len(chain.shape) == 2
        assert chain.shape[0] == chain.shape[1]
        self.eig, self.ev = np.linalg.eig(chain)

        self.call_vec = np.vectorize(self.call_proto)

        if np.isclose(np.linalg.det(self.ev), 0):
            print('PATHOLOGICAL MARKOV CHAIN')
            self.patho = pathological_solver(chain)
        else:
            self.patho = None
            self.coeffs = (np.linalg.pinv(self.ev) @ [1, 0, 0]) * self.ev[-1]
    
    def call_proto(self, n):
        if self.patho is not None:
            return self.patho(n)
        if n <= 0: return 0.0
        xp = self.eig**(n-1)
        return np.sum(self.coeffs * (self.eig - 1) * xp, dtype=np.float)

    def __call__(self, nvec) -> float:
        return self.call_vec(nvec)

    def longterm_rate(self):
        if self.patho is not None:
            return 0
        eig_abs = np.abs(self.eig)
        eig_abs[eig_abs == 1] = 0
        return -np.log(self.eig[np.argmax(eig_abs)])

def distr_hitter(mkv: MarkovSolver, multi: multi_distr.MultiDistr):
    @cache
    def hit(w):
        return np.sum(mkv(range(w+1)) * multi(w, range(w+1)), dtype=np.float)

    return np.vectorize(hit, otypes=[np.float])
