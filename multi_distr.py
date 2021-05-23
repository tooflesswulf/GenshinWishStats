import numpy as np
from typing import Callable, List, Union
from functools import cache
from scipy.interpolate import interp1d

from wish_distr import e_5s, e_4s, w_5s, w_4s
from markov import markov_soln

class MultiDistr:
    def __init__(self, base_distr: Callable[[int], float], n_max: int):
        self.base = base_distr
        self.n_max = n_max + 1
        self.distr = np.array([base_distr(n) for n in range(self.n_max)])

        self.norm = np.sum(self.distr)

        self.mean = np.sum(np.arange(self.n_max) * self.distr)
        en2 = np.sum((np.arange(self.n_max)**2) * self.distr)
        self.std = np.sqrt(en2 - self.mean * self.mean)

    # P(w | n). w = wishes, n = num hits/num iterations
    def _pwn(self, w: int, n_iter: int=1, force_exact=False) -> float:
        if force_exact or n_iter <= 20:
            return self._cached_pwn(w, n_iter)
        
        mu = self.mean * n_iter
        sigma = self.std * np.sqrt(n_iter)
        return np.exp(-0.5*(mu - w)**2 / sigma/sigma) / np.sqrt(2*np.pi) / sigma

    def __call__(self, w: Union[List[int], int], n_iter: int, force_exact=False) -> float:
        if isinstance(w, int):
            return self._pwn(w, n_iter, force_exact)
        return np.array([self._pwn(wi, n_iter, force_exact) for wi in w])


    @cache
    def _cached_pwn(self, w: int, n: int) -> float:
        if n == 0: return 0
        if n == 1:
            return self.base(w)

        iter_lim = max(self.n_max, w+1)
        return sum([self.base(z) * self._cached_pwn(w-z, n-1) for z in range(iter_lim)])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    oracle = MultiDistr(e_4s, 10)

    print(oracle(3, 1))
    print(oracle(range(10), 1))

    e5s_mkv = [[0,0,0],[1/2,0,0],[1/2,1,1]]
    e4s_mkv = [[1/3,2/3,0],[1/2,0,0],[1/6,1/3,1]]
    w4s_mkv = [[3/5,4/5,0],[1/4,0,0],[3/20,1/5,1]]

    # oot = markov_soln([[3/5,4/5,0],[1/4,0,0],[3/20,1/5,1]])
    oot = markov_soln(e4s_mkv)

    print(oot(range(10)))
    print([oot(w) for w in range(10)])

    # print(oot(232342343))


    # # Distribution of Aw
    # @cache
    # def fourstar_hit(w):
    #     return sum([markov_soln(n) * oracle.p_iter(w, n) for n in range(w+1)])

    # target_fourstar = MultiDistr(fourstar_hit, 1000)
    # print(f'mean: {target_fourstar.mean}')
    # print(f'std:  {target_fourstar.std}')

    # mr = round(target_fourstar.mean)
    # perc = sum([target_fourstar.p_iter(w, 1) for w in range(mr+1)])
    # print(f'mean {mr}({perc})')

    # cdf = np.cumsum([target_fourstar.p_iter(w, 1) for w in range(150)])
    # print(f'50th perc: {np.argmin(np.abs(cdf - .5))}')
    # print(f'80th perc: {np.argmin(np.abs(cdf - .8))}')
