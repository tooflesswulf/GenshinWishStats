import numpy as np
from typing import Callable
from functools import cache

import matplotlib.pyplot as plt

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
    def p_iter(self, w: int, n_iter: int, force_exact=False) -> float:
        if force_exact or n_iter <= 20:
            return self._cached_pwn(w, n_iter)
        
        mu = self.mean * n_iter
        sigma = self.std * np.sqrt(n_iter)
        return np.exp(-0.5*(mu - w)**2 / sigma/sigma) / np.sqrt(2*np.pi) / sigma


    @cache
    def _cached_pwn(self, w: int, n: int) -> float:
        if n == 0: return 0
        if n == 1:
            return self.base(w)

        iter_lim = max(self.n_max, w+1)
        return sum([self.base(z) * self._cached_pwn(w-z, n-1) for z in range(iter_lim)])

    def get_distr(self, n, force_exact=False):
        return np.array([self.p_iter(w, n, force_exact) for w in range(self.n_max * n + 1)])


def wish_distr(r: float, p: float, pity: int=75, hard: int=None) -> float:
    if hard is None: hard = pity+15

    def prob(n):
        if 1 <= n <= pity:
            return r * (1-r)**(n-1)
        elif pity < n < hard:
            return (1-r)**pity * p * (1-p)**(n - pity - 1)
        elif n == hard:
            return (1-r)**pity * (1-p)**(hard - pity - 1)
        else:
            return 0
    return prob


# Solution of markov chain with probability distribution of An
# Replace later w/ function to solve a markov chain
def markov_soln(n: int) -> float:
    if n == 0: return 0

    a2 = -.5 - 2 / (np.sqrt(13))
    a3= -.5 + 2 / (np.sqrt(13))
    l2 = (1+np.sqrt(13))/6
    l3 = (1-np.sqrt(13))/6
    # a2 = -.5 - 11/(4*np.sqrt(29))
    # a3 = -.5 + 11/(4*np.sqrt(29))
    # l2 = (3+np.sqrt(29)) / 10
    # l3 = (3-np.sqrt(29)) / 10

    return a2 * (l2-1) * l2**(n-1) + a3 * (l3-1) * l3**(n-1)




if __name__ == '__main__':
    # base = [.051*(1-.051)**n for n in range(9)] + [(1-.051)**9]
    # b_fn = lambda n: base[n-1] if 1<=n<=len(base) else 0
    # fourstar = MultiDistr(b_fn, len(base)+1)
    event_5star = wish_distr(.006, 0.32383924389327824, pity=75, hard=90)
    event_4star = wish_distr(.051, 1, pity=9, hard=10)
    weap_4star = wish_distr(.06, 1, pity=9, hard=10)
    oracle = MultiDistr(event_4star, 10)

    # Distribution of Aw
    @cache
    def fourstar_hit(w):
        return sum([markov_soln(n) * oracle.p_iter(w, n) for n in range(w+1)])

    target_fourstar = MultiDistr(fourstar_hit, 1000)
    print(f'mean: {target_fourstar.mean}')
    print(f'std:  {target_fourstar.std}')

    mr = round(target_fourstar.mean)
    perc = sum([target_fourstar.p_iter(w, 1) for w in range(mr+1)])
    print(f'mean {mr}({perc})')

    cdf = np.cumsum([target_fourstar.p_iter(w, 1) for w in range(150)])
    print(f'50th perc: {np.argmin(np.abs(cdf - .5))}')
    print(f'80th perc: {np.argmin(np.abs(cdf - .8))}')

    # plt.plot([target_fourstar.p_iter(w, 1) for w in range(100)])
    # plt.figure()
    # plt.plot([target_fourstar.p_iter(w, 2) for w in range(300)])
    # plt.figure()
    # plt.plot([target_fourstar.p_iter(w, 3) for w in range(500)])
    # plt.show()
