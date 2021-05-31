import numpy as np
import scipy.special
from typing import Callable
from functools import cache
import util

from wish_distr import e_5s, e_4s, w_5s, w_4s
from markov import MarkovSolver, distr_hitter

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
    @util.vectorize(otypes=[np.float])
    def __call__(self, w: int, n_iter: int=1, force_exact=False) -> float:
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

        # iter_lim = max(self.n_max, w+1)
        iter_lim = min(self.n_max, w+1)
        return sum([self.base(z) * self._cached_pwn(w-z, n-1) for z in range(iter_lim)])


# Solution only works for reversible markov chains.
class DupeTargMulti:
    def __init__(self, mkv: MarkovSolver, multi: MultiDistr, x_inflim: int) -> None:
        assert mkv.patho is None, 'DupeTargMulti works only for reversible markov chains.'
        l2 = mkv.longterm_rate()
        mu = multi.mean
        s = multi.std
        self.m2 = np.sqrt(mu*mu/s/s + 2*l2)/s - mu/s/s

        self.hitter = distr_hitter(mkv, multi)
        xr = np.arange(x_inflim, x_inflim+10)

        inflim_check = self.hitter(xr) / np.exp(-self.m2*xr) / self.m2
        chk_err = np.amax(inflim_check) - np.amin(inflim_check)
        assert chk_err <= 1e-10, f'x_inflim is too small. {chk_err}'
        self.c = np.mean(inflim_check)
        print(self.c)
        self.dt = 1 - self.c*self.m2 / (1-np.exp(-self.m2))

        xr = np.arange(x_inflim)
        # pt = (xr == 0).astype(int) * self.dt
        self.delts = self.hitter(xr) - self.expconv(xr, 1)
        derr_locs = np.cumsum(np.abs(self.delts[::-1])) > 1e-3
        ix_rev = np.where(derr_locs)[0][0]
        self.derr_len = len(self.delts) - ix_rev

        self.delt_multi = MultiDistr(lambda x: self.hitter(x) - self.expconv(x, 1), self.derr_len)
    
    def expconv(self, x, n):
        x = np.atleast_1d(x)
        zarg = -self.c * self.m2 / self.dt
        hyp = scipy.special.hyp2f1(1-n, 1+x, 2, zarg)
        nz = n * (self.dt**(n-1))
        exp = self.c * self.m2 * np.exp(-self.m2*x)
        ret = nz * exp * hyp

        ret[x==0] -= self.dt**n
        return ret
    
    # convde should take scalar arguments
    @cache
    def convde(self, w, dn, en):
        if dn == 0:
            return self.expconv(w, en)
        if en == 0:
            return self.delt_multi(w, dn)
        iterlim = min(w+1, self.derr_len*dn)
        z = np.arange(iterlim)
        return np.sum(self.delt_multi(z, dn) * self.expconv(w - z, en), dtype=np.float)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    e5s_mkv = [[0,0,0],[1/2,0,0],[1/2,1,1]]
    e4s_mkv = [[1/3,2/3,0],[1/2,0,0],[1/6,1/3,1]]
    w4s_mkv = [[3/5,4/5,0],[1/4,0,0],[3/20,1/5,1]]

    e4s_mult = MultiDistr(e_4s, 10)

    # testing multi distr
    x = np.arange(50)
    plt.plot(e4s_mult(19, x))
    # plt.plot(e4s_mult(x, 1))
    # plt.plot(e4s_mult(x, 2))
    # plt.plot(e4s_mult(x, 3))
    # plt.plot(e4s_mult(x, 4))
    # plt.plot(e4s_mult(x, 5))
    # plt.plot(e4s_mult(x, 6))
    plt.show()

    # e4s_mdist = MarkovSolver(e4s_mkv)
    # e4s_targ = distr_hitter(e4s_mdist, e4s_mult)

    # e4s_multi_targ = DupeTargMulti(e4s_mdist, e4s_mult, 250)
    # print(e4s_multi_targ.derr_len)


    # print(e4s_targ(x))
    # x = np.arange(100)
    # plt.plot(x, e4s_targ(x))
    # plt.show()

