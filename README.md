# GenshinWishStats
Wish statistics math for Genshin Impact. Code is for computing quantities described in the writeup (put link here). See Jupyter Notebook for more.

## Dependencies
Requires Python3.9 because `@functools.cache` decorator is vital to the code's runtime not becoming a combinatorial mess.

`numpy` and `scipy.special.hyp2f1` are required for the math.

## Running the code

Details are in the Jupyter notebook [here](wish_plots.ipynb).

Naming convention is (e5s-, e4s-, w5s-, w4s-) for event banner 4/5 star or weapon banner 4/5 star. The example code below uses e4s.


```python
from multi_distr import MultiDistr
from markov import MarkovSolver, distr_hitter
from wish_distr import e4s_pdf

e4s_pdf(w)  # P(X = w)
e4s_multi = MultiDistr(e4s_pdf)  # Object for section 2.1.2
e4s_multi(w, m)  # P(Xm = w)
e4s_markov = [[...]]]  # Markov chain dynamics go here.

# Does not work for e5s, instead use MultiDistr(distr_hitter(MarkovSolver(e5s_markov), e5s_multi)).
e4s_target_multi = DupeTargMulti(MarkovSolver(e4s_markov), e4s_multi, 500)  # Can take long time to initialize
e4s_target_multi(w, m)  # P(hit m-th copy of target on w-th wish)
```

## Notes
### DupeTargMulti
- Does not work with e5s. It is faster to solve exactly with `MultiDistr(distr_hitter(MarkovSolver(e5s_markov), e5s_multi))`
- Needs some 'infinity' for which behavior is approximately exponential.
- e4s and w4s: set to 500
- w5s: set to 2000 (very slow)



