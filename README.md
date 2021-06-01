# GenshinWishStats
Wish statistics math for Genshin Impact. Code is for computing quantities described in the writeup (put link here). See Jupyter Notebook for more.

## Dependencies
Requires Python3.9 because `@functools.cache` decorator is vital to the code's runtime not becoming a combinatorial mess.

`numpy` and `scipy.special.hyp2f1` are required for the math.

## Running the code

Details are in the Jupyter notebook [here](wish_plots.ipynb).

Naming convention is (e5s-, e4s-, w5s-, w4s-) for event banner 4/5 star or weapon banner 4/5 star. I will call them `prefix`. 


```python
prefix_pdf(w)  # P(X = w)
prefix_multi = MultiDistr(prefix_pdf)  # Object for section 2.1.2
prefix_multi(w, m)  # P(Xm = w)
prefix_markov = [[...]]]  # Markov chain dynamics go here.
```


Functions:
- `{prf}_pdf(w)` gives P(X=w)
- `{prf}_multi = MultiDistr({prf}_pdf(w))`
- c





