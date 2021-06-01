import numpy as np
from scipy.interpolate import interp1d
from functools import cache

@cache
def e5s_pdf(w: int):
    def e5_phw(w):
        if 1 <= w <= 73:
            return .006
        elif 73 <= w <= 90:
            return .006 + (1-.006)/17 * (w-73)
        else:
            return 0
    return np.prod([1-e5_phw(w2) for w2 in range(w)]) * e5_phw(w)

@cache
def e4s_pdf(w: int):
    def e4_phw(w):
        if 1 <= w <= 8:
            return .051
        elif w == 9:
            return 13/23
        elif w == 10:
            return 1
        return 0
    return np.prod([1-e4_phw(w2) for w2 in range(w)]) * e4_phw(w)

@cache
def w5s_pdf(w):
    # pts = [[0, .009], [1, .07978], [7,.49], [10, .55], [13, .24], [16, .1], [28, 1]]
    # x, y = np.array(pts).T
    # cinterp = interp1d(x+62, y, kind='cubic')
    def w5_phw(w):
        if 1 <= w <= 62:
            return .009
        elif 62 <= w <= 76.5:
            return .009 + (1 - .009) / (76.5 - 62) * (w - 62)
        elif 76.5 < w:
            return 1
        return 0
    return np.prod([1-w5_phw(w2) for w2 in range(w)]) * w5_phw(w)



@cache
def w4s_pdf(w: int):
    def w4_phw(w):
        if 1 <= w <= 7:
            return .06
        elif 8 <= w <= 10:
            return [2/3, .9895, 1][w-8]
        return 0
    return np.prod([1-w4_phw(w2) for w2 in range(w)]) * w4_phw(w)

    