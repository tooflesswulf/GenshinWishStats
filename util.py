import numpy as np
import functools

def vectorize(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return vectorize_noarg(*args, **kwargs)
    else:
        def f(pyfunc):
            return vectorize_noarg(pyfunc, **kwargs)
        return f

class vectorize_noarg(np.vectorize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)


# def vectorize(*args, **kwargs):
#     if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
#         return np.vectorize(args[0])
#     else:
#         return functools.partial(np.vectorize, **kwargs)
