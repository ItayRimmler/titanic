from exceptions import tooSmall
import random as r

def Batch(x, y, size):
    try:

        size = round(size)

        if size > x.size(0) or size < 1:
            raise tooSmall(x.size(0), size)

    except tooSmall as e:
        size = e.b

    finally:
        ind = sorted(r.sample(range(0, x.size(0)), size))
        x = x[ind]
        y = y[ind]
        return x, y, size
