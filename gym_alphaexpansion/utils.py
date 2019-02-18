import numpy as np
import operator


def negative_allowing_log_10(input):
    mask = input < 0
    output = np.clip(np.log10(np.abs(input)), 0, np.inf)
    output[mask] = np.negative(output[mask])
    return output


def abs_max_scaling(input):
    abs_max = np.abs(input).max()
    return np.divide(input, abs_max, where=abs_max != 0)


def apply_f(a, f):
    if isinstance(a, list):
        return map(lambda t: apply_f(t, f), a)
    else:
        return f(a)


def apply_attr_getter(input):
    return operator.attrgetter('tile')(input)


vectorized_tile_getter = np.vectorize(apply_attr_getter)
