import numpy as np
import operator
from alphaexpansion import gamerules


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


def apply_tile_getter(input):
    return operator.attrgetter('tile')(input)


def apply_can_build(input, build, game):
    if not hasattr(input, 'build') and gamerules.BUILDING_DEFINITIONS[build]['tile'] & getattr(input, 'tile') \
            and gamerules.isAffordable(build, 0, game):
        return 1
    else:
        return 0


# tile_getter = np.frompyfunc(operator.attrgetter('tile'), 1, 1)
tile_getter = np.vectorize(apply_tile_getter)

can_build = np.vectorize(apply_can_build)
