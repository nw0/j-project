from functools import reduce
from operator import mul

import numpy as np
from sympy import solve

from devito import Eq, Grid, Operator, TimeFunction, configuration, clear_cache


def operator(shape, time_order, space_order=2, skew_factor=0, autotune=True, iterations=16, **kwargs):
    configuration['skew_factor'] = skew_factor
    grid = Grid(shape=shape)
    spacing = 0.1
    a = 0.5
    c = 0.5
    dx2, dy2, dz2 = spacing**2, spacing**2, spacing**2
    dt = dx2 * dy2 * dz2 / (2 * a * (dx2 + dy2 + dz2))

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=time_order, space_order=space_order, save=iterations+1)
    # u.data[0, :] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    u.data[0, :] = np.linspace(1e-20, 2e-20, num=reduce(mul, shape)).reshape(shape)

    # Derive the stencil according to devito conventions
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2) - c * (u.dxl + u.dyl))
    stencil = solve(eqn, u.forward, rational=False)[0]
    op = Operator(Eq(u.forward, stencil), **kwargs)

    # Execute the generated Devito stencil operator
    op.apply(u=u, dt=dt, autotune=autotune, time=iterations-1)
    # from IPython import embed; embed()
    return u.data, op


def no_blocking(shape):
    configuration['skew_factor'] = 0
    return operator(shape, time_order=2, dle='noop')


def space_blocking(shape, blockshape):
    configuration['skew_factor'] = 0
    return operator(shape, time_order=2,
                    dle=('blocking,openmp', {'blockinner': True}))

def time_blocking(shape, blockshape):
    configuration['skew_factor'] = 2
    return operator(shape, time_order=2, dse='skewing',
                    dle=('blocking,openmp', {'blockinner': True}))


if __name__ == '__main__':
    shape = (300, 300, 300)
    blockshape = (32, 32, 32)

    print("Running time-tiling code")
    time_blocking(shape, blockshape)
    clear_cache()

    print("Running space-tiling code")
    space_blocking(shape, blockshape)
    clear_cache()

    print("Running code without tiling")
    no_blocking(shape)

    print("Ran successfully")
