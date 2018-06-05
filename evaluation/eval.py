#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import numpy as np

from operators import laplace, acoustic


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


parser = argparse.ArgumentParser(description="Evaluation script for time-tiling in Devito")
tiling = parser.add_mutually_exclusive_group(required=True)
tiling.add_argument('-N', '--no-tiling', action='store_true')
tiling.add_argument('-S', '--space-tiling', action='store_true')
tiling.add_argument('-T', '--time-tiling', action='store_true')

operator = parser.add_mutually_exclusive_group(required=True)
operator.add_argument('-L', '--laplace', action='store_true')
operator.add_argument('-W', '--acoustic', action='store_true')

parser.add_argument('-a', '--autotune', action='store_true')
parser.add_argument('-s', '--shape', nargs='+', type=int)
parser.add_argument('-b', '--blockshape', nargs='+', type=int)
parser.add_argument('-i', '--blockinner', action='store_true')
parser.add_argument('-t', '--timesteps', type=int, default=16)
parser.add_argument('-so', '--space-order', type=int, default=4)
parser.add_argument('-to', '--time-order', type=int, default=2)
parser.add_argument('-k', '--skew-factor', type=int, default=0)

args = parser.parse_args()


def compare(arr1, arr2):
    return (np.amax(abs(np.subtract(arr1, arr2))),
            np.equal(arr1.data, arr2.data).all(),
            np.allclose(arr1.data, arr2.data, atol=10e-3))


def check_control(result, no_bl):
    eprint("result nonzero-count: %d" % np.count_nonzero(result.data))
    eprint("untile nonzero-count: %d" % np.count_nonzero(untile.data))

    i = args.timesteps
    l, r = 0, i
    while l < r - 1:
        comp = compare(result[i], no_bl[i])
        eprint("t=%d: max diff: %f, np.eq: %s, close: %s" %
                (i, comp[0], comp[1], comp[2]))
        if comp[2] and not np.isnan(comp[0]):
            l = i
        else:
            r = i
        i = int((l + r) / 2)


if __name__ == '__main__':
    eprint(args)

    if args.laplace:
        operator = laplace
        shape = args.shape or (512, 512, 512)
    elif args.acoustic:
        operator = acoustic
        shape = args.shape or (512, 512, 512)
        # TODO
        pass
    else:
        raise ValueError("Unknown operator")

    kwargs = {
        'shape': shape,
        'iterations': args.timesteps,
        'space_order': args.space_order,
        'time_order': 2,
        'autotune': args.autotune,
    }

    dse = 'advanced'
    dle = ('blocking,openmp', {'blockinner': args.blockinner})

    if args.blockshape:
        dle[1]['blockshape'] = args.blockshape

    if args.no_tiling:
        dle = 'noop'
    elif args.space_tiling:
        pass
    elif args.time_tiling:
        dse = 'skewing'
    else:
        raise ValueError("No tiling selected")

    result = operator(dse=dse, skew_factor=args.skew_factor, dle=dle, **kwargs)
    if args.no_tiling:
        eprint("MATCH: NONE: non-tiled code run")
    else:
        untile = operator(dse='advanced', skew_factor=0, dle='noop', **kwargs)
        check_control(result, untile)
