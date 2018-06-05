#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import numpy as np

from operators import laplace


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
parser.add_argument('-d', '--space-order', type=int, default=2)
parser.add_argument('-k', '--skew-factor', type=int, default=2)
parser.add_argument('-p', '--print', action='store_true')

args = parser.parse_args()


def compare(arr1, arr2):
    return (np.amax(abs(np.subtract(arr1, arr2))),
            np.equal(arr1.data, arr2.data).all(),
            np.allclose(arr1.data, arr2.data, atol=10e-3))


def check_control(result):
    eprint("result nonzero-count: %d" % np.count_nonzero(result.data))
    if not args.no_tiling:
        eprint("Running non-blocking code as control...")
        no_bl, _ = operator(shape, 2, dle='noop', iterations=args.timesteps,
                            space_order=args.space_order)

        if args.print:
            print(result)
            print(no_bl)

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
    else:
        if args.print:
            print(result)
        eprint("NO MATCH: control not run")


if __name__ == '__main__':
    eprint(args)

    if args.laplace:
        operator = laplace
        shape = args.shape or (512, 512, 512)
        blockshape = args.blockshape or (16, 16, 16, 16)

        dle_args = {'blockinner': args.blockinner}
        if not args.autotune:
            dle_args['blockshape'] = blockshape

        if args.no_tiling:
            eprint("Tiling: None")
            eprint("Iteration space: %s" % str(shape))
            result, _ = operator(shape, 2, dle='noop', iterations=args.timesteps,
                            space_order=args.space_order)
        elif args.space_tiling:
            eprint("Tiling: Space")
            eprint("Iteration space: %s" % str(shape))
            eprint("Block shape: %s" % str(blockshape))
            result, _ = operator(shape, 2, autotune=args.autotune, iterations=args.timesteps,
                            dle=('blocking,openmp', dle_args),
                            space_order=args.space_order)
        elif args.time_tiling:
            eprint("Tiling: Time")
            eprint("Iteration space: %s" % str(shape))
            eprint("Block shape: %s" % str(blockshape))
            result, _ = operator(shape, 2, autotune=args.autotune, skew_factor=args.skew_factor, dse='skewing',
                    dle=('blocking,openmp', dle_args),
                            iterations=args.timesteps, space_order=args.space_order)
        else:
            raise ValueError("No tiling selected")

        check_control(result)
    elif args.acoustic:
        # TODO
        pass
    else:
        raise ValueError("Unknown operator")
