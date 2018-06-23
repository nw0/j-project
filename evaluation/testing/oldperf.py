#!/usr/bin/env python3
import argparse
import logging
import os
import sys

from apply import operator


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


parser = argparse.ArgumentParser(description="Evaluation script for time-tiling in Devito")
tiling = parser.add_mutually_exclusive_group(required=True)
tiling.add_argument('-N', '--no-tiling', action='store_true')
tiling.add_argument('-S', '--space-tiling', action='store_true')
tiling.add_argument('-T', '--time-tiling', action='store_true')
parser.add_argument('-a', '--autotune', action='store_true')
parser.add_argument('-s', '--shape', nargs='+', type=int)
parser.add_argument('-b', '--blockshape', nargs='+')
parser.add_argument('-r', '--threads', type=int, default=8)
parser.add_argument('-i', '--blockinner', action='store_true')

args = parser.parse_args()

shape = args.shape or (384, 384, 384)
blockshape = args.blockshape or (32, 32, 32, 32)

if args.no_tiling:
    eprint("Tiling: None")
    eprint("Iteration space: %s" % str(shape))
    operator(shape, 2, dle='noop')
elif args.space_tiling:
    eprint("Tiling: Space")
    eprint("Iteration space: %s" % str(shape))
    eprint("Block shape: %s" % str(blockshape))
    operator(shape, 2, autotune=args.autotune,
             dle=('blocking,openmp', {'blockinner': args.blockinner}))
elif args.time_tiling:
    eprint("Tiling: Time")
    eprint("Iteration space: %s" % str(shape))
    eprint("Block shape: %s" % str(blockshape))
    operator(shape, 2, autotune=args.autotune, skew_factor=2, dse='skewing',
             dle=('blocking,openmp', {'blockinner': args.blockinner}))
else:
    raise ValueError("No tiling selected")
