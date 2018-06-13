#!/usr/bin/env python3
import argparse
from functools import reduce
from math import floor


parser = argparse.ArgumentParser(description="Calculate bounds for arithmetic intensity")
tiling = parser.add_mutually_exclusive_group(required=True)
tiling.add_argument('-N', '--no-tiling', action='store_true')
tiling.add_argument('-S', '--space-tiling', action='store_true')
tiling.add_argument('-T', '--time-tiling', action='store_true')
parser.add_argument('-c', '--cache-size', type=int)
parser.add_argument('-s', '--shape', nargs='+', type=int)
parser.add_argument('-r', '--stencil-radius', nargs='+', type=int)
parser.add_argument('-b', '--blockshape', nargs='+', type=int)
parser.add_argument('-i', '--blockinner', action='store_true')
parser.add_argument('-oi', '--reported-oi', type=float)
args = parser.parse_args()

mul = lambda x, y: x * y


def calc_Fi(i, tiles, radii, extents):
    # Beware: 0-indexed
    return 2 * reduce(mul, tiles[:i], 1) * radii[i] * reduce(mul, extents[i+1:], 1)

def calc_Bi(i, tiles, radii, extents):
    return 2 * (radii[i] / extents[i]) * floor(extents[i] / tiles[i])

def pie(a, b):
    return a + b - a * b

def calc_D(extents):
    return reduce(mul, extents)

def calc_BD(tiles, radii, extents, omega, tt=1):
    B = 0
    for i in reversed(range(len(tiles))):
        if B == 0 and tt * calc_Fi(i, tiles, radii, extents) > omega:
            B = 1
        elif B > 0:
            B = pie(B, calc_Bi(i, tiles, radii, extents))
    
    if B == 1:
        B = (tt * calc_Fi(0, tiles, radii, extents) - omega) * floor(extents[0] / tiles[0])
    
    return B / calc_D(extents)

if args.blockinner:
    args.blockshape.append(args.shape[-1])

if args.no_tiling:
    print(args.reported_oi)

if args.space_tiling:
    assert (len(args.shape) == len(args.blockshape))
    assert (len(args.stencil_radius) == len(args.shape))
    BD = calc_BD(args.blockshape, args.stencil_radius, args.shape, args.cache_size)
    print(args.reported_oi / (1 + BD))

if args.time_tiling:
    # Assume r_t == 1, as in tested stencils
    assert (len(args.shape) + 1 == len(args.blockshape))
    assert (len(args.stencil_radius) == len(args.shape))
    tt = args.blockshape[0]
    BD = calc_BD(args.blockshape[1:], args.stencil_radius, args.shape, args.cache_size, tt)
    print(args.reported_oi * tt / (1 + BD))
