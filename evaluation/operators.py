from functools import reduce
from operator import mul

import numpy as np
from sympy import solve

from devito import Eq, Grid, Operator, TimeFunction, configuration, clear_cache
from devito.logger import info
from devito import Constant, configuration
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, RickerSource, Receiver


def laplace(shape, time_order=2, space_order=2, skew_factor=0, autotune=True, iterations=16, **kwargs):
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
    return u.data


def acoustic_setup(shape=(50, 50, 50), spacing=(15.0, 15.0, 15.0),
                   tn=500., time_order=2, space_order=4, nbpml=10,
                   constant=False, **kwargs):
    nrec = shape[0]
    preset = 'constant-isotropic' if constant else 'layers-isotropic'
    model = demo_model(preset, shape=shape,
                       spacing=spacing, nbpml=nbpml)

    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if time_order == 4 else 1.0)
    t0 = 0.0
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='nrec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                time_order=time_order,
                                space_order=space_order, **kwargs)
    return solver


# Velocity models
def smooth10(vel, shape):
    if np.isscalar(vel):
        return .9 * vel * np.ones(shape, dtype=np.float32)
    out = np.ones(shape, dtype=np.float32)
    nz = shape[-1]

    for a in range(5, nz-6):
        if len(shape) == 2:
            out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
        else:
            out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10

    return out


def acoustic(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=1000.0,
        time_order=2, space_order=4, nbpml=40, full_run=False,
        autotune=False, constant=False, skew=0, iterations=0, **kwargs):

    configuration['skew_factor'] = skew
    solver = acoustic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                            space_order=space_order, time_order=time_order,
                            constant=constant, **kwargs)

    initial_vp = smooth10(solver.model.m.data, solver.model.shape_domain)
    dm = np.float32(initial_vp**2 - solver.model.m.data)
    info("Applying Forward")
    rec, u, summary = solver.forward(save=iterations+1, autotune=autotune)

    if constant:
        # With  a new m as Constant
        m0 = Constant(name="m", value=.25, dtype=np.float32)
        solver.forward(save=full_run, m=m0)
        # With a new m as a scalar value
        solver.forward(save=full_run, m=.25)

    if not full_run:
        return u.data

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    info("Applying Born")
    solver.born(dm, autotune=autotune)
    info("Applying Gradient")
    solver.gradient(rec, u, autotune=autotune)
