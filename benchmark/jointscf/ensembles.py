#!/usr/bin/env python
'''
Geometry ensembles of the JointSCF benchmark.

The four random ensembles (h2o, h2o_stretched, c2h4, h2co) use the same
distributions and random seeds as the benchmark scripts of the
shared_lowrank_joint_scf prototype.  Two structured ensembles (a 2D scan and
a closed path) represent PES scans / trajectory-like data, in which
neighbouring configurations are close to each other.
'''

import numpy as np
from pyscf import gto


def _water(r1, r2, ang_deg, basis):
    ang = np.deg2rad(ang_deg)
    return gto.M(
        atom=[('O', (0., 0., 0.)), ('H', (r1, 0., 0.)),
              ('H', (r2*np.cos(ang), r2*np.sin(ang), 0.))],
        basis=basis, unit='Angstrom', verbose=0)

def h2o(n, basis='def2-svp', seed=2026):
    '''Random distortions around the equilibrium (3 degrees of freedom)'''
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r1 = float(rng.uniform(0.89, 1.16))
        r2 = float(rng.uniform(0.89, 1.16))
        ang = float(rng.uniform(96.0, 114.0))
        out.append(_water(r1, r2, ang, basis))
    return out

def h2o_stretched(n, basis='def2-svp', seed=5050):
    '''Large amplitude bond stretching, 0.86 - 1.65 Angstrom'''
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r1 = float(rng.uniform(0.86, 1.65))
        r2 = float(rng.uniform(0.88, 1.45))
        ang = float(rng.uniform(90.0, 122.0))
        out.append(_water(r1, r2, ang, basis))
    return out

def h2o_scan2d(n, basis='def2-svp'):
    '''Regular 2D scan: symmetric stretch x bending angle'''
    m = int(round(np.sqrt(n)))
    assert m * m == n
    out = []
    for r in np.linspace(0.90, 1.10, m):
        for ang in np.linspace(96.0, 114.0, m):
            out.append(_water(float(r), float(r), float(ang), basis))
    return out

def _ethylene(r_cc, r_ch1, r_ch2, ang_hcc, twist, wag, basis):
    c1 = np.array([-0.5 * r_cc, 0.0, 0.0])
    c2 = np.array([0.5 * r_cc, 0.0, 0.0])
    dx_l = r_ch1 * (-np.cos(ang_hcc))
    hy_l = r_ch1 * np.sin(np.pi - ang_hcc)
    h1 = np.array([c1[0] - dx_l, hy_l * np.cos(wag), hy_l * np.sin(wag)])
    h2 = np.array([c1[0] - dx_l, -hy_l * np.cos(wag), hy_l * np.sin(wag)])
    dx_r = r_ch2 * (-np.cos(ang_hcc))
    hy_r = r_ch2 * np.sin(ang_hcc)
    h3 = np.array([c2[0] + dx_r, hy_r * np.cos(twist), hy_r * np.sin(twist)])
    h4 = np.array([c2[0] + dx_r, -hy_r * np.cos(twist), -hy_r * np.sin(twist)])
    return gto.M(
        atom=[('C', tuple(c1)), ('C', tuple(c2)), ('H', tuple(h1)),
              ('H', tuple(h2)), ('H', tuple(h3)), ('H', tuple(h4))],
        basis=basis, unit='Angstrom', verbose=0)

_C2H4_RANGES = ((1.28, 1.42), (1.05, 1.13), (1.05, 1.13),
                (117.0, 124.5), (-18.0, 18.0), (-10.0, 10.0))

def c2h4(n, basis='def2-svp', seed=3030):
    '''Random distortions of ethylene (6 degrees of freedom)'''
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r_cc, r_ch1, r_ch2, ang, twist, wag = (
            float(rng.uniform(lo, hi)) for lo, hi in _C2H4_RANGES)
        out.append(_ethylene(r_cc, r_ch1, r_ch2, np.deg2rad(ang),
                             np.deg2rad(twist), np.deg2rad(wag), basis))
    return out

def c2h4_path(n, basis='def2-svp'):
    '''A smooth closed path through the same 6-dimensional region which
    c2h4() samples at random.  Model of a trajectory / reaction path.'''
    freq = (1, 1, 2, 1, 2, 1)
    phase = (0.0, 1.3, 0.4, 2.1, 0.9, 2.9)
    out = []
    for t in np.arange(n) / n:
        p = [0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sin(2 * np.pi * f * t + ph)
             for (lo, hi), f, ph in zip(_C2H4_RANGES, freq, phase)]
        out.append(_ethylene(p[0], p[1], p[2], np.deg2rad(p[3]),
                             np.deg2rad(p[4]), np.deg2rad(p[5]), basis))
    return out

def h2co(n, basis='def2-tzvp', seed=4040):
    '''Random distortions of formaldehyde (5 degrees of freedom)'''
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r_co = float(rng.uniform(1.16, 1.28))
        r_ch1 = float(rng.uniform(1.07, 1.15))
        r_ch2 = float(rng.uniform(1.07, 1.15))
        ang = np.deg2rad(float(rng.uniform(118.0, 125.0)))
        oop = np.deg2rad(float(rng.uniform(-12.0, 12.0)))
        hx1, hy1 = r_ch1 * np.cos(ang), r_ch1 * np.sin(ang)
        hx2, hy2 = r_ch2 * np.cos(ang), r_ch2 * np.sin(ang)
        out.append(gto.M(
            atom=[('C', (0.0, 0.0, 0.0)), ('O', (r_co, 0.0, 0.0)),
                  ('H', (hx1, hy1 * np.cos(oop), hy1 * np.sin(oop))),
                  ('H', (hx2, -hy2 * np.cos(oop), hy2 * np.sin(oop)))],
            basis=basis, unit='Angstrom', verbose=0))
    return out


# name: (generator, number of configurations, xc, grid level)
CASES = {
    'h2o-16': (h2o, 16, 'pbe', 3),
    'h2o-32': (h2o, 32, 'pbe', 3),
    'h2o-64': (h2o, 64, 'pbe', 3),
    'h2o-128': (h2o, 128, 'pbe', 3),
    'h2o-stretched-64': (h2o_stretched, 64, 'pbe', 3),
    'h2o-scan2d-64': (h2o_scan2d, 64, 'pbe', 3),
    'c2h4-48': (c2h4, 48, 'pbe', 2),
    'c2h4-path-48': (c2h4_path, 48, 'pbe', 2),
    'h2co-40': (h2co, 40, 'b3lyp', 2),
}

def build_case(name):
    gen, n, xc, level = CASES[name]
    return gen(n), xc, level
