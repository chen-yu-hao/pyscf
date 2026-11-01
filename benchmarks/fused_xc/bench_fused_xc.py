"""Fused RKS XC integration versus the block-loop path.

    python bench_fused_xc.py xc  <pyscf-root> <nwater> <xc> <threads> [repeat]
    python bench_fused_xc.py scf <pyscf-root> <nwater> <xc> <threads> [repeat]

`xc` times ni.nr_rks on one converged density in a single process, alternating
PYSCF_DFT_FUSED=0/1. `scf` runs complete cold SCFs, each in a fresh process,
alternating the two settings. BLAS is limited to one thread (pass BLAS=n in the
environment to change it). Geometry: the water clusters of dft_water_grid6.py,
def2-TZVPP, grid level 6, direct J/K (max_memory 1500 MB).
"""
import json
import os
import statistics
import subprocess
import sys
import time


def water_cluster(n):
    atoms = []
    for i in range(n):
        x, y = 4.0*(i % 3), 4.0*(i//3)
        atoms += [('O', (x, y, 0)), ('H', (x+.757, y, .587)), ('H', (x-.757, y, .587))]
    return atoms


def setup(root, n, xc, threads):
    sys.path.insert(0, root)
    from pyscf import gto, dft, lib
    lib.num_threads(threads)
    mol = gto.M(atom=water_cluster(n), basis='def2-TZVPP', unit='Angstrom',
                verbose=0, max_memory=1500)
    mf = dft.RKS(mol, xc=xc)
    mf.grids.level = 6
    mf.conv_tol = 1e-8
    return mol, mf


def run_scf(root, n, xc, threads):
    from threadpoolctl import threadpool_limits
    mol, mf = setup(root, n, xc, threads)
    with threadpool_limits(limits=int(os.environ.get('BLAS', 1)), user_api='blas'):
        t = time.perf_counter()
        e = mf.kernel()
        wall = time.perf_counter() - t
    print(json.dumps({'wall': wall, 'energy': e, 'cycles': mf.cycles,
                      'converged': bool(mf.converged), 'nao': mol.nao,
                      'ngrids': int(mf.grids.weights.size)}))


def bench_xc(root, n, xc, threads, repeat):
    import numpy
    from threadpoolctl import threadpool_limits
    mol, mf = setup(root, n, xc, threads)
    os.environ['PYSCF_DFT_FUSED'] = '1'
    mf.kernel()
    dm = mf.make_rdm1()
    ni = mf._numint
    times = {'0': [], '1': []}
    out = {}
    with threadpool_limits(limits=int(os.environ.get('BLAS', 1)), user_api='blas'):
        for _ in range(repeat):
            for flag in ('0', '1'):
                os.environ['PYSCF_DFT_FUSED'] = flag
                t = time.perf_counter()
                out[flag] = ni.nr_rks(mol, mf.grids, xc, dm)
                times[flag].append(time.perf_counter() - t)
    ref, new = out['0'], out['1']
    t0, t1 = statistics.median(times['0']), statistics.median(times['1'])
    print(f'(H2O){n} nao={mol.nao} ngrids={mf.grids.weights.size} {xc} threads={threads}: '
          f'block loop {t0:.3f} s, fused {t1:.3f} s, speedup {t0/t1:.2f}x; '
          f'|dN|={abs(new[0]-ref[0]):.1e} |dExc|={abs(new[1]-ref[1]):.1e} '
          f'max|dV|={numpy.abs(new[2]-ref[2]).max():.1e}')


def bench_scf(root, n, xc, threads, repeat):
    res = {'0': [], '1': []}
    for _ in range(repeat):
        for flag in ('0', '1'):
            env = dict(os.environ, PYSCF_DFT_FUSED=flag)
            p = subprocess.run([sys.executable, __file__, '_scf', root, str(n), xc,
                                str(threads)], env=env, check=True,
                               capture_output=True, text=True)
            res[flag].append(json.loads(p.stdout.strip().splitlines()[-1]))
    t0 = statistics.median(r['wall'] for r in res['0'])
    t1 = statistics.median(r['wall'] for r in res['1'])
    de = max(abs(a['energy'] - b['energy']) for a in res['0'] for b in res['1'])
    r = res['1'][0]
    print(f'(H2O){n} nao={r["nao"]} {xc} threads={threads}: SCF block loop {t0:.2f} s, '
          f'fused {t1:.2f} s, speedup {t0/t1:.2f}x; cycles '
          f'{[x["cycles"] for x in res["0"]]} / {[x["cycles"] for x in res["1"]]}; '
          f'max|dE|={de:.1e}')


if __name__ == '__main__':
    mode, root, n, xc, threads = sys.argv[1:6]
    repeat = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    if mode == '_scf':
        run_scf(root, int(n), xc, int(threads))
    elif mode == 'xc':
        bench_xc(root, int(n), xc, int(threads), repeat)
    else:
        bench_scf(root, int(n), xc, int(threads), repeat)
