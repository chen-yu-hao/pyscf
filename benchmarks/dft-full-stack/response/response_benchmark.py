#!/usr/bin/env python3
"""Fresh-process analytic electric-field CPKS timing, without an SCF in run mode."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time


def git_metadata(source):
    source = str(Path(source).resolve())
    commit = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', source, 'diff', 'HEAD', '--name-only'], text=True).strip())
    if dirty:
        raise RuntimeError('Tracked benchmark source differs from its commit')
    return {'source': source, 'commit': commit, 'tracked_source_dirty': dirty}


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('VmRSS unavailable')


class MemorySample:
    def __init__(self):
        self.baseline = self.peak = rss_mib()
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, rss_mib())
            self.samples += 1
            self.stop_event.wait(.005)

    def start(self):
        self.thread.start()

    def stop(self):
        self.peak = max(self.peak, rss_mib())
        self.stop_event.set()
        self.thread.join()
        return {'rss_baseline_mib': self.baseline, 'rss_sampled_peak_mib': self.peak,
                'rss_peak_increment_mib': self.peak - self.baseline,
                'rss_sample_interval_s': .005, 'rss_sample_count': self.samples}


def cache_info(function):
    """Read cache metadata from response closures, without replacing any methods."""
    seen = set()
    pending = [function]
    while pending:
        fn = pending.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        cells = getattr(fn, '__closure__', None)
        if cells is None:
            continue
        local = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in cells)))
        if 'fxc' in local:
            fxc = local['fxc']
            arrays = []
            def add(value):
                if isinstance(value, (tuple, list)):
                    for item in value:
                        add(item)
                elif hasattr(value, 'nbytes'):
                    arrays.append(value)
            for key in ('rho0', 'vxc', 'fxc'):
                add(local.get(key))
            if hasattr(fxc, 'values'):
                add(fxc.values)
                add(fxc.rho)
            unique = {int(a.ctypes.data): int(a.nbytes) for a in arrays}
            return {'type': type(fxc).__name__, 'unique_array_bytes': sum(unique.values()),
                    'array_shapes': [list(a.shape) for a in arrays]}
        pending.extend(value for value in local.values() if callable(value))
    raise RuntimeError('Response closure has no identifiable XC cache')


def make_mf(metadata):
    from pyscf import dft, gto
    mol = gto.M(atom=metadata['atom'], basis=metadata['basis'], spin=metadata['spin'],
                charge=0, unit='Angstrom', verbose=0)
    mf = dft.UKS(mol) if metadata['spin'] else dft.RKS(mol)
    mf.xc = metadata['xc']
    mf.max_memory = metadata['max_memory_mb']
    mf.direct_scf = True
    mf.small_rho_cutoff = 0
    mf.grids.level = metadata['grid_level']
    return mf


def prepare(args, provenance):
    import numpy as np
    count = args.waters
    atom = []
    for k in range(count):
        x, y = (k % 4) * 3.3, (k // 4) * 3.3
        atom.extend([['O', [x, y, 0]], ['H', [x + .758602, y, .504284]]])
        if not (args.spin and k == count - 1):
            atom.append(['H', [x - .758602, y, .504284]])
    meta = {'atom': atom, 'basis': 'cc-pVDZ', 'spin': args.spin, 'xc': args.xc,
            'grid_level': args.grid_level, 'max_memory_mb': args.max_memory,
            'scf_source': provenance, 'fixture_role': 'frozen converged reference orbitals and grid'}
    mf = make_mf(meta)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('Fixture SCF did not converge')
    gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
    if gradient > 2e-7:
        raise RuntimeError('Fixture orbital gradient exceeds 2e-7')
    meta.update({'scf_energy_hartree': float(mf.e_tot), 'scf_gradient_norm': gradient,
                 'nao': mf.mol.nao, 'ngrids': len(mf.grids.weights)})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ, mo_energy=mf.mo_energy,
             coords=mf.grids.coords, weights=mf.grids.weights,
             metadata=np.asarray(json.dumps(meta)))
    print(json.dumps(meta, indent=2))


def run(args, provenance):
    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf.hessian import rhf, uhf
    from pyscf.scf import cphf, ucphf
    from threadpoolctl import threadpool_info
    fixture = Path(args.input).resolve()
    with np.load(fixture, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        mf = make_mf(meta)
        mf.mo_coeff = data['mo_coeff'].copy()
        mf.mo_occ = data['mo_occ'].copy()
        mf.mo_energy = data['mo_energy'].copy()
        mf.grids.coords = data['coords'].copy()
        mf.grids.weights = data['weights'].copy()
    mf.grids.non0tab = None
    mf.converged = True
    mf.e_tot = meta['scf_energy_hartree']
    mf._eri = None
    coeff, occ, energies = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    dipole = mf.mol.intor_symmetric('int1e_r', comp=3)
    if meta['spin']:
        h1 = tuple(np.einsum('pi,xpq,qj->xij', c, dipole, c[:,o > 0])
                   for c, o in zip(coeff, occ))
        s1 = tuple(np.zeros_like(h) for h in h1)
        factory, solver = uhf.gen_vind, ucphf.solve
    else:
        h1 = np.einsum('pi,xpq,qj->xij', coeff, dipole, coeff[:,occ > 0])
        s1 = np.zeros_like(h1)
        factory, solver = rhf.gen_vind, cphf.solve
    del dipole
    gc.collect()
    memory = MemorySample()
    process_hwm_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    memory.start()
    cpu0 = time.process_time()
    t0 = time.perf_counter()
    action = factory(mf, coeff, occ)
    t1 = time.perf_counter()
    calls = 0
    def counted_action(x):
        nonlocal calls
        calls += 1
        return action(x)
    mo1, mo_e1 = solver(counted_action, energies, occ, h1, s1,
                       tol=1e-10, max_cycle=80, verbose=0)
    t2 = time.perf_counter()
    cpu = time.process_time() - cpu0
    memory_record = memory.stop()
    hwm_response = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    solve_calls = calls
    if meta['spin']:
        flat = np.hstack([x.reshape(3, -1) for x in mo1])
        induced = action(flat)
        nalpha = mo1[0].shape[1] * mo1[0].shape[2]
        potentials = (induced[:,:nalpha].reshape(mo1[0].shape),
                      induced[:,nalpha:].reshape(mo1[1].shape))
        residuals = []
        polarizability = np.zeros((3, 3))
        for h, x, potential, e, o in zip(h1, mo1, potentials, energies, occ):
            gap = e[o == 0,None] - e[o > 0]
            residuals.append(x[:,o == 0] * gap + h[:,o == 0] + potential[:,o == 0])
            polarizability -= 2 * np.einsum('xai,yai->xy', h, x)
    else:
        potential = action(mo1)
        gap = energies[occ == 0,None] - energies[occ > 0]
        residuals = [mo1[:,occ == 0] * gap + h1[:,occ == 0] + potential[:,occ == 0]]
        polarizability = -4 * np.einsum('xai,yai->xy', h1, mo1)
    residual_max = max(float(np.max(np.abs(r))) for r in residuals)
    symmetry_error = float(np.max(np.abs(polarizability - polarizability.T)))
    accepted = (residual_max < 2e-8 and symmetry_error < 2e-7 and
                bool(np.isfinite(polarizability).all()))
    native = {}
    for name in ('libdft.so', 'libcgto.so', 'libxc_itrf.so'):
        path = Path(pyscf.__file__).resolve().parent / 'lib' / name
        native[name] = {'path': str(path.resolve()),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {'schema': 1, 'operation': 'analytic electric-field CPKS, three Cartesian fields',
              'source': provenance, 'pyscf_import': pyscf.__file__, 'native_libraries': native,
              'fixture': str(fixture), 'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'fixture_metadata': meta, 'affinity': sorted(os.sched_getaffinity(0)),
              'omp_threads': lib.num_threads(), 'threadpools': threadpool_info(),
              'cache_wall_s': t1-t0, 'solver_wall_s': t2-t1, 'response_wall_s': t2-t0,
              'response_cpu_s': cpu, 'solver_action_calls': solve_calls,
              'response_peak_rss': memory_record, 'process_hwm_before_mib': process_hwm_before,
              'process_hwm_after_response_mib': hwm_response, 'cache': cache_info(action),
              'coulomb_eri_cached': mf._eri is not None,
              'coulomb_eri_bytes': 0 if mf._eri is None else int(mf._eri.nbytes),
              'polarizability_au': polarizability.tolist(), 'residual_max': residual_max,
              'polarizability_symmetry_error': symmetry_error, 'scientific_accepted': accepted,
              'timing_scope': 'cache construction plus complete iterative CPKS; imports, frozen input loading, SCF, dipole integrals and postsolve residual check excluded',
              'memory_scope': 'sampled response interval RSS, 5 ms; process high-water mark diagnostic includes startup'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    if not accepted:
        raise RuntimeError('CPKS residual or symmetry acceptance failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--source', required=True, help='Checked-out benchmarked source repository')
    parser.add_argument('--input')
    parser.add_argument('--output', required=True)
    parser.add_argument('--xc', default='TPSS')
    parser.add_argument('--waters', type=int, default=4)
    parser.add_argument('--spin', type=int, choices=[0, 1], default=0,
                        help='1 replaces the last water by an OH radical')
    parser.add_argument('--grid-level', type=int, default=3)
    parser.add_argument('--max-memory', type=int, default=1000)
    args = parser.parse_args()
    provenance = git_metadata(args.source)
    import pyscf
    if Path(pyscf.__file__).resolve().parent.parent != Path(args.source).resolve():
        raise RuntimeError('Imported PySCF does not match --source')
    if args.mode == 'prepare':
        prepare(args, provenance)
    else:
        if args.input is None:
            parser.error('--input is required for run')
        run(args, provenance)


if __name__ == '__main__':
    main()
