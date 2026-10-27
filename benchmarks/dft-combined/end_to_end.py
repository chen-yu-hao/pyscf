"""Fresh-process grid construction or ordinary SCF on the requested checkout."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

from xc_benchmark import array_digest, checkout_info, digest, native_maps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--mode', choices=['scf', 'grid-build'], required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--xc', default='PBE')
    p.add_argument('--threads', type=int, required=True)
    p.add_argument('--affinity', required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = checkout_info(a.source.resolve())
    assert source['commit'] == a.commit and not source['tracked_changes']
    cpus = list(map(int, a.affinity.split(',')))
    assert len(cpus) == a.threads
    os.sched_setaffinity(0, cpus)
    os.environ.update(OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                      OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0', NUMPY_MADVISE_HUGEPAGE='0')
    sys.path.insert(0, str(a.source.resolve()))
    import numpy as np
    import pyscf
    from pyscf import dft, gto, lib
    from threadpoolctl import threadpool_info
    assert Path(pyscf.__file__).resolve().is_relative_to(a.source.resolve())
    lib.num_threads(a.threads)
    protocol = json.loads(a.protocol.read_text())
    record = {'source': source, 'pyscf_import': pyscf.__file__, 'case': a.case,
              'mode': a.mode, 'threads': a.threads, 'affinity': sorted(cpus),
              'protocol_sha256': digest(a.protocol), 'worker_sha256': digest(Path(__file__)),
              'python': sys.executable, 'numpy_version': np.__version__}
    if a.mode == 'grid-build':
        meta = json.loads((a.fixtures / (a.case + '.json')).read_text())
        mol = gto.M(atom=meta['atoms'], basis=meta['basis'], spin=meta['spin'],
                    charge=meta['charge'], unit=meta['unit'], verbose=0)
        samples = []
        reference = None
        cfg = protocol['grid_build']
        for i in range(cfg['warmups'] + cfg['calls']):
            started = time.perf_counter()
            grids = dft.gen_grid.Grids(mol)
            grids.level = cfg['level']
            grids.cutoff = meta['grid_cutoff']
            grids.build(with_non0tab=cfg['with_non0tab'])
            wall = time.perf_counter() - started
            identity = [array_digest(x) for x in (grids.coords, grids.weights, grids.non0tab)]
            if reference is None:
                reference = identity
            assert identity == reference
            if i >= cfg['warmups']:
                samples.append(wall)
        with np.load(a.fixtures / (a.case + '.npz')) as frozen:
            for name in ('coords', 'weights', 'non0tab'):
                np.testing.assert_array_equal(getattr(grids, name), frozen[name])
        record.update(wall_seconds=samples, median_wall_seconds=statistics.median(samples),
                      exact_fixture_grid=True, fixture_sha256=meta['fixture_sha256'])
    else:
        spec = next(c for c in protocol['cases'] if c['id'] == a.case)
        cfg = protocol['scf']
        mol = gto.M(atom=spec['atoms_bohr'], basis=spec['basis'], spin=spec['spin'],
                    charge=spec['charge'], unit='Bohr', cart=False, symmetry=False,
                    max_memory=cfg['max_memory_mb'], verbose=0)
        assert mol.nao == spec['expected_nao']
        mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
        mf.xc = a.xc
        mf.grids.level = cfg['grid_level']
        mf.grids.cutoff = cfg['grid_cutoff']
        mf._numint.cutoff = cfg['numint_cutoff']
        for key in ('small_rho_cutoff', 'conv_tol', 'conv_tol_grad', 'max_cycle',
                    'diis_space', 'init_guess', 'direct_scf', 'direct_scf_tol'):
            setattr(mf, key, cfg[key])
        mf._is_mem_enough = lambda: False
        mf.chkfile = None
        history = []
        def callback(env):
            history.append({'cycle': int(env['cycle'] + 1), 'energy': float(env['e_tot']),
                            'energy_delta': float(env['e_tot'] - env['last_hf_e']),
                            'gradient': float(env['norm_gorb'])})
        mf.callback = callback
        started = time.perf_counter()
        energy = mf.kernel()
        wall = time.perf_counter() - started
        dm = np.asarray(mf.make_rdm1())
        overlap = mf.get_ovlp()
        assert mf.converged and np.isfinite(energy) and np.isfinite(dm).all()
        nelec = np.einsum('...ij,ji->...', dm, overlap)
        expected = mol.nelec if mol.spin else mol.nelectron
        np.testing.assert_allclose(nelec, expected, atol=protocol['gates']['overlap_electrons'], rtol=0)
        arrays = a.output.with_suffix('.npz')
        np.savez(arrays, dm=dm, energy=energy)
        grids = mf.grids
        record.update(xc=a.xc, energy=float(energy), converged=bool(mf.converged),
                      kernel_wall_seconds=wall, cycles=int(mf.cycles), history=history,
                      overlap_electrons=np.asarray(nelec).tolist(), output_arrays=str(arrays))
        if mol.spin:
            record['spin_square'] = list(map(float, mf.spin_square()))
    record.update(status='succeeded', nao=mol.nao, grid_points=len(grids.weights),
                  grid_coords_sha256=array_digest(grids.coords),
                  grid_weights_sha256=array_digest(grids.weights),
                  grid_mask_sha256=array_digest(grids.non0tab),
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  threadpools=threadpool_info(), native_maps=native_maps())
    a.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
