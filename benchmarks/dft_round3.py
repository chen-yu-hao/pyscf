"""Round-three MGGA packing / exact s8 precheck ablations against round two.

Both variants enable the accepted round-two dense BLAS path. New switches are
selected independently; no grid, convergence, integral cutoff or basis changes.
Ordinary samples have no profiler. Extra fixed-density J/K checks are untimed.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import dft_water_grid6 as harness
import dft_round2 as previous

original_verify = harness.verify


def worker(cfg):
    sys.path.insert(0, str(Path(cfg['root']).resolve()))
    import numpy as np
    from pyscf import gto, scf
    from pyscf.scf import _vhf
    from threadpoolctl import threadpool_limits
    counter = getattr(_vhf.libcvhf, 'CVHFround3_bound_stats', None)
    counts = (ctypes.c_ulonglong * 2)()
    if counter is not None:
        counter.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_int]
        counter.restype = None
        counter(counts, 1)
    result = previous.worker(cfg)
    if counter is not None:
        counter(counts, 0)
    result['jk_block_counts'] = dict(considered=int(counts[0]), rejected=int(counts[1]))
    result['jk_count_scope'] = 'profile SCF only; extra J/K verification is after the read'
    # The first baseline DM is shared by ALL later trials, including profiles.
    with np.load(cfg.get('reference') or result['arrays']) as ref:
        dm = ref['dm'].copy()
    mol = gto.M(atom=harness.water_cluster(cfg['n']), basis='def2-TZVPP',
                unit='Angstrom', verbose=0, max_memory=2300)
    mf = scf.RHF(mol)
    mf.max_memory = 2300
    mf._is_mem_enough = lambda: False
    with threadpool_limits(limits=1, user_api='blas'):
        vj, vk = mf.get_jk(mol, dm, hermi=1)
    if not np.isfinite(vj).all() or not np.isfinite(vk).all():
        raise AssertionError('Nonfinite fixed-density direct J/K')
    jk_file = str(Path(cfg['result']).with_suffix('.jk.npz'))
    np.savez(jk_file, vj=vj, vk=vk, fixed_dm=dm)
    result.update(round=3, component=cfg['component'], jk_arrays=jk_file,
                  mgga_pack_simd=os.getenv('PYSCF_DFT_MGGA_PACK_SIMD'),
                  jk_block_bound=os.getenv('PYSCF_JK_BLOCK_BOUND'),
                  libcvhf_sha256=hashlib.sha256(
                      (Path(cfg['root'])/'pyscf/lib/libcvhf.so').read_bytes()).hexdigest())
    return result


def verify(reference, candidate):
    import numpy as np
    result = original_verify(reference, candidate)
    with np.load(reference['jk_arrays']) as ref, np.load(candidate['jk_arrays']) as val:
        np.testing.assert_array_equal(ref['fixed_dm'], val['fixed_dm'])
        differences = {}
        for name in ('vj', 'vk'):
            np.testing.assert_allclose(ref[name], val[name], atol=1e-9, rtol=1e-10)
            differences[name] = float(np.max(np.abs(ref[name]-val[name])))
    result['fixed_density_jk_max_abs'] = differences
    return result


def launch(cfg, timeout):
    cfg['component'] = os.environ['ROUND3_COMPONENT']
    candidate = cfg['variant'] == 'candidate'
    mgga = candidate and cfg['component'] in ('mgga', 'both')
    jk = candidate and cfg['component'] in ('jk', 'both')
    config = Path(cfg['result']).with_suffix('.config.json')
    config.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=cfg['root'], OMP_NUM_THREADS=str(cfg['threads']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', PYSCF_NUMA=cfg['numa'],
               PYSCF_DFT_VMAT_TILE_SCREEN='0', PYSCF_DFT_DENSE_BLAS='1',
               PYSCF_DFT_MGGA_PACK_SIMD=str(int(mgga)),
               PYSCF_JK_BLOCK_BOUND=str(int(jk)),
               PYSCF_JK_BLOCK_BOUND_STATS=str(int(cfg['profile'])))
    stamp = time.perf_counter()
    with Path(cfg['result']).with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config)],
                       cwd=cfg['root'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)
    result = json.loads(Path(cfg['result']).read_text())
    result['process_seconds'] = time.perf_counter()-stamp
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--component', choices=['mgga', 'jk', 'both'], default='both')
    args, remaining = parser.parse_known_args()
    os.environ['ROUND3_COMPONENT'] = args.component
    sys.argv = [sys.argv[0]] + remaining
    harness.worker, harness.launch, harness.verify = worker, launch, verify
    harness.main()
