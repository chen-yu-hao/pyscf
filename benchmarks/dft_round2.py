"""Round-two molecular benchmark, reusing the unchanged grid6 acceptance rules.

Both variants disable the round-one tile experiment. Only the candidate enables
PYSCF_DFT_DENSE_BLAS. Full SCF timings are fresh processes and uninstrumented;
C dispatch counters belong only to the separately collected profile run.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest import mock

import dft_water_grid6 as harness

original_worker = harness.worker


def worker(cfg):
    sys.path.insert(0, str(Path(cfg['root']).resolve()))
    from pyscf.dft import numint
    calls = Counter()
    original = numint.libdft.VXCxc_vmat

    def counted(*args):
        result = original(*args)
        mode = 'sparse' if bool(args[8]) else 'dense'
        component = args[7].value
        calls[f'{mode}_ncomp{component}_returned{result}'] += 1
        return result

    if cfg['profile']:
        with mock.patch.object(numint.libdft, 'VXCxc_vmat', counted):
            result = original_worker(cfg)
    else:
        result = original_worker(cfg)
    result['dense_blas'] = os.environ.get('PYSCF_DFT_DENSE_BLAS', '0')
    result['c_dispatch_counts'] = dict(calls)
    result['dispatch_count_scope'] = 'profile SCF plus post-SCF fixed-density verification'
    library = Path(cfg['root']) / 'pyscf/lib/libdft.so'
    result['libdft_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    result['round'] = 2
    return result


def launch(cfg, timeout):
    config = Path(cfg['result']).with_suffix('.config.json')
    config.write_text(json.dumps(cfg))
    env = dict(os.environ, PYTHONPATH=cfg['root'], OMP_NUM_THREADS=str(cfg['threads']),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
               OMP_DYNAMIC='FALSE', OMP_MAX_ACTIVE_LEVELS='1', PYSCF_NUMA=cfg['numa'],
               PYSCF_DFT_VMAT_TILE_SCREEN='0',
               PYSCF_DFT_DENSE_BLAS='1' if cfg['variant'] == 'candidate' else '0')
    started = time.perf_counter()
    with Path(cfg['result']).with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(config)],
                       cwd=cfg['root'], env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)
    value = json.loads(Path(cfg['result']).read_text())
    value['process_seconds'] = time.perf_counter()-started
    return value


if __name__ == '__main__':
    harness.worker = worker
    harness.launch = launch
    harness.main()
