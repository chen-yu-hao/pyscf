#!/usr/bin/env python
"""One-update RCCSD/UCCSD comparison for H2O/cc-pVTZ after UCCSD tuning."""

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


GEOMETRY = ('O 0 0 0; H 0 0 0.9584; '
            'H 0.9266 0 -0.2396')


def worker(method):
    import numpy as np
    from pyscf import cc, gto, lib, scf
    lib.num_threads(1)
    mol = gto.M(atom=GEOMETRY, basis='cc-pvtz', spin=0, charge=0,
                cart=False, verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-12).run()
    mycc = cc.RCCSD(mf, frozen=0) if method == 'RCCSD' else cc.UCCSD(mf, frozen=0)
    eris = mycc.ao2mo()
    _, t1, t2 = mycc.init_amps(eris)
    mycc.async_io = False
    lib.num_threads(1)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    t1new, t2new = mycc.update_amps(t1, t2, eris)
    seconds = time.perf_counter() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(method=method, molecule='H2O', basis='cc-pVTZ', threads=1,
                nocc=mycc.nocc, nmo=mycc.nmo, seconds=seconds,
                residual_norm=float(np.linalg.norm(np.asarray(t1new).ravel()) +
                                   np.linalg.norm(np.asarray(t2new).ravel())),
                rss_before_kib=int(rss0), rss_after_kib=int(rss1),
                amplitude_bytes=int(sum(x.nbytes for x in
                                        ([t1] if isinstance(t1, np.ndarray) else t1))) +
                                int(sum(x.nbytes for x in
                                        ([t2] if isinstance(t2, np.ndarray) else t2))))


def main():
    rows = []
    for method in ('RCCSD', 'UCCSD'):
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', method],
                              env=env, check=True, capture_output=True, text=True)
        rows.append(json.loads(proc.stdout))
    print(json.dumps({'case': 'H2O/cc-pVTZ, one update, async_io=False',
                      'measurements': rows}, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        print(json.dumps(worker(sys.argv[2])))
    else:
        main()
