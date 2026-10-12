#!/usr/bin/env python
"""Compare one RCCSDT and UCCSDT update on closed-shell H2 and NH3."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

CASES = {
    'h2-ccpvtz': ('H 0 0 0; H 0 0 0.74', 'cc-pvtz'),
    'nh3-ccpvdz': ('N 0 0 0; H 0 0.9377 0.3816; '
                   'H 0.8121 -0.4688 0.3816; H -0.8121 -0.4688 0.3816', 'cc-pvdz'),
}


def _setup_maps(mycc):
    if not mycc.do_tri_max_t:
        return
    values = mycc.setup_tri2block()
    if np_scalar(mycc.nocc):
        mycc.tri2block_map, mycc.tri2block_mask, mycc.tri2block_tp = values
        mycc.blksize = min(mycc.blksize, mycc.nocc)
        mycc.blksize_oovv = min(mycc.blksize_oovv, mycc.nocc)
        mycc.blksize_oooo = min(mycc.blksize_oooo, mycc.nocc)
    else:
        names = ('t2c_map_6f_oa', 't2c_mask_6f_oa', 't2c_map_2f_oa', 't2c_mask_2f_oa',
                 't2c_map_6f_va', 't2c_mask_6f_va', 't2c_map_2f_va', 't2c_mask_2f_va',
                 't2c_map_6f_ob', 't2c_mask_6f_ob', 't2c_map_2f_ob', 't2c_mask_2f_ob',
                 't2c_map_6f_vb', 't2c_mask_6f_vb', 't2c_map_2f_vb', 't2c_mask_2f_vb')
        for name, value in zip(names, values):
            setattr(mycc, name, value)
        nocc = max(mycc.nocc)
        nvir = max(m - o for m, o in zip(mycc.nmo, mycc.nocc))
        mycc.blksize_o_aaa = min(mycc.blksize_o_aaa, nocc)
        mycc.blksize_o_aab = min(mycc.blksize_o_aab, nocc)
        mycc.blksize_v_aaa = min(mycc.blksize_v_aaa, nvir)
        mycc.blksize_v_aab = min(mycc.blksize_v_aab, nvir)


def np_scalar(value):
    import numpy as np
    return np.isscalar(value)


def worker(case, method, nthreads, frozen):
    import numpy as np
    from pyscf import cc, gto, lib, scf
    lib.num_threads(1)
    atom, basis = CASES[case]
    mol = gto.M(atom=atom, basis=basis, charge=0, spin=0, cart=False, verbose=0)
    t0 = time.perf_counter()
    mf = scf.RHF(mol).set(conv_tol=1e-12).run()
    scf_seconds = time.perf_counter() - t0
    if method == 'RCCSDT':
        mycc = cc.RCCSDT(mf, frozen=frozen, compact_tamps=True)
    elif method == 'UCCSDT':
        mycc = cc.UCCSDT(mf.to_uhf(), frozen=frozen, compact_tamps=True)
    else:
        raise ValueError(method)
    t0 = time.perf_counter(); eris = mycc.ao2mo(); ao2mo_seconds = time.perf_counter()-t0
    t0 = time.perf_counter(); _, tamps = mycc.init_amps(eris); init_seconds = time.perf_counter()-t0
    _setup_maps(mycc)
    sources = [Path(__file__).resolve(), Path(lib.load_library('libccsdt')._name),
               Path(cc.rccsdt.__file__), Path(cc.uccsdt.__file__)]
    source_sha256 = {str(x.resolve()): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    lib.num_threads(nthreads)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter(); norms = mycc.update_amps_(tamps, eris); elapsed = time.perf_counter()-t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        'case': case, 'method': method, 'basis': basis, 'frozen': frozen,
        'threads': nthreads, 'nocc': ([int(mycc.nocc)] if np.isscalar(mycc.nocc) else list(map(int,mycc.nocc))),
        'nmo': ([int(mycc.nmo)] if np.isscalar(mycc.nmo) else list(map(int,mycc.nmo))),
        'scf_seconds': scf_seconds, 'ao2mo_seconds': ao2mo_seconds,
        'initialization_seconds': init_seconds, 'first_iteration_seconds': elapsed,
        'residual_norm': float(np.linalg.norm(norms)),
        'rss_ru_maxrss_before_kib': int(rss_before), 'rss_ru_maxrss_after_kib': int(rss_after),
        'e_hf': float(mf.e_tot), 'source_sha256': source_sha256,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=sorted(CASES), default=sorted(CASES))
    parser.add_argument('--methods', nargs='+', choices=('RCCSDT','UCCSDT'), default=('RCCSDT','UCCSDT'))
    parser.add_argument('--threads', nargs='+', type=int, default=[1,2,4])
    parser.add_argument('--frozen', type=int, default=0)
    parser.add_argument('--worker', nargs=4, metavar=('CASE','METHOD','THREADS','FROZEN'))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker[0], args.worker[1], int(args.worker[2]), int(args.worker[3]))))
        return
    rows=[]
    for case in args.cases:
        for method in args.methods:
            for threads in args.threads:
                env=dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                proc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',case,method,str(threads),str(args.frozen)],
                                    env=env,check=True,capture_output=True,text=True)
                rows.append(json.loads(proc.stdout))
    for case in args.cases:
        for method in args.methods:
            ref=next(x for x in rows if x['case']==case and x['method']==method and x['threads']==args.threads[0])
            for row in rows:
                if row['case']==case and row['method']==method:
                    row['speedup_vs_first_thread']=ref['first_iteration_seconds']/row['first_iteration_seconds']
    print(json.dumps({'case': 'RCCSDT vs UCCSDT, first update only', 'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
