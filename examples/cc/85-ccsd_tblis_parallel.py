#!/usr/bin/env python
"""One CCSD update with explicit PySCF/TBLIS threads and serial pthread BLAS.

Requires pytblis 0.0.16 for --backend pytblis. Bind the process to the
allocated physical cores externally (e.g. taskset on this workstation).
The printed energy is after one update, not a converged CCSD result.
"""
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--method', choices=('RCCSD', 'UCCSD'), default='UCCSD')
    parser.add_argument('--backend', choices=('pyscf', 'pytblis'), default='pytblis')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    # Set these before importing any library that initializes a thread pool.
    os.environ.update(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='false',
                      OMP_WAIT_POLICY='PASSIVE', OMP_MAX_ACTIVE_LEVELS='1')
    from threadpoolctl import threadpool_limits
    from pyscf import cc, gto, lib, scf
    if args.backend == 'pytblis':
        import pytblis
        lib.numpy_helper.pytblis = pytblis
        lib.numpy_helper.EINSUM_BACKEND = 'pytblis'
        pytblis.set_num_threads(args.threads)
        assert pytblis.get_num_threads() == args.threads
    else:
        lib.numpy_helper.EINSUM_BACKEND = None
    lib.num_threads(args.threads)
    with threadpool_limits(limits=1, user_api='blas'):
        mol = gto.M(atom='O 0 0 0; H 0 0 .9584; H .9266 0 -.2396',
                    basis='cc-pvqz', spin=0, charge=0, cart=False, verbose=0,
                    max_memory=16000)
        mf = scf.RHF(mol).run(conv_tol=1e-12)
        assert mf.converged
        solver = getattr(cc, args.method)(mf, frozen=0)
        solver.max_memory = 16000
        eris = solver.ao2mo()
        _, t1, t2 = solver.init_amps(eris)
        start = time.perf_counter()
        updated = solver.update_amps(t1, t2, eris)
        elapsed = time.perf_counter()-start
        print('%s / %s / %d threads: first update %.6f s' %
              (args.method, args.backend, args.threads, elapsed))
        print('Correlation energy after one update: %.12f Eh' % solver.energy(*updated, eris))


if __name__ == '__main__':
    main()
