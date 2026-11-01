"""Run the three original scanner/gradient assertions at one effective thread.

--old-guess-diagnostic is causal isolation only and can never satisfy the normal
source regression gate. No assertion, tolerance, or selected test is changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


TESTS = [
    'pyscf/solvent/test/test_ddcosmo_tdscf_grad.py::KnownValues::test_scanner',
    'pyscf/grad/test/test_tdrks_grad.py::KnownValues::test_tda_triplet_lda',
    'pyscf/grad/test/test_tdrhf_grad.py::KnownValues::test_tda_singlet',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--old-guess-diagnostic', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    status_path = args.status_file.resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(expected_sha=args.expected_sha, tests=TESTS,
                  diagnostic_only=args.old_guess_diagnostic, exit_code=1)
    try:
        sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if sha != args.expected_sha:
            raise RuntimeError('Unexpected source SHA: ' + sha)
        result['source_sha'] = sha
        os.chdir(root)
        sys.path.insert(0, str(root))
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[key] = '1'
        os.environ['PYSCF_NUMA'] = 'off'
        os.environ['OMP_DYNAMIC'] = 'FALSE'
        import pyscf
        from pyscf import lib
        from pyscf.scf import hf
        import pytest
        from threadpoolctl import threadpool_info, threadpool_limits
        if not Path(pyscf.__file__).resolve().is_relative_to(root):
            raise RuntimeError('Wrong PySCF checkout imported')
        if args.old_guess_diagnostic:
            if not hasattr(hf, '_guess_from_last_geometry'):
                raise RuntimeError('Candidate has no projected-guess function to diagnose')
            hf._guess_from_last_geometry = lambda mf, mol: mf.make_rdm1()
        lib.num_threads(1)
        with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
            result['effective_threads'] = lib.num_threads()
            result['threadpools'] = threadpool_info()
            assert result['effective_threads'] == 1
            assert all(p['num_threads'] == 1 for p in result['threadpools'])
            print(json.dumps(result, indent=2), flush=True)
            result['exit_code'] = int(pytest.main(['-q', *TESTS]))
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        print(result['error'], file=sys.stderr, flush=True)
    finally:
        status_path.write_text(json.dumps(result, indent=2) + '\n')
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
