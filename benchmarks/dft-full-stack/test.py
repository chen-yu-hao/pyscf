"""Test changed features at each stage, then test their complete combination."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/lyl/soft/conda-envs/pyscf-new/bin/python'
FEATURES = {
    '01-grid': ['pyscf.dft.test.test_grids'],
    '02-libxc': ['pyscf.dft.test.test_libxc'],
    '03-ao': ['pyscf.gto.test.test_grid_exp1'],
    '04-density-dm': ['pyscf.dft.test.test_numint', 'pyscf.dft.test.test_numint_rho_fused'],
    '05-density-mo': ['pyscf.dft.test.test_numint_rho', 'pyscf.dft.test.test_numint_rho_stream'],
    '06-sparse-tiles': ['pyscf.dft.test.test_numint_pair_tiles'],
    '07-potential': ['pyscf.dft.test.test_numint_mgga_vmat', 'pyscf.dft.test.test_numint_tau',
                     'pyscf.dft.test.test_numint_xc_response_assembly'],
    '08-cache-blocks': ['pyscf.dft.test.test_numint_cache_blocks'],
    '09-response': ['pyscf.dft.test.test_xc_cache'],
}


def main():
    stack = json.loads((ROOT / 'stack.json').read_text())
    out = ROOT / 'tests'
    out.mkdir()
    tmp = ROOT / 'tmp-tests'
    tmp.mkdir()
    records = []
    modules = list(dict.fromkeys(m for group in FEATURES.values() for m in group))
    tasks = [(s, FEATURES[s['label']], s['label']) for s in stack['stages'][1:]]
    tasks.append((stack['stages'][-1], modules, 'full-combination'))
    for stage, selected, name in tasks:
        env = os.environ.copy()
        env.pop('LD_LIBRARY_PATH', None)
        env.update(PYTHONPATH=stage['source'], PYSCF_EXT_PATH='/dev/null', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR=str(tmp),
                   PYSCF_TMPDIR=str(tmp), OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
        cmd = ['taskset', '-c', '20-23', PYTHON, '-m', 'unittest', *selected]
        started = time.perf_counter()
        with (out / (name + '.log')).open('w') as stream:
            p = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        log = (out / (name + '.log')).read_text()
        counts = re.search(r'Ran (\d+) tests?', log)
        skip = re.search(r'OK \(skipped=(\d+)\)', log)
        records.append({'label': name, 'head': stage['head'], 'command': cmd,
                        'returncode': p.returncode, 'tests': int(counts.group(1)) if counts else None,
                        'skipped': int(skip.group(1)) if skip else 0,
                        'duration_s': time.perf_counter() - started})
        (ROOT / 'tests.json').write_text(json.dumps(records, indent=2) + '\n')
        print('TESTED', name, records[-1]['tests'], 'skipped', records[-1]['skipped'], flush=True)
        if p.returncode:
            raise RuntimeError('Regression failed ' + name)
    final = Path(stack['stages'][-1]['source'])
    lint = []
    commands = [['/home/lyl/soft/pyscf-pr-tools/python/bin/ruff', 'check', '--config', '.ruff.toml', 'pyscf/dft', 'pyscf/lib', 'pyscf/scf', 'pyscf/gto'],
                ['/home/lyl/soft/pyscf-pr-tools/python/bin/ruff', 'check', '--select', 'NPY', '--ignore', 'NPY002', 'pyscf/dft', 'pyscf/scf', 'pyscf/gto'],
                ['flake8', '--config', '.flake8', 'pyscf/dft', 'pyscf/scf', 'pyscf/gto'],
                ['pycodestyle', '--select=E111,W191,W293', 'pyscf/dft', 'pyscf/scf', 'pyscf/gto']]
    for cmd in commands:
        p = subprocess.run(cmd, cwd=final, capture_output=True, text=True)
        lint.append({'command': cmd, 'returncode': p.returncode, 'output': p.stdout + p.stderr})
    (ROOT / 'lint.json').write_text(json.dumps(lint, indent=2) + '\n')
    if any(r['returncode'] for r in lint):
        raise RuntimeError('Lint failed; inspect lint.json')


if __name__ == '__main__':
    main()
