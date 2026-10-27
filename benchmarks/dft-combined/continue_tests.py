"""Finish the already-running XC batch, then run and accept remaining tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FIXTURES = '/home/lyl/soft/pyscf-review-bench-20261026/fixtures'
VERSIONS = ['baseline=/home/lyl/soft/pyscf-split-20261026/base',
            'combined=/home/lyl/soft/pyscf-split-20261026/combined']
AFFINITY = ','.join(map(str, list(range(5, 28)) + [3]))


def run(script, *args):
    print('START', script, *map(str, args), flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


if __name__ == '__main__':
    os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      PYSCF_EXT_PATH='/dev/null', TMPDIR=str(ROOT / 'tmp'))
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('LD_LIBRARY_PATH', None)
    while json.loads((ROOT / 'xc-v1/batch.json').read_text())['status'] == 'running':
        time.sleep(5)
    run('accept_xc_campaign.py', ROOT / 'xc-v1/batch.json', '--baseline', 'baseline',
        '--output', ROOT / 'xc-v1/accepted')
    run('accept_native.py', ROOT / 'xc-v1')
    run('run_micro_campaign.py', '--protocol', ROOT / 'protocols/micro-grid.json',
        '--versions', *VERSIONS, '--python', sys.executable, '--fixtures', FIXTURES,
        '--affinity', AFFINITY, '--output', ROOT / 'grid-sort-v1')
    run('accept_xc_campaign.py', ROOT / 'grid-sort-v1/batch.json', '--baseline', 'baseline',
        '--output', ROOT / 'grid-sort-v1/accepted')
    run('accept_native.py', ROOT / 'grid-sort-v1')
    run('run_end_to_end.py', '--protocol', ROOT / 'protocols/end-to-end.json',
        '--versions', *VERSIONS, '--fixtures', FIXTURES, '--output', ROOT / 'end-to-end-v1')
    run('accept_end_to_end.py', ROOT / 'end-to-end-v1')
    run('accept_native.py', ROOT / 'end-to-end-v1')
    print('ALL THREE CAMPAIGNS ACCEPTED', flush=True)
