"""Run the supplementary DM API matrix after the SCF/CPKS pipeline ends."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def sample():
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        f = line.split()
        if f[0].startswith('cpu') and f[0][3:].isdigit():
            values = list(map(int, f[1:9]))
            result[int(f[0][3:])] = (sum(values), sum(values) - values[3] - values[4])
    return result


if __name__ == '__main__':
    while not (ROOT / 'cleanup.json').is_file():
        batch = json.loads((ROOT / 'runs/batch.json').read_text())
        if batch['status'] == 'failed':
            raise SystemExit('SCF failed; diagnose before supplementary tests')
        time.sleep(5)
    assert json.loads((ROOT / 'acceptance.json').read_text())['accepted']
    assert json.loads((ROOT / 'response/runs/acceptance.json').read_text())['accepted']
    preferred = list(range(14, 22)) + [c for c in range(28) if c not in range(14, 22) and c != 4]
    while True:
        a = sample()
        time.sleep(2)
        b = sample()
        busy = {c: 100 * (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in b}
        available = [c for c in preferred if max(busy[c], busy[c + 56]) < 10]
        if len(available) >= 16:
            affinity = available[:16]
            break
        print('Waiting for16 idle physical cores on socket0; currently', len(available), flush=True)
        time.sleep(30)
    (ROOT / 'dm/cpu-selection.json').write_text(json.dumps({'affinity_order': affinity,
        'busy_pct_selected_and_siblings': {c: busy[c] for c in affinity + [c + 56 for c in affinity]},
        'max_busy_pct_for_selection': 10, 'same_socket': True}, indent=2) + '\n')
    stages = {s['label']: s for s in json.loads((ROOT / 'stack.json').read_text())['stages']}
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('LD_LIBRARY_PATH', None)
    env.update(PYSCF_EXT_PATH='/dev/null', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def call(script, *args):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], env=env, check=True)
    call('dm/run_xc_campaign.py', '--protocol', ROOT / 'dm/protocol.json', '--versions',
         'parent=' + stages['03-ao']['source'], 'dm=' + stages['04-density-dm']['source'],
         '--python', sys.executable, '--affinity', ','.join(map(str, affinity)),
         '--fixtures', '/home/lyl/soft/pyscf-review-bench-20261026/fixtures', '--output', ROOT / 'dm/runs')
    call('dm/accept_xc_campaign.py', ROOT / 'dm/runs/batch.json', '--baseline', 'parent',
         '--output', ROOT / 'dm/runs/accepted')
    call('dm/native_gate.py')
    call('report.py')
    print('All90 SCFs,18 CPKS runs and48 direct-DM API processes accepted; reports regenerated.', flush=True)
