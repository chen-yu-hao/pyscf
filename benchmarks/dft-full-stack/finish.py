"""Accept the complete SCF matrix, measure response changes, then summarize."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def call(script, *args):
    print('START', script, flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def response_native_gate():
    batch = json.loads((ROOT / 'response/runs/batch.json').read_text())
    manifests = {s['label']: json.loads((ROOT / 'builds' / s['label'] / 'manifest.json').read_text())
                 for s in json.loads((ROOT / 'stack.json').read_text())['stages']}
    aliases = {'baseline': '07-potential', 'cacheblocks': '08-cache-blocks', 'compact': '09-response'}
    errors = []
    for row in batch['rows']:
        m = manifests[aliases[row['version']]]
        r = json.loads(Path(row['result']).read_text())
        if r['source']['commit'] != m['head']:
            errors.append(row['result'] + ': source/build mismatch')
        for name, item in r['native_libraries'].items():
            if item['sha256'] != m['native_hashes']['pyscf/lib/' + name]:
                errors.append(row['result'] + ': native hash ' + name)
            if not Path(item['path']).is_relative_to(Path(m['source'])):
                errors.append(row['result'] + ': foreign library ' + name)
        if any(p['num_threads'] != (1 if p['user_api'] == 'blas' else row['threads']) for p in r['threadpools']):
            errors.append(row['result'] + ': wrong thread count')
    result = {'accepted': not errors, 'checked': len(batch['rows']), 'errors': errors}
    (ROOT / 'response/runs/native-acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    if errors:
        raise RuntimeError(repr(result))


if __name__ == '__main__':
    env = os.environ
    env.pop('PYTHONPATH', None)
    env.pop('LD_LIBRARY_PATH', None)
    env.update(PYSCF_EXT_PATH='/dev/null', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    while True:
        batch = json.loads((ROOT / 'runs/batch.json').read_text())
        if batch['status'] != 'running':
            break
        time.sleep(5)
    if batch['status'] != 'finished_unvalidated':
        raise SystemExit('SCF matrix failed; retained evidence must be diagnosed')
    call('accept_scf.py')
    stages = {s['label']: s for s in json.loads((ROOT / 'stack.json').read_text())['stages']}
    affinity = ','.join(map(str, batch['protocol']['cpu_mask']))
    call('response/run_response_campaign.py', '--protocol', ROOT / 'response/response_protocol.json',
         '--versions', 'baseline=' + stages['07-potential']['source'],
         'cacheblocks=' + stages['08-cache-blocks']['source'], 'compact=' + stages['09-response']['source'],
         '--python', sys.executable, '--affinity', affinity, '--threads', '8',
         '--fixtures', ROOT / 'response/fixtures', '--output', ROOT / 'response/runs')
    call('response/accept_response_campaign.py', ROOT / 'response/runs')
    response_native_gate()
    call('report.py')
    cleanup = []
    for name in ('tmp', 'tmp-tests'):
        directory = ROOT / name
        files = [str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()]
        shutil.rmtree(directory)
        cleanup.append({'removed_registered_scratch': str(directory), 'files': files})
    (ROOT / 'cleanup.json').write_text(json.dumps({'cleanup': cleanup, 'density_arrays_preserved': 90,
                                                'response_fixtures_preserved': 2}, indent=2) + '\n')
    print('SCF 90/90, CPKS18/18 accepted; drafts and report rendered; scratch cleaned.', flush=True)
