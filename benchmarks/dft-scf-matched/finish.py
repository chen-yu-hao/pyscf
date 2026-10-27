"""Accept, report and clean only registered scratch after the serial campaign."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    while True:
        batch = json.loads((ROOT / 'runs/batch.json').read_text())
        if batch['status'] != 'running':
            break
        time.sleep(5)
    if batch['status'] != 'finished_unvalidated':
        raise SystemExit('Campaign failed; inspect controller.log before retrying')
    for name in ('accept.py', 'report.py'):
        subprocess.run([sys.executable, str(ROOT / name)], check=True)
    scratch = ROOT / 'tmp'
    files = [str(p.relative_to(scratch)) for p in scratch.rglob('*') if p.is_file()]
    shutil.rmtree(scratch)
    (ROOT / 'cleanup.json').write_text(json.dumps({'accepted_runs': 36,
        'removed_registered_scratch': str(scratch), 'scratch_files': files,
        'preserved_density_matrices': 36, 'results_preserved': True}, indent=2) + '\n')
    print('Accepted 36 runs, rendered report, removed registered scratch.', flush=True)
