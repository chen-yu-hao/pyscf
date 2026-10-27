"""Stage accepted text evidence in a new directory of the evidence branch."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/lyl/soft/pyscf-dft-selection-20261026/evidence-repo')
DEST = REPO / 'benchmarks/dft-branch-scf'


def main():
    result = json.loads((ROOT / 'acceptance.json').read_text())
    assert result['accepted'] and result['accepted_runs'] == 18 and result['accepted_pairs'] == 9
    assert not DEST.exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.json')) + list(ROOT.glob('*.csv'))
    files += [ROOT / name for name in ('controller.log', 'finish.log')]
    files += list((ROOT / 'runs').glob('*.json')) + list((ROOT / 'provenance').glob('*.json'))
    mapping = {p: DEST / p.relative_to(ROOT) for p in files}
    mapping[ROOT / 'REPORT.md'] = DEST / 'README.md'
    for source, target in mapping.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    inventory = {'added': sorted(str(p.relative_to(REPO)) for p in mapping.values()),
                 'overwritten': [], 'removed': [], 'added_files_excluding_this_inventory': len(mapping),
                 'excluded': {'density_matrix_arrays': 18, 'compiled_libraries': 'all',
                              'worker_stdout_logs': 18, 'temporary_scratch': 'removed after acceptance'},
                 'local_archive': str(ROOT)}
    (DEST / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (ROOT / 'publication-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print('Prepared', len(mapping) + 1, 'new evidence files; no existing files changed.')


if __name__ == '__main__':
    main()
