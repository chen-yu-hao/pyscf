"""Package accepted stack evidence and supersede the old selection recommendation."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'evidence-repo'
DEST = REPO / 'benchmarks/dft-full-stack'


def main():
    scf = json.loads((ROOT / 'acceptance.json').read_text())
    response = json.loads((ROOT / 'response/runs/acceptance.json').read_text())
    dm = json.loads((ROOT / 'dm/runs/accepted/acceptance.json').read_text())
    assert scf['accepted'] and scf['accepted_runs'] == 90 and scf['accepted_pairs'] == 405
    assert response['accepted'] and response['completed'] == 18
    assert dm['accepted'] and dm['completed'] == 48
    assert json.loads((ROOT / 'dm/runs/native-acceptance.json').read_text())['accepted']
    assert not DEST.exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.json')) + list(ROOT.glob('*.csv'))
    files += [ROOT / name for name in ('build.log', 'test.log', 'scf.log', 'finish.log', 'dm.log', 'prepare.log', 'mo-only.patch')]
    for directory in ('drafts', 'tests', 'response', 'dm'):
        files += [p for p in (ROOT / directory).rglob('*')
                  if p.suffix in ('.py', '.json', '.csv', '.md', '.log') and '__pycache__' not in p.parts]
    files += list((ROOT / 'runs').glob('*.json'))
    files += list((ROOT / 'builds').glob('*/manifest.json'))
    mapping = {p: DEST / p.relative_to(ROOT) for p in files}
    mapping[ROOT / 'REPORT.md'] = DEST / 'README.md'
    mapping[ROOT / 'REVIEW_GUIDE.md'] = DEST / 'REVIEW_GUIDE.md'
    for source, target in mapping.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == '.csv':
            target.write_bytes(target.read_bytes().replace(b'\r\n', b'\n'))
    previous = REPO / 'benchmarks/dft-speed-selection/README.md'
    old = previous.read_text()
    assert 'Superseded by the complete-implementation stack' not in old
    previous.write_text('> **Superseded by the complete-implementation stack:** '
       '[nine cumulative steps and full-SCF incremental measurements](../dft-full-stack/README.md). '
       'The three-change recommendation below is retained as historical evidence from isolated XC selection; '
       'the current plan preserves the complete implementation and evaluates parent-to-child full workloads.\n\n' + old)
    inventory = {'added': sorted(str(p.relative_to(REPO)) for p in mapping.values()),
                 'modified': ['benchmarks/dft-speed-selection/README.md'], 'removed': [],
                 'modified_scope': 'Only a superseding link/notice; all old benchmark values and files retained.',
                 'added_files_excluding_this_inventory': len(mapping), 'csv_format': 'LF line endings; numeric values unchanged.',
                 'whitespace_check_exclusions': {'benchmarks/dft-full-stack/mo-only.patch':
                   'Raw unified-diff context blank lines contain a required leading space; patch bytes retained unchanged.'},
                 'excluded': {'full_scf_density_arrays': 90, 'response_binary_fixtures': 2,
                              'DM_XC_output_arrays': 48,
                              'compiled_libraries': 'all', 'cmake_build_products': 'all',
                              'SCF_worker_stdout_logs': 90, 'temporary_scratch': 'removed after acceptance'},
                 'local_archive': str(ROOT)}
    (DEST / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2) + '\n')
    (ROOT / 'publication-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print('Prepared', len(mapping) + 1, 'new files and one superseding notice; no removals.')


if __name__ == '__main__':
    main()
