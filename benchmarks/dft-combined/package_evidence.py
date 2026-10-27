"""Prepare an additive public evidence directory after all measurements pass."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/lyl/soft/pyscf-dft-selection-20261026/evidence-repo')
DEST = REPO / 'benchmarks/dft-combined'


def main():
    reports = json.loads((ROOT / 'results.json').read_text())
    assert all(report['accepted'] for report in reports.values())
    if DEST.exists():
        raise RuntimeError('Use an untouched evidence namespace')
    files = list(ROOT.glob('*.py'))
    files += [ROOT / name for name in ('REPRODUCE.md', 'comparison.csv', 'results.json',
              'components.json', 'push.json', 'machine.json', 'tests.json', 'tests.log',
              'lint.json', 'lint-ci.json', 'xc-v1.controller.log', 'continue_tests.log')]
    for subdir in ('protocols', 'builds', 'xc-v1', 'grid-sort-v1', 'end-to-end-v1'):
        files += [p for p in (ROOT / subdir).rglob('*') if p.suffix in ('.json', '.csv')]
    copies = {p: DEST / p.relative_to(ROOT) for p in files}
    copies[ROOT / 'REPORT.md'] = DEST / 'README.md'
    fixture_root = Path('/home/lyl/soft/pyscf-review-bench-20261026/fixtures')
    for case in ('water12-rks', 'water11oh-uks', 'water20-rks', 'water19oh-uks'):
        copies[fixture_root / (case + '.json')] = DEST / 'fixtures' / (case + '.json')
    before = sorted(str(p.relative_to(REPO)) for p in (REPO / 'benchmarks').rglob('*') if p.is_file())
    for source, target in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == '.csv':
            target.write_bytes(target.read_bytes().replace(b'\r\n', b'\n'))
    after = sorted(str(p.relative_to(REPO)) for p in (REPO / 'benchmarks').rglob('*') if p.is_file())
    omitted = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.npz', '.npy')]
    manifest = {'added': sorted(set(after) - set(before)), 'removed': sorted(set(before) - set(after)),
                'overwritten': [], 'previous_benchmark_files': len(before),
                'added_files_excluding_this_inventory': len(copies),
                'omitted': {'local_numerical_array_files': len(omitted),
                            'local_numerical_array_bytes': sum(p.stat().st_size for p in omitted),
                            'other': ['compiled libraries', 'binary input fixtures', 'worker stdout logs',
                                      'temporary files', 'Python bytecode']},
                'raw_archive': str(ROOT), 'csv_format': 'Public CSV line endings normalized to LF; values unchanged.',
                'note': 'All original evidence files retained; only a new subtree added.'}
    assert not manifest['removed'] and len(manifest['added']) == len(copies)
    (DEST / 'INVENTORY.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (ROOT / 'publication-inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('added_files_excluding_this_inventory', 'removed', 'overwritten', 'omitted')}))


if __name__ == '__main__':
    main()
