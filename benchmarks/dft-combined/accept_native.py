"""Bind each measured process to the recorded build and thread settings."""
import json
from pathlib import Path
import sys


def accept(folder):
    root = Path(__file__).resolve().parent
    batch = json.loads((folder / 'batch.json').read_text())
    versions = {v['label']: v for v in batch['versions']}
    errors = []
    for row in batch['rows']:
        version = versions[row['version']]
        manifest = json.loads((root / 'builds' / (row['version'] + '.json')).read_text())
        expected = {Path(k).name: h for k, h in manifest['native_hashes'].items()}
        result = json.loads(Path(row['result']).read_text())
        if result['source']['commit'] != manifest['head']:
            errors.append(row['result'] + ': commit/build mismatch')
        native = {Path(x['path']).name: x for x in result['native_maps']}
        for name in ('libdft.so', 'libcgto.so', 'libxc_itrf.so', 'libcint.so'):
            entry = native.get(name)
            if entry is None or entry['sha256'] != expected.get(name):
                errors.append(row['result'] + ': native identity ' + name)
            elif not Path(entry['path']).is_relative_to(Path(version['source'])):
                errors.append(row['result'] + ': foreign library ' + name)
        pools = result.get('threadpools')
        if not pools or any(p['num_threads'] != (1 if p['user_api'] == 'blas' else row['threads']) for p in pools):
            errors.append(row['result'] + ': thread count')
    report = {'accepted': not errors, 'checked': len(batch['rows']), 'failures': errors}
    (folder / 'native-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    accept(Path(sys.argv[1]))
