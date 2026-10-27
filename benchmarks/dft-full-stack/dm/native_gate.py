"""Bind the direct-parent DM API measurements to their actual cumulative builds."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    batch = json.loads((ROOT / 'runs/batch.json').read_text())
    aliases = {'parent': '03-ao', 'dm': '04-density-dm'}
    versions = {v['label']: v for v in batch['versions']}
    errors = []
    for row in batch['rows']:
        manifest = json.loads((ROOT.parent / 'builds' / aliases[row['version']] / 'manifest.json').read_text())
        result = json.loads(Path(row['result']).read_text())
        libraries = {Path(x['path']).name: x for x in result['native_maps']}
        if result['source']['commit'] != manifest['head']:
            errors.append(row['result'] + ': source/build mismatch')
        for name in ('libdft.so', 'libcgto.so', 'libxc_itrf.so'):
            entry = libraries.get(name)
            if entry is None or entry['sha256'] != manifest['native_hashes']['pyscf/lib/' + name]:
                errors.append(row['result'] + ': native hash ' + name)
            elif not Path(entry['path']).is_relative_to(Path(versions[row['version']]['source'])):
                errors.append(row['result'] + ': native origin ' + name)
        if any(p['num_threads'] != (1 if p['user_api'] == 'blas' else row['threads']) for p in result['threadpools']):
            errors.append(row['result'] + ': thread count')
    report = {'accepted': not errors, 'checked': len(batch['rows']), 'errors': errors}
    (ROOT / 'runs/native-acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    if errors:
        raise SystemExit(2)
