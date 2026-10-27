"""Accept all runs before publishing grid-build and SCF speedups."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


def accept(folder):
    batch = json.loads((folder / 'batch.json').read_text())
    errors, groups, seen = [], {}, set()
    versions = {v['label']: v for v in batch['versions']}
    protocol = batch['protocol']
    expected = {(c['mode'], c['case'], c.get('xc'), c['threads'], v, repeat)
                for c in protocol['cells'] for v in versions for repeat in range(3)}
    if (batch['status'] != 'finished_unvalidated' or batch['completed'] != len(expected)
            or batch['expected'] != len(expected) or len(batch['rows']) != len(expected)):
        errors.append('Incomplete batch')
    for row in batch['rows']:
        key = (row['mode'], row['case'], row.get('xc'), row['threads'])
        identity = (*key, row['version'], row['repeat'])
        if identity in seen or identity not in expected:
            errors.append('Duplicate/unexpected task ' + repr(identity))
        seen.add(identity)
        result = json.loads(Path(row['result']).read_text())
        version = versions[row['version']]
        if (row['returncode'] or result['status'] != 'succeeded'
                or result['source']['commit'] != version['commit'] or result['source']['tracked_changes']
                or not Path(result['pyscf_import']).resolve().is_relative_to(Path(version['source']))
                or result['protocol_sha256'] != batch['protocol_sha256']
                or result['worker_sha256'] != batch['worker_sha256']):
            errors.append('Process/provenance failure ' + row['result'])
        cpus = sorted(protocol['affinity_order'][:row['threads']])
        if result['affinity'] != cpus or result['threads'] != row['threads']:
            errors.append('Wrong CPU placement ' + row['result'])
        samples = result.get('wall_seconds', [result.get('kernel_wall_seconds')])
        if any(x is None or not math.isfinite(x) or x <= 0 for x in samples + [row['process_wall_seconds']]):
            errors.append('Invalid timing ' + row['result'])
        if row['mode'] == 'grid-build':
            if len(samples) != 3 or not result['exact_fixture_grid']:
                errors.append('Invalid grid protocol ' + row['result'])
        elif not result['converged']:
            errors.append('SCF not converged ' + row['result'])
        groups.setdefault(key, {}).setdefault(row['version'], []).append((row, result))
    if seen != expected:
        errors.append('Missing tasks')
    comparisons = []
    shared = ('grid_points', 'grid_coords_sha256', 'grid_weights_sha256', 'grid_mask_sha256',
              'nao', 'numpy_version', 'python', 'affinity')
    for key, group in groups.items():
        reference = group['baseline'][0][1]
        worst = {'energy_hartree': 0., 'dm_absolute': 0.}
        for records in group.values():
            for row, result in records:
                if any(result[k] != reference[k] for k in shared):
                    errors.append('Inputs/settings mismatch ' + row['result'])
                if key[0] == 'scf':
                    with np.load(result['output_arrays']) as current, np.load(reference['output_arrays']) as base:
                        for field, gate in [('energy', 'energy_hartree'), ('dm', 'dm_absolute')]:
                            value, target = current[field], base[field]
                            if value.shape != target.shape or not np.isfinite(value).all():
                                errors.append('Nonfinite/shape mismatch ' + row['result'])
                                continue
                            delta = float(np.max(np.abs(value - target)))
                            worst[gate] = max(worst[gate], delta)
                            if delta > protocol['gates'][gate]:
                                errors.append(gate + ' failed: ' + row['result'])
        metrics = {'peak_rss_mib': lambda row, result: result['peak_rss_mib'],
                   'process_seconds': lambda row, result: row['process_wall_seconds']}
        if key[0] == 'scf':
            metrics['kernel_seconds'] = lambda row, result: result['kernel_wall_seconds']
        else:
            metrics['grid_seconds'] = lambda row, result: result['median_wall_seconds']
        comparison = dict(zip(('mode', 'case', 'xc', 'threads'), key))
        comparison.update(nao=reference['nao'], grid_points=reference['grid_points'], errors=worst)
        for name, get in metrics.items():
            for version, records in group.items():
                values = [get(*pair) for pair in records]
                comparison[version + '_' + name] = statistics.median(values)
                comparison[version + '_' + name + '_range'] = [min(values), max(values)]
            comparison[name + '_ratio'] = comparison['baseline_' + name] / comparison['combined_' + name]
        if key[0] == 'scf':
            for version, records in group.items():
                comparison[version + '_cycles'] = [result['cycles'] for _, result in records]
        comparisons.append(comparison)
    report = {'accepted': not errors, 'expected': len(expected), 'checked': len(seen),
              'failures': errors, 'gates': protocol['gates'], 'comparisons': comparisons}
    (folder / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    with (folder / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in comparisons))))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps({k: report[k] for k in ('accepted', 'expected', 'checked', 'failures')}))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    accept(Path(sys.argv[1]))
