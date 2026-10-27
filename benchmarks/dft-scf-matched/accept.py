"""Validate all four-version full-SCF runs before summarizing their timings."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
FINAL_LIMITS = {'density_rms': 1e-9, 'density_max': 1e-7,
                'energy_delta': 1e-7, 'gradient': 3e-5}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    versions = {v['label']: v for v in json.loads((ROOT / 'versions.json').read_text())}
    batch = json.loads((ROOT / 'runs/batch.json').read_text())
    errors, details, results, arrays = [], [], {}, {}
    expected = set(itertools.product([c['id'] for c in protocol['cases']], versions, range(3)))
    seen = set()
    def check(ok, message):
        if not ok:
            errors.append(message)
    check(batch['status'] == 'finished_unvalidated', 'Batch must finish successfully')
    check(batch['expected'] == batch['completed'] == len(batch['rows']) == len(expected), 'Incomplete batch')
    check(batch['protocol'] == protocol and batch['versions'] == list(versions.values()), 'Protocol/version manifest changed')
    check(digest(ROOT / 'pyscf_case.py') == protocol['source_worker_sha256'], 'Worker changed')
    for version in versions.values():
        root = Path(version['root'])
        for relative, expected_hash in version['source_files'].items():
            check(digest(root / relative) == expected_hash, 'Source changed: ' + str(root / relative))
        for filename, expected_hash in version['native_files'].items():
            check(digest(filename) == expected_hash, 'Native library changed: ' + filename)
        if version['commit']:
            current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip()
            check(current == version['commit'] and not dirty, 'Git source mismatch: ' + version['label'])
    specs = {c['id']: c for c in protocol['cases']}
    for row in batch['rows']:
        key = row['case'], row['engine'], row['repeat']
        context = '-'.join(map(str, key))
        check(key not in seen and key in expected, 'Duplicate/unexpected: ' + context)
        seen.add(key)
        before = len(errors)
        result = json.loads(Path(row['result']).read_text())
        results[key] = result
        version, spec = versions[row['engine']], specs[row['case']]
        package = Path(version['root']) / 'pyscf'
        check(row['returncode'] == 0 and result['strict'], context + ': failed process/SCF')
        check(result['pyscf_path'] == str(package / '__init__.py'), context + ': wrong import')
        check(result['case'] == spec['id'] and result['basis'] == spec['basis']
              and result['spin'] == spec['spin'] and result['nao'] == spec['expected_nao']
              and result['xc'] == 'MN15', context + ': wrong scientific inputs')
        check(result['affinity'] == protocol['cpu_mask'] == row['affinity']
              and result['cores'] == row['cores'] == 8, context + ': affinity/thread mismatch')
        check(result['grid_nominal'] == [99, 590] and result['cutoff'] == 1e-10
              and result['numint_cutoff'] == 1e-8, context + ': grid/cutoff mismatch')
        check(result['grid_points'] == {'water4': 700920, 'o2': 116824, 'water12': 2102760}[row['case']],
              context + ': unexpected grid size')
        for value in (row['process_wall_s'], result['kernel_wall_s'], result['energy'], row['process_cpu_seconds']):
            check(math.isfinite(value), context + ': nonfinite value')
        check(row['process_wall_s'] > result['kernel_wall_s'] > 0, context + ': invalid timing boundary')
        final = result['history'][-1]
        check(all(math.isfinite(final[k]) and final[k] < limit for k, limit in FINAL_LIMITS.items()),
              context + ': strict final convergence failed')
        maps = result['native_maps']
        check(maps and all(Path(p).is_relative_to(package) for p in maps), context + ': foreign native library')
        for name in ('libcint.so', 'libcgto.so', 'libdft.so', 'libxc_itrf.so', 'libcvhf.so'):
            entries = [p for p in maps if Path(p).name == name]
            check(len(entries) == 1 and str(Path(entries[0]).resolve()) in version['native_files'],
                  context + ': native identity missing ' + name)
        pools = result['threadpools']
        check({p['user_api'] for p in pools} == {'blas', 'openmp'}
              and all(p['num_threads'] == (1 if p['user_api'] == 'blas' else 8) for p in pools),
              context + ': wrong runtime thread counts')
        electrons = sum({'H': 1, 'O': 8}[s] for s, _ in spec['atoms_bohr']) - spec['charge']
        target_nelec = [(electrons + spec['spin']) / 2, (electrons - spec['spin']) / 2] if spec['spin'] else electrons
        nelec_error = float(np.max(np.abs(np.asarray(result['nelec_overlap']) - target_nelec)))
        check(nelec_error <= 1e-8, context + ': electron count error')
        dm = np.load(Path(row['result']).with_suffix('.dm.npy'))
        target_shape = (2, spec['expected_nao'], spec['expected_nao']) if spec['spin'] else (spec['expected_nao'], spec['expected_nao'])
        check(dm.shape == target_shape and dm.dtype == np.float64 and np.isfinite(dm).all(), context + ': invalid DM')
        arrays[key] = dm
        ticks0, ticks1 = row['cpu_ticks_before'], row['cpu_ticks_after']
        fractions = {c: (ticks1[c]['busy'] - ticks0[c]['busy']) / max(1, ticks1[c]['total'] - ticks0[c]['total']) for c in ticks0}
        siblings = [fractions['cpu' + str(c)] for c in protocol['smt_siblings_monitored']]
        details.append({'case': row['case'], 'engine': row['engine'], 'repeat': row['repeat'],
                        'accepted': len(errors) == before, 'nelec_error': nelec_error,
                        'final': final, 'mean_sibling_busy_fraction': statistics.mean(siblings),
                        'max_sibling_busy_fraction': max(siblings),
                        'average_used_cores': row['process_cpu_seconds'] / row['process_wall_s']})
    check(seen == expected, 'Missing run identities')
    pairs = []
    summaries = []
    for case in specs:
        for a, b in itertools.combinations(versions, 2):
            for repeat in range(3):
                ka, kb = (case, a, repeat), (case, b, repeat)
                if ka not in results or kb not in results:
                    continue
                ra, rb = results[ka], results[kb]
                delta = arrays[ka] - arrays[kb]
                pair = {'case': case, 'a': a, 'b': b, 'repeat': repeat,
                        'energy_delta_hartree': abs(ra['energy'] - rb['energy']),
                        'density_rms': float(np.sqrt(np.mean(delta * delta))),
                        'density_max': float(np.max(np.abs(delta))),
                        'grid_identical': all(ra[k] == rb[k] for k in ('grid_points', 'nonzero_weight_points',
                                                                    'grid_coords_digest', 'grid_weights_digest')),
                        'spin_square_delta': abs(ra.get('spin_square', [0])[0] - rb.get('spin_square', [0])[0])}
                pair['accepted'] = (pair['energy_delta_hartree'] <= 1e-8 and pair['density_rms'] <= 1e-9
                                    and pair['density_max'] <= 1e-7 and pair['spin_square_delta'] <= 1e-8
                                    and pair['grid_identical'])
                check(pair['accepted'], 'Cross-version numerical gate: ' + repr((case, a, b, repeat)))
                pairs.append(pair)
        summary = {'case': case, 'nao': specs[case]['expected_nao'], 'basis': specs[case]['basis'],
                   'method': 'UKS' if specs[case]['spin'] else 'RKS', 'versions': {}}
        for version in versions:
            group = [row for row in batch['rows'] if row['case'] == case and row['engine'] == version]
            records = [results[(case, version, row['repeat'])] for row in group]
            energy_range = max(r['energy'] for r in records) - min(r['energy'] for r in records)
            check(energy_range <= 1e-9, 'Repeat energy drift: ' + case + ' ' + version)
            fields = {'process_wall_s': [r['process_wall_s'] for r in group],
                      'kernel_wall_s': [r['kernel_wall_s'] for r in records],
                      'peak_rss_mib': [r['peak_rss_kib'] / 1024 for r in records]}
            metrics = {k: {'median': statistics.median(v), 'range': [min(v), max(v)], 'samples': v}
                       for k, v in fields.items()}
            metrics.update(cycles=[r['cycles'] for r in records], polish_cycles=[r['polish_cycles'] for r in records],
                           energy_range_hartree=energy_range, grid_points=records[0]['grid_points'])
            summary['versions'][version] = metrics
        summary['ratios'] = {a + '/' + b: {boundary: summary['versions'][a][boundary]['median'] /
                                         summary['versions'][b][boundary]['median']
                                         for boundary in ('process_wall_s', 'kernel_wall_s')}
                             for a, b in [('legacy-original', 'full-optimized'), ('upstream', 'combined'),
                                          ('legacy-original', 'combined'), ('combined', 'full-optimized'),
                                          ('legacy-original', 'upstream')]}
        summaries.append(summary)
    report = {'accepted': not errors, 'expected_runs': len(expected), 'accepted_runs': sum(r['accepted'] for r in details),
              'expected_pairs': 54, 'accepted_pairs': sum(p['accepted'] for p in pairs),
              'errors': errors, 'summaries': summaries, 'pairs': pairs, 'runs': details,
              'protocol': protocol, 'versions': list(versions.values()),
              'max_errors': {k: max(p[k] for p in pairs) for k in ('energy_delta_hartree', 'density_rms', 'density_max', 'spin_square_delta')}}
    (ROOT / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('accepted', 'expected_runs', 'accepted_runs', 'expected_pairs', 'accepted_pairs', 'errors', 'max_errors')}, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
