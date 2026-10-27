#!/usr/bin/env python3
"""Validate every CPKS run and compare physical tensors before summarizing timings."""
import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import statistics


def flat(matrix):
    return list(itertools.chain.from_iterable(matrix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    batch = json.loads((args.campaign/'batch.json').read_text())
    versions = {v['label']: v for v in batch['versions']}
    cases = {c['name']: c for c in batch['cases']}
    expected_keys = set(itertools.product(cases, batch['threads'], versions, range(batch['repeats'])))
    errors, runs = [], {}
    if batch['status'] != 'finished_unvalidated':
        errors.append('Campaign has not finished normally: '+batch['status'])
    if len(batch['rows']) != len(expected_keys) or batch['completed'] != len(expected_keys):
        errors.append('Run count does not match complete campaign matrix')
    fixture_hashes = collections.defaultdict(set)
    native_hashes = collections.defaultdict(set)
    for row in batch['rows']:
        key = (row['case'], row['threads'], row['version'], row['repeat'])
        label = '/'.join(map(str, key))
        if key not in expected_keys or key in runs:
            errors.append(label+': unexpected or duplicate run')
            continue
        path = Path(row['result'])
        if row['returncode'] != 0 or not path.exists():
            errors.append(label+': unsuccessful child or missing result')
            continue
        result = json.loads(path.read_text())
        runs[key] = result
        fixture_hashes[row['case']].add(result['fixture_sha256'])
        native_hashes[row['version']].add(tuple(sorted(
            (name, item['sha256']) for name, item in result['native_libraries'].items())))
        checks = {
            'clean source': result['source']['tracked_source_dirty'] is False,
            'commit': result['source']['commit'] == versions[row['version']]['commit'] == row['commit'],
            'source path': result['source']['source'] == versions[row['version']]['source'],
            'fixture': result['fixture_sha256'] == row['fixture_sha256'],
            'affinity': result['affinity'] == sorted(row['affinity']),
            'omp threads': result['omp_threads'] == row['threads'],
            'blas threads': all(p['num_threads'] == 1 for p in result['threadpools'] if p.get('user_api') == 'blas'),
            'nao': result['fixture_metadata']['nao'] == cases[row['case']]['expected_nao'],
            'functional': result['fixture_metadata']['xc'] == batch['protocol']['functional'],
            'grid level': result['fixture_metadata']['grid_level'] == batch['protocol']['grid_level'],
            'spin': result['fixture_metadata']['spin'] == cases[row['case']]['spin'],
            'reported acceptance': result['scientific_accepted'] is True,
            'residual': result['residual_max'] < 2e-8,
            'polarizability symmetry': result['polarizability_symmetry_error'] < 2e-7,
            'finite response': all(math.isfinite(x) for x in flat(result['polarizability_au'])),
            'response timing': 0 < result['cache_wall_s'] < result['response_wall_s'],
            'memory sampling': result['response_peak_rss']['rss_sample_count'] > 0,
        }
        errors.extend(label+': '+name+' failed' for name, ok in checks.items() if not ok)
    if set(runs) != expected_keys:
        errors.append('Missing accepted result keys: '+repr(sorted(expected_keys-set(runs))))
    for case, values in fixture_hashes.items():
        if len(values) != 1:
            errors.append(case+': fixtures differ across versions/repeats')
    for version, values in native_hashes.items():
        if len(values) != 1:
            errors.append(version+': native libraries changed across runs')
    pairs = []
    version_order = list(versions)
    for case, threads, repeat in itertools.product(cases, batch['threads'], range(batch['repeats'])):
        for before, after in zip(version_order, version_order[1:]):
            ka, kb = (case, threads, before, repeat), (case, threads, after, repeat)
            if ka not in runs or kb not in runs:
                continue
            a, b = runs[ka], runs[kb]
            delta = max(abs(x-y) for x,y in zip(flat(a['polarizability_au']), flat(b['polarizability_au'])))
            pairs.append({'case': case, 'threads': threads, 'repeat': repeat,
                          'reference': before, 'candidate': after, 'polarizability_max_abs_au': delta,
                          'accepted': delta < 2e-7})
            if delta >= 2e-7:
                errors.append(f'{case}/t{threads}/r{repeat} {before}->{after}: polarizability gate failed')
        baseline = runs.get((case, threads, version_order[0], repeat))
        candidate = runs.get((case, threads, version_order[-1], repeat))
        if baseline and candidate:
            delta = max(abs(x-y) for x,y in zip(flat(baseline['polarizability_au']), flat(candidate['polarizability_au'])))
            if delta >= 2e-7:
                errors.append(f'{case}/t{threads}/r{repeat}: final-versus-base polarizability gate failed')
    groups = []
    for case, threads, version in itertools.product(cases, batch['threads'], versions):
        values = [runs[(case, threads, version, i)] for i in range(batch['repeats'])
                  if (case, threads, version, i) in runs]
        if len(values) != batch['repeats']:
            continue
        item = {'case': case, 'threads': threads, 'version': version,
                'commit': versions[version]['commit'], 'repeats': len(values),
                'cache_type': sorted({v['cache']['type'] for v in values}),
                'cache_bytes': sorted({v['cache']['unique_array_bytes'] for v in values})}
        metrics = {'response_wall_s': lambda x:x['response_wall_s'],
                   'cache_wall_s': lambda x:x['cache_wall_s'],
                   'solver_wall_s': lambda x:x['solver_wall_s'],
                   'sampled_peak_rss_mib': lambda x:x['response_peak_rss']['rss_sampled_peak_mib'],
                   'sampled_rss_increment_mib': lambda x:x['response_peak_rss']['rss_peak_increment_mib'],
                   'process_hwm_mib': lambda x:x['process_hwm_after_response_mib']}
        for name, getter in metrics.items():
            numbers = [getter(v) for v in values]
            item[name] = {'median': statistics.median(numbers), 'min': min(numbers), 'max': max(numbers)}
        item['solver_action_calls'] = [v['solver_action_calls'] for v in values]
        groups.append(item)
    for item in groups:
        references = [g for g in groups if g['case'] == item['case'] and g['threads'] == item['threads']]
        item['speedup_vs_base'] = references[0]['response_wall_s']['median']/item['response_wall_s']['median']
        pos = version_order.index(item['version'])
        item['speedup_vs_parent'] = references[max(pos-1,0)]['response_wall_s']['median']/item['response_wall_s']['median']
    interference = []
    for row in batch['rows']:
        hz = row['clock_ticks_per_second']
        seconds = 0
        for cpu, after in row['cpu_ticks_after'].items():
            before = row['cpu_ticks_before'][cpu]
            delta = [b-a for a,b in zip(before, after)]
            seconds += (sum(delta[:8])-delta[3]-delta[4])/hz
        interference.append({'case': row['case'], 'threads': row['threads'], 'version': row['version'],
                             'repeat': row['repeat'], 'monitored_busy_cpu_s': seconds,
                             'child_cpu_s': row['child_cpu_s'],
                             'busy_minus_child_cpu_s': seconds-row['child_cpu_s']})
    report = {'schema': 1, 'accepted': not errors, 'completed': len(runs), 'expected': len(expected_keys),
              'errors': errors, 'groups': groups, 'adjacent_pairs': pairs, 'cpu_diagnostics': interference,
              'max_adjacent_polarizability_delta_au': max((p['polarizability_max_abs_au'] for p in pairs), default=None),
              'memory_caveat': 'VmRSS sampled every 5 ms can miss brief transients; process HWM includes imports and input loading',
              'cpu_caveat': 'Shared host. Busy CPU seconds on selected cores and SMT siblings minus child CPU seconds are a diagnostic, not proof of exclusive occupancy.'}
    (args.campaign/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Analytic CPKS benchmark', '', f"Accepted: **{not errors}**; results **{len(runs)}/{len(expected_keys)}**.", '',
             '| Case | Threads | Version | CPKS total s | Cache s | Solver s | Peak RSS MiB | RSS increment MiB | vs base | vs parent |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['case']} | {g['threads']} | {g['version']} | {g['response_wall_s']['median']:.4f} | {g['cache_wall_s']['median']:.4f} | {g['solver_wall_s']['median']:.4f} | {g['sampled_peak_rss_mib']['median']:.1f} | {g['sampled_rss_increment_mib']['median']:.1f} | {g['speedup_vs_base']:.3f}x | {g['speedup_vs_parent']:.3f}x |")
    lines.extend(['', 'Medians of three fresh processes when the campaign uses its default three repeats. Minimum and maximum timings, commit identifiers, persistent cache bytes, and all numerical gates are in acceptance.json. Full cache construction and iterative Coulomb/XC response are included. SCF, imports and input loading are excluded from timing.', '', report['memory_caveat'], '', report['cpu_caveat']])
    if errors:
        lines.extend(['', 'Validation failures:', '']+['- '+e for e in errors])
    (args.campaign/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'accepted': not errors, 'completed': len(runs), 'expected': len(expected_keys), 'errors': errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
