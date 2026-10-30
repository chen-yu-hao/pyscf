"""Reopen every array; no dependence on a summary's claimed verification result."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np

BASE = 'b5b4f1c711b63c1dbb4fb409e979f53852601eca'


def audit(folder, candidate, smoke=False):
    expected = ({'smoke': {'MN15-n4-t2'}} if smoke else {
        'full': {f'MN15-n{n}-t{t}' for n in (4, 8) for t in (1, 2, 4)},
        'dense-control': {'MN15-n2-t2'}, 'gga-control': {'PBE-n4-t2'}})
    repeat = 1 if smoke else 3
    report = dict(cases=[], records=0, max_energy_abs=0., max_abs={},
                  has_observed_merging=False, errors=[], acceptance='pending')
    for group, expected_cases in expected.items():
        path = folder/group
        summary = json.loads((path/'summary.json').read_text())
        assert not summary['errors'], summary['errors']
        assert len(summary['cases']) == len(expected_cases)
        assert {c['case'] for c in summary['cases']} == expected_cases
        for case in summary['cases']:
            rows = case['trials']
            assert len(rows) == 2*(repeat+1)
            ref = rows[0]
            assert ref['variant'] == 'baseline'
            with np.load(path/Path(ref['arrays']).name, allow_pickle=False) as val:
                reference = {k:val[k] for k in val.files}
            for row in rows:
                variant = row['variant']
                assert variant in ('baseline', 'candidate')
                assert row['source_sha'] == (BASE if variant == 'baseline' else candidate)
                assert row['converged'] and isinstance(row.get('verification'), dict)
                assert case['case'] == f"{row['xc']}-n{row['n_water']}-t{row['threads']}"
                assert row['basis'] == 'def2-TZVPP' and row['grid_level'] == 6
                assert row['nao'] == 59*row['n_water']
                assert row['grid_sha256'] == ref['grid_sha256']
                assert row['grid_points'] == ref['grid_points']
                assert row['numa'] == 'off' and not row['used_ao_cache']
                assert row['dense_blas'] == row['mgga_pack_simd'] == '1'
                assert row['tile_screen'] == row['jk_block_bound'] == '0'
                assert row['coalesce'] == str(int(variant == 'candidate'))
                assert row['timed_diagnostic_counters'] is False
                assert row['used_incore_eri'] == ref['used_incore_eri']
                assert math.isfinite(row['scf_seconds']) and row['scf_seconds'] > 0
                for pool in row['threadpools']:
                    if pool['user_api'] == 'blas':
                        assert pool['num_threads'] == 1
                    elif pool['user_api'] == 'openmp':
                        assert pool['num_threads'] == row['threads']
                delta = abs(row['energy_hartree']-ref['energy_hartree'])
                assert math.isfinite(delta) and delta <= 1e-8
                report['max_energy_abs'] = max(report['max_energy_abs'], delta)
                with np.load(path/Path(row['arrays']).name, allow_pickle=False) as val:
                    assert all(np.isfinite(val[k]).all() for k in val.files)
                    np.testing.assert_array_equal(val['fixed_dm'], reference['fixed_dm'])
                    for key in ('ne', 'exc', 'vxc', 'vj', 'vk'):
                        np.testing.assert_allclose(val[key], reference[key], atol=1e-9, rtol=1e-10)
                        err = float(np.max(np.abs(val[key]-reference[key])))
                        report['max_abs'][key] = max(report['max_abs'].get(key, 0.), err)
                if not row['profiled']:
                    assert not row['stages_seconds'] and not row['c_dispatch_counts']
                elif row['xc'] == 'MN15' and row['n_water'] >= 4:
                    assert row['c_dispatch_counts'].get('sparse_ncomp4_returned1', 0) > 0
                if variant == 'candidate':
                    diagnostic = row['coalesce_diagnostic']
                    assert diagnostic is not None
                    assert diagnostic['raw_blocks'] == diagnostic['batches']+diagnostic['joined']
                    assert 0 <= diagnostic['widest'] <= 4
                    if row['xc'] == 'MN15' and row['n_water'] >= 4:
                        assert diagnostic['raw_blocks'] > 0
                    report['has_observed_merging'] |= diagnostic['joined'] > 0
                report['records'] += 1
            medians, samples = {}, {}
            for v in ('baseline', 'candidate'):
                times = [r['scf_seconds'] for r in rows if r['variant']==v and not r['profiled']]
                assert len(times) == repeat
                assert sum(r['variant']==v and r['profiled'] for r in rows) == 1
                medians[v] = statistics.median(times)
                assert math.isclose(medians[v], case['median_s'][v], rel_tol=1e-12)
                samples[v] = times
            report['cases'].append(dict(group=group, case=case['case'], samples=samples,
                                       median_s=medians, speedup=medians['baseline']/medians['candidate']))
    assert report['records'] == (4 if smoke else 64)
    for name in ('coalesce-native.log', 'regression-0.log', 'regression-1.log', 'jk-regression.log'):
        text = (folder/name).read_text()
        assert re.search(r'\b\d+ passed\b', text), name
        assert not re.search(r'\b(?:FAILED|ERROR|ERRORS)\b|\b[1-9]\d* failed\b', text), name
        report[name] = text.strip().splitlines()[-1]
    report['acceptance'] = 'numerical gates passed; performance is evaluated separately'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    value = audit(args.folder, args.candidate, args.smoke)
    target = 'smoke-acceptance.json' if args.smoke else 'acceptance.json'
    (args.folder/target).write_text(json.dumps(value, indent=2))
    print(json.dumps(value, indent=2))
