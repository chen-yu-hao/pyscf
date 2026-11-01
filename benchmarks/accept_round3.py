"""Audit every ordinary/profile SCF and all fixed-density XC/J/K arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def audit(root, candidate):
    expected = {
        'full': {f'{xc}-n{n}-t{t}' for xc in ('PBE', 'MN15') for n in (1, 2, 4) for t in (1, 2, 4)},
        'mgga': {f'MN15-n{n}-t2' for n in (1, 2, 4)},
        'jk': {f'MN15-n{n}-t2' for n in (4, 8)},
    }
    base = '5060bd524fa5e15ad0408fc80ddb3bc3d5cc6130'
    report = dict(cases=[], records=0, max_energy_difference=0., max_abs={})
    for group, keys in expected.items():
        folder = root/group
        data = json.loads((folder/'summary.json').read_text())
        assert not data['errors'], data['errors']
        assert len(data['cases']) == len(keys)
        assert {c['case'] for c in data['cases']} == keys
        component = 'both' if group == 'full' else group
        for case in data['cases']:
            trials = case['trials']
            assert len(trials) == 8
            ref = trials[0]
            assert ref['variant'] == 'baseline'
            arrays = np.load(folder/Path(ref['arrays']).name)
            jk_ref = np.load(folder/Path(ref['jk_arrays']).name)
            for row in trials:
                variant = row['variant']
                assert variant in ('baseline', 'candidate')
                assert row['converged'] and 'verification' in row
                assert row['source_sha'] == (base if variant == 'baseline' else candidate)
                assert row['component'] == component
                assert row['grid_level'] == 6 and row['numa'] == 'off'
                assert row['dense_blas'] == '1' and row['tile_screen'] == '0'
                assert row['mgga_pack_simd'] == str(int(variant == 'candidate' and component in ('mgga', 'both')))
                assert row['jk_block_bound'] == str(int(variant == 'candidate' and component in ('jk', 'both')))
                assert case['case'] == f"{row['xc']}-n{row['n_water']}-t{row['threads']}"
                assert row['grid_sha256'] == ref['grid_sha256']
                error = abs(row['energy_hartree']-ref['energy_hartree'])
                assert np.isfinite(error) and error <= 1e-8
                report['max_energy_difference'] = max(report['max_energy_difference'], error)
                with np.load(folder/Path(row['arrays']).name) as value:
                    for key in ('ne', 'exc', 'vxc'):
                        assert np.isfinite(value[key]).all()
                        np.testing.assert_allclose(value[key], arrays[key], atol=1e-9, rtol=1e-10)
                        report['max_abs'][key] = max(report['max_abs'].get(key, 0), float(np.max(np.abs(value[key]-arrays[key]))))
                with np.load(folder/Path(row['jk_arrays']).name) as value:
                    np.testing.assert_array_equal(value['fixed_dm'], jk_ref['fixed_dm'])
                    for key in ('vj', 'vk'):
                        assert np.isfinite(value[key]).all()
                        np.testing.assert_allclose(value[key], jk_ref[key], atol=1e-9, rtol=1e-10)
                        report['max_abs'][key] = max(report['max_abs'].get(key, 0), float(np.max(np.abs(value[key]-jk_ref[key]))))
                if variant == 'candidate' and row['profiled']:
                    assert sum(row['c_dispatch_counts'].values()) > 0
                    if row['n_water'] >= 4 and component in ('jk', 'both'):
                        assert not row['used_incore_eri']
                        assert row['jk_block_counts']['considered'] > 0
                report['records'] += 1
            medians = {}
            for variant in ('baseline', 'candidate'):
                ordinary = [r for r in trials if r['variant'] == variant and not r['profiled']]
                profiles = [r for r in trials if r['variant'] == variant and r['profiled']]
                assert len(ordinary) == 3 and len(profiles) == 1
                medians[variant] = statistics.median(r['scf_seconds'] for r in ordinary)
                assert abs(medians[variant]-case['median_s'][variant]) < 1e-9
            report['cases'].append(dict(group=group, case=case['case'], median_s=medians,
                                       speedup=medians['baseline']/medians['candidate']))
            arrays.close(); jk_ref.close()
    assert report['records'] == 184 and len(report['cases']) == 23
    for name in ('round3-native.log', 'regression-0.log', 'regression-1.log', 'jk-regression.log'):
        content = (root/name).read_text()
        assert 'passed' in content and 'FAILED' not in content and 'ERRORS' not in content, name
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--candidate', required=True)
    args = p.parse_args()
    result = audit(args.root, args.candidate)
    (args.root/'acceptance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
