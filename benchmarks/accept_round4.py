"""Independent array/source/configuration acceptance for one round-four run."""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import numpy as np

BASELINE = '5060bd524fa5e15ad0408fc80ddb3bc3d5cc6130'


def audit(folder, candidate_sha, comparison, sizes, threads, xcs, repeat):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text())
    expected = {f'{xc}-n{n}-t{t}' for xc,n,t in itertools.product(xcs,sizes,threads)}
    assert not summary['errors'], summary['errors']
    cases = summary['cases']
    assert len(cases) == len(expected)
    assert {c['case'] for c in cases} == expected
    maximum = dict(energy=0., ne=0., exc=0., vxc=0., vj=0., vk=0.)
    rows = []
    record_count = 0
    for case in cases:
        trials = case['trials']
        assert len(trials) == 2*(repeat+1)
        ref = next(r for r in trials if r['variant']=='baseline' and not r['profiled'])
        with np.load(folder/Path(ref['arrays']).name) as z:
            reference = {k:z[k].copy() for k in z.files}
        for variant in ('baseline','candidate'):
            group = [r for r in trials if r['variant']==variant]
            assert sum(not r['profiled'] for r in group) == repeat
            assert sum(r['profiled'] for r in group) == 1
            for r in group:
                record_count += 1
                assert r['source_sha'] == (BASELINE if variant=='baseline' else candidate_sha)
                assert r['converged'] and r['verification'] and r['comparison']==comparison
                assert f"{r['xc']}-n{r['n_water']}-t{r['threads']}" == case['case']
                assert r['grid_level']==6 and r['basis']=='def2-TZVPP'
                assert r['grid_sha256']==ref['grid_sha256']
                assert r['dense_blas']=='1' and r['numa']=='off'
                assert (r['tile_screen'], r['mgga_simd'], r['jk_bound']) == ('0','0','0')
                opt = r['build_options']
                native = ((comparison=='native' and variant=='candidate') or comparison=='qcint'
                          or (comparison=='combined' and variant=='candidate'))
                qcint = comparison in ('qcint','combined') and variant=='candidate'
                assert opt['BUILD_MARCH_NATIVE'] == ('ON' if native else 'OFF'), opt
                assert opt['USE_QCINT'] == ('ON' if qcint else 'OFF'), opt
                assert len(r['native_hashes'])==4
                assert r['threads'] <= len(r['affinity'])
                assert np.isfinite(r['scf_seconds']) and r['scf_seconds'] > 0
                delta = abs(r['energy_hartree']-ref['energy_hartree'])
                assert np.isfinite(delta) and delta <= 1e-8
                maximum['energy'] = max(maximum['energy'],delta)
                with np.load(folder/Path(r['arrays']).name) as arr:
                    np.testing.assert_array_equal(arr['fixed_dm'],reference['fixed_dm'])
                    for key in ('dm','fixed_dm','ne','exc','vxc','vj','vk'):
                        assert np.isfinite(arr[key]).all()
                    for key in ('ne','exc','vxc','vj','vk'):
                        np.testing.assert_allclose(arr[key], reference[key],atol=1e-9,rtol=1e-10)
                        maximum[key] = max(maximum[key],float(np.max(np.abs(arr[key]-reference[key]))))
                enabled = variant=='candidate' and comparison in ('cache','combined')
                stats = r['ao_cache']
                if enabled:
                    assert stats is not None and stats['calls'] > 0
                    assert stats['retained_bytes'] <= stats['peak_bytes'] <= stats['max_bytes']
                    if r['n_water']==1:
                        assert stats['hits']>0, 'cache was never exercised'
                else:
                    assert stats is None
        medians = {v:statistics.median(r['scf_seconds'] for r in trials
                        if r['variant']==v and not r['profiled']) for v in ('baseline','candidate')}
        for v in medians:
            assert abs(medians[v]-case['median_s'][v]) < 1e-12
        speedup = medians['baseline']/medians['candidate']
        assert abs(speedup-case['speedup']) < 1e-12
        rows.append(dict(case=case['case'],median_s=medians,speedup=speedup))
    return dict(passed=True, comparison=comparison, cases=len(cases), records=record_count,
                max_abs=maximum, results=rows,
                numerical_gates=dict(energy_abs=1e-8, matrix_atol=1e-9,matrix_rtol=1e-10))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder')
    p.add_argument('--candidate',required=True)
    p.add_argument('--comparison',required=True)
    p.add_argument('--sizes',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--threads',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--xc',nargs='+',default=['PBE','MN15'])
    p.add_argument('--repeat',type=int,default=3)
    a=p.parse_args()
    result=audit(a.folder,a.candidate,a.comparison,a.sizes,a.threads,a.xc,a.repeat)
    (Path(a.folder)/'independent-acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
