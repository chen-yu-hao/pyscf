"""Reopen every timed record and array; preserve the established numeric gates."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

BASE = 'b5b4f1c711b63c1dbb4fb409e979f53852601eca'

def check(root, candidate, groups):
    total = 0
    maximum = {k: 0. for k in ('energy', 'ne', 'exc', 'vxc', 'vj', 'vk')}
    cases = []
    for group, count, repetitions in groups:
        directory = root/group
        summary = json.loads((directory/'summary.json').read_text())
        assert not summary.get('errors')
        assert len(summary['cases']) == count
        for case in summary['cases']:
            trials = case['trials']
            assert len(trials) == 2*(repetitions+1)
            records = []
            for trial in trials:
                name = Path(trial['arrays']).with_suffix('.json').name
                record = json.loads((directory/name).read_text())
                for key in ('source_sha','scf_seconds','converged','grid_sha256','profiled','variant'):
                    assert record[key] == trial[key], key
                assert record['converged'] and trial.get('verification')
                assert record['source_sha'] == (BASE if record['variant']=='baseline' else candidate)
                assert record['basis']=='def2-TZVPP' and record['grid_level']==6
                assert record['dense_blas']==record['mgga_pack_simd']=='1'
                assert record['tile_screen']==record['jk_block_bound']==record['coalesce']=='0'
                assert record['numa']=='off' and record['timed_diagnostic_counters'] is False
                assert record['eri_cache_enabled'] == (record['variant']=='candidate')
                if not record['profiled']:
                    assert not record['stages_seconds'] and not record['c_dispatch_counts']
                for pool in record['threadpools']:
                    if pool['user_api']=='blas': assert pool['num_threads']==1
                    if pool['user_api']=='openmp': assert pool['num_threads']==record['threads']
                if record['variant']=='candidate':
                    for cache in record['eri_cache_scf']['caches']:
                        assert cache['calls']==cache['hits']==cache['misses']==0
                        assert cache['data_bytes'] <= cache['allocated_bytes'] <= cache['max_bytes']
                    if record['n_water'] >= 4:
                        assert record['used_incore_eri'] is False
                        assert record['eri_cache_scf']['caches']
                        assert sum(x['hits'] for x in record['eri_cache_diagnostic']['caches']) > 0
                records.append(record)
            ref = next(r for r in records if r['variant']=='baseline' and not r['profiled'])
            with np.load(directory/Path(ref['arrays']).name, allow_pickle=False) as npz:
                arrays = {k:npz[k] for k in npz.files}
            for record in records:
                assert record['grid_sha256']==ref['grid_sha256']
                assert record['used_incore_eri']==ref['used_incore_eri']
                err=abs(record['energy_hartree']-ref['energy_hartree'])
                assert np.isfinite(err) and err <= 1e-8
                maximum['energy']=max(maximum['energy'],err)
                with np.load(directory/Path(record['arrays']).name, allow_pickle=False) as npz:
                    for key in npz.files: assert np.isfinite(npz[key]).all(), key
                    np.testing.assert_array_equal(npz['fixed_dm'],arrays['fixed_dm'])
                    for key in ('ne','exc','vxc','vj','vk'):
                        np.testing.assert_allclose(npz[key],arrays[key],atol=1e-9,rtol=1e-10)
                        maximum[key]=max(maximum[key],float(np.max(np.abs(npz[key]-arrays[key]))))
                    for key in ('vj','vk'):
                        np.testing.assert_allclose(npz[key],npz[key+'_uncached'],atol=1e-9,rtol=1e-10)
                total+=1
            samples={v:[r['scf_seconds'] for r in records if r['variant']==v and not r['profiled']]
                     for v in ('baseline','candidate')}
            for v in samples:
                assert len(samples[v])==repetitions
                assert sum(r['profiled'] for r in records if r['variant']==v)==1
            cases.append(dict(case=case['case'],samples=samples,
                              speedup=statistics.median(samples['baseline'])/statistics.median(samples['candidate'])))
    return dict(records=total,cases=cases,max_abs=maximum,errors=[],
                gates=dict(energy_atol=1e-8,array_atol=1e-9,array_rtol=1e-10))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--candidate',required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    groups=[('smoke',1,1)] if a.smoke else [('full',6,3),('dense-control',1,3),('gga-control',1,3)]
    result=check(a.root,a.candidate,groups)
    path=a.root/('smoke-acceptance.json' if a.smoke else 'independent-acceptance.json')
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
