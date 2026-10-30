"""Targeted saturation-filter replay, not a full SCF benchmark.

Reuse the original independent J/K worker and the same accepted reference D.
Three modes distinguish parent code, same-code filter off, and filter on.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import eri_cache_lookup_replay as previous

BASELINE = 'c10ab2ce464f60a37e74924e770f4110913369f6'
MODES = ('reference', 'control', 'filtered')


def audit(root):
    import numpy as np
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text())
    assert not summary['errors'] and len(summary['cases']) == 2
    assert {(c['n'], c['budget_mb'], c['threads']) for c in summary['cases']} == {(4,64,2),(8,64,2)}
    checked = 0
    maxima = {'vj': 0., 'vk': 0.}
    for case in summary['cases']:
        assert len(case['records']) == 9
        first = next(r for r in case['records'] if r['mode']=='reference' and r['repeat']==0)
        with np.load(root/Path(first['output']).with_suffix('.npz').name, allow_pickle=False) as f:
            common = {k:f[k].copy() for k in f.files}
        ref_path = root.parent/'reference'/f"MN15-n{case['n']}-t2-0-baseline.npz"
        with np.load(ref_path, allow_pickle=False) as f:
            saved = {k:f[k].copy() for k in f.files}
        for mode in MODES:
            records = [r for r in case['records'] if r['mode']==mode]
            assert len(records)==3 and sorted(r['repeat'] for r in records)==[0,1,2]
            assert statistics.median(r['sequence_seconds'] for r in records)==case['median_sequence_seconds'][mode]
            for r in records:
                path = root/Path(r['output']).name
                assert json.loads(path.read_text())==r
                expected = BASELINE if mode=='reference' else summary['candidate_sha']
                assert r['source_sha']==r['sha']==expected
                assert r['n']==case['n'] and r['budget_mb']==64
                assert r['effective_threads']==r['threads']==2 and r['nao']==59*case['n']
                assert r['basis']=='def2-TZVPP' and not r['used_incore_eri']
                assert r['cache_options']==({'fast_lookup':True} if mode=='reference' else
                       {'fast_lookup':True,'saturation_filter':mode=='filtered'})
                assert r['verification']==dict(arrays_atol=1e-9,arrays_rtol=1e-10,passed=True)
                assert r['flags']==dict(PYSCF_NUMA='off',OPENBLAS_NUM_THREADS='1',PYSCF_DFT_DENSE_BLAS='1',PYSCF_DFT_MGGA_PACK_SIMD='1',PYSCF_JK_BLOCK_BOUND='0',PYSCF_DFT_MGGA_COALESCE='0')
                assert not r['timed_diagnostic_counters']
                assert all(p['num_threads']==1 for p in r['threadpools'] if p['user_api']=='blas')
                for st in r['timed_cache']['caches']:
                    assert st['data_bytes']<=st['allocated_bytes']<=st['max_bytes']<=64*1024**2
                    assert all(st[k]==0 for k in ('calls','hits','misses','cost_bypass'))
                    assert st['lookup']['fast_lookup']==1
                    assert all(st['lookup'][k]==0 for k in ('probes','empty_stops','reserve_failures','tombstone_reuses'))
                    if mode!='reference':
                        filt=st['saturation_filter']
                        assert all(filt[k]==0 for k in ('queries','negatives','recompute_bypasses'))
                        assert (filt['bytes']>0)==(mode=='filtered')
                assert sum(st['hits'] for st in r['diagnostics']['caches'])>0
                # Queries or bypasses may be zero: report, never fake an activation.
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:
                    data={k:f[k].copy() for k in f.files}
                assert all(np.isfinite(a).all() for a in data.values())
                assert data['fixed_dm'].tobytes()==common['fixed_dm'].tobytes()==saved['fixed_dm'].tobytes()
                assert hashlib.sha256(data['fixed_dm'].tobytes()).hexdigest()==r['fixed_dm_sha256']
                assert hashlib.sha256(ref_path.read_bytes()).hexdigest()==r['reference_npz_sha256']
                np.testing.assert_array_equal(data['factors'],[1.,.1,.01])
                for k in ('vj','vk'):
                    for ref in (common[k],data[k+'_uncached']):
                        np.testing.assert_allclose(data[k],ref,atol=1e-9,rtol=1e-10)
                        maxima[k]=max(maxima[k],float(np.max(np.abs(data[k]-ref))))
                    np.testing.assert_allclose(data[k][0],saved[k],atol=1e-9,rtol=1e-10)
                checked+=1
        for key in ('libdft.so','libcgto.so'):
            assert len({r['native_hashes'][key] for r in case['records']})==1
        assert len({r['direct_scf_tol'] for r in case['records']})==1
    assert checked==18
    result=dict(passed=True,cases=2,records=checked,max_abs=maxima,
                scope='J/K-only D,0.1D,0.01D replay. Full SCF energy/XC gates still required.')
    (root/'independent-acceptance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    for flag in ('worker','audit','baseline-root','candidate-root','candidate-sha','reference-dir','output'):
        p.add_argument('--'+flag)
    args=p.parse_args()
    if args.worker:
        cfg=json.loads(Path(args.worker).read_text())
        # Adapt only the one-time context factory; no per-quartet Python hook.
        sys.path.insert(0,cfg['root'])
        from pyscf.scf import eri_cache
        original_context=eri_cache.cached_eri
        def configured(*a,**kw):
            kw.update(cfg['cache_options'])
            return original_context(*a,**kw)
        eri_cache.cached_eri=configured
        try:
            return previous.worker(cfg)
        finally:
            eri_cache.cached_eri=original_context
    if args.audit:return audit(args.audit)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    summary=dict(baseline_sha=BASELINE,candidate_sha=args.candidate_sha,cases=[],errors=[])
    env=dict(os.environ,PYSCF_NUMA='off',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',PYSCF_DFT_DENSE_BLAS='1',PYSCF_DFT_MGGA_PACK_SIMD='1',PYSCF_JK_BLOCK_BOUND='0',PYSCF_JK_BLOCK_BOUND_STATS='0',PYSCF_DFT_MGGA_COALESCE='0',PYSCF_DFT_MGGA_COALESCE_STATS='0',PYSCF_DFT_VMAT_TILE_SCREEN='0')
    for n in (4,8):
        case=dict(n=n,budget_mb=64,threads=2,records=[]);summary['cases'].append(case)
        for repeat in range(3):
            order=MODES[repeat:]+MODES[:repeat]
            for mode in order:
                root=Path(args.baseline_root if mode=='reference' else args.candidate_root).resolve()
                opts=dict(fast_lookup=True)
                if mode!='reference':opts['saturation_filter']=mode=='filtered'
                cfg=dict(root=str(root),sha=BASELINE if mode=='reference' else args.candidate_sha,n=n,threads=2,budget_mb=64,mode=mode,repeat=repeat,cache_options=opts,
                         reference=str((Path(args.reference_dir)/f'MN15-n{n}-t2-0-baseline.npz').resolve()),
                         output=str(out/f'n{n}-{repeat}-{mode}.json'))
                config=Path(cfg['output']).with_suffix('.config.json');config.write_text(json.dumps(cfg))
                try:
                    with config.with_suffix('.log').open('w') as log:
                        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],cwd=root,env={**env,'PYTHONPATH':str(root)},stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
                    case['records'].append(json.loads(Path(cfg['output']).read_text()))
                except Exception as exc:
                    summary['errors'].append(f'{config.name}: {exc}')
                    (out/'summary.json').write_text(json.dumps(summary,indent=2));raise
        case['median_sequence_seconds']={m:statistics.median(r['sequence_seconds'] for r in case['records'] if r['mode']==m) for m in MODES}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({k:v for k,v in case.items() if k!='records'}),flush=True)
    audit(out)


if __name__=='__main__':main()
