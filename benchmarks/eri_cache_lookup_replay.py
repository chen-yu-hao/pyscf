"""Fresh-process J/K density-sequence ablation; NOT full-SCF performance.

The accepted round-five artifact supplies D on the same water geometry. Each
sample builds J/K for D, 0.1*D and 0.01*D, including the cold cache allocation.
No SCF kernel is invoked, no convergence or new energy result is claimed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

BASELINE = 'bb8e98d04fbfa790e943f9c5b0c2e0702b44c872'
FACTORS = [1., .1, .01]


def water_cluster(n):
    atoms=[]
    for i in range(n):
        x,y=4.*(i%3),4.*(i//3)
        atoms.extend([('O',(x,y,0.)),('H',(x+.757,y,.587)),('H',(x-.757,y,.587))])
    return atoms


def worker(cfg):
    import numpy as np
    sys.path.insert(0,cfg['root'])
    import pyscf
    from pyscf import gto,scf,lib
    from pyscf.scf.eri_cache import cached_eri
    from threadpoolctl import threadpool_info,threadpool_limits
    root=Path(cfg['root']).resolve()
    assert Path(pyscf.__file__).resolve().is_relative_to(root)
    sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert sha==cfg['sha']
    lib.num_threads(cfg['threads'])
    with np.load(cfg['reference'],allow_pickle=False) as saved:
        dm=saved['fixed_dm'].copy(); saved_j=saved['vj'].copy(); saved_k=saved['vk'].copy()
    mol=gto.M(atom=water_cluster(cfg['n']),basis='def2-TZVPP',unit='Angstrom',verbose=0,max_memory=2300)
    assert dm.shape==(mol.nao,mol.nao) and mol.nao==59*cfg['n']
    mf=scf.RHF(mol); mf.max_memory=2300
    mf._is_mem_enough=lambda:False  # this experiment explicitly measures direct J/K
    kwargs={'max_mb':cfg['budget_mb']}
    if cfg['mode']!='legacy': kwargs['fast_lookup']=cfg['mode']=='fast'
    outputs=[]; times=[]
    with threadpool_limits(limits=1,user_api='blas'):
        start=time.perf_counter()
        with cached_eri(mf,**kwargs) as cache:
            for factor in FACTORS:
                t=time.perf_counter(); outputs.append(mf.get_jk(dm=factor*dm,hermi=1))
                times.append(time.perf_counter()-t)
            elapsed=time.perf_counter()-start
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            timed_cache=cache.stats()
            # All performance timers have stopped before any diagnostic counters.
            cache.stats(diagnostics=True,reset=True)
            diagnostic_jk=mf.get_jk(dm=dm,hermi=1)
            diagnostics=cache.stats(diagnostics=False)
        pools=threadpool_info()
        assert all(p['num_threads']==1 for p in pools if p['user_api']=='blas')
        independent=scf.RHF(mol); independent.max_memory=2300
        independent._is_mem_enough=lambda:False
        references=[independent.get_jk(dm=factor*dm,hermi=1) for factor in FACTORS]
    arrays={'fixed_dm':dm,'factors':np.asarray(FACTORS),
            'vj':np.asarray([x[0] for x in outputs]),'vk':np.asarray([x[1] for x in outputs]),
            'vj_uncached':np.asarray([x[0] for x in references]),'vk_uncached':np.asarray([x[1] for x in references])}
    for value in arrays.values():assert np.isfinite(value).all()
    for name in ('vj','vk'):
        np.testing.assert_allclose(arrays[name],arrays[name+'_uncached'],atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(arrays['vj'][0],saved_j,atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(arrays['vk'][0],saved_k,atol=1e-9,rtol=1e-10)
    for actual,reference in zip(diagnostic_jk,references[0]):
        np.testing.assert_allclose(actual,reference,atol=1e-9,rtol=1e-10)
    assert mf._eri is None
    assert timed_cache['caches'] and diagnostics['caches']
    for record in timed_cache['caches']:
        assert record['enabled'] and record['allocated_bytes']<=record['max_bytes']
        assert record['data_bytes']<=record['allocated_bytes']
        assert all(record[k]==0 for k in ('calls','hits','misses','cost_bypass'))
        if 'lookup' in record:
            assert all(record['lookup'][k]==0 for k in ('probes','empty_stops','reserve_failures','tombstone_reuses'))
    assert sum(x['hits'] for x in diagnostics['caches'])>0
    if cfg['mode']=='fast':assert diagnostics['caches'][0]['lookup']['fast_lookup']==1
    dest=Path(cfg['output']); np.savez(dest.with_suffix('.npz'),**arrays)
    result={**cfg,'source_sha':sha,'basis':'def2-TZVPP','nao':mol.nao,
            'effective_threads':lib.num_threads(),'direct_scf_tol':mf.direct_scf_tol,
            'sequence_seconds':elapsed,'jk_call_seconds':times,
            'max_rss_kib_at_sequence_end':rss,'timed_cache':timed_cache,'diagnostics':diagnostics,
            'timed_diagnostic_counters':False,'used_incore_eri':mf._eri is not None,
            'reference_npz_sha256':hashlib.sha256(Path(cfg['reference']).read_bytes()).hexdigest(),
            'fixed_dm_sha256':hashlib.sha256(dm.tobytes()).hexdigest(),
            'threadpools':pools,'affinity':sorted(os.sched_getaffinity(0)),
            'flags':{k:os.environ.get(k) for k in ('PYSCF_NUMA','OPENBLAS_NUM_THREADS','PYSCF_DFT_DENSE_BLAS','PYSCF_DFT_MGGA_PACK_SIMD','PYSCF_JK_BLOCK_BOUND','PYSCF_DFT_MGGA_COALESCE')},
            'native_hashes':{n:hashlib.sha256((root/'pyscf/lib'/n).read_bytes()).hexdigest() for n in ('libdft.so','libcvhf.so','libcgto.so')},
            'verification':{'arrays_atol':1e-9,'arrays_rtol':1e-10,'passed':True},
            'scope':'J/K-only density sequence: no SCF kernel, new energy, or convergence claim'}
    dest.write_text(json.dumps(result,indent=2))


def audit(root):
    import numpy as np
    root=Path(root); summary=json.loads((root/'summary.json').read_text())
    assert not summary['errors'] and len(summary['cases'])==4
    checked=0; maxima={'vj':0.,'vk':0.}
    for case in summary['cases']:
        assert len(case['records'])==9
        ref=None
        for mode in ('legacy','control','fast'):
            records=[r for r in case['records'] if r['mode']==mode]
            assert len(records)==3 and sorted(r['repeat'] for r in records)==[0,1,2]
            med=statistics.median(r['sequence_seconds'] for r in records)
            assert med==case['median_sequence_seconds'][mode]
            for r in records:
                path=root/Path(r['output']).name
                assert json.loads(path.read_text())==r
                assert r['source_sha']==(BASELINE if mode=='legacy' else summary['candidate_sha'])
                assert r['effective_threads']==case['threads']==2
                assert r['flags']==dict(PYSCF_NUMA='off',OPENBLAS_NUM_THREADS='1',PYSCF_DFT_DENSE_BLAS='1',PYSCF_DFT_MGGA_PACK_SIMD='1',PYSCF_JK_BLOCK_BOUND='0',PYSCF_DFT_MGGA_COALESCE='0')
                assert all(p['num_threads']==1 for p in r['threadpools'] if p['user_api']=='blas')
                assert not r['timed_diagnostic_counters'] and not r['used_incore_eri']
                assert r['n']==case['n'] and r['budget_mb']==case['budget_mb'] and r['nao']==59*case['n']
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:
                    data={k:f[k].copy() for k in f.files}
                assert all(np.isfinite(x).all() for x in data.values())
                assert data['fixed_dm'].shape==(59*case['n'],)*2
                assert hashlib.sha256(data['fixed_dm'].tobytes()).hexdigest()==r['fixed_dm_sha256']
                np.testing.assert_array_equal(data['factors'],FACTORS)
                if ref is None:ref=data
                assert data['fixed_dm'].tobytes()==ref['fixed_dm'].tobytes()
                for key in ('vj','vk'):
                    for target in (ref[key],data[key+'_uncached']):
                        np.testing.assert_allclose(data[key],target,atol=1e-9,rtol=1e-10)
                        maxima[key]=max(maxima[key],float(np.max(np.abs(data[key]-target))))
                checked+=1
        assert len({r['direct_scf_tol'] for r in case['records']})==1
        assert len({r['reference_npz_sha256'] for r in case['records']})==1
        # Only cvhf should differ. Same integral backend, DFT and AO kernels.
        for key in ('libdft.so','libcgto.so'):
            assert len({r['native_hashes'][key] for r in case['records']})==1
    assert checked==36
    result={'cases':4,'records':checked,'max_abs':maxima,'passed':True,
            'scope':'fixed-density J/K only; full SCF energy<=1e-8 Eh gate remains required before release'}
    (root/'independent-acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--audit')
    parser.add_argument('--baseline-root');parser.add_argument('--candidate-root');parser.add_argument('--candidate-sha')
    parser.add_argument('--reference-dir');parser.add_argument('--output')
    args=parser.parse_args()
    if args.worker:return worker(json.loads(Path(args.worker).read_text()))
    if args.audit:return audit(args.audit)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    summary={'baseline_sha':BASELINE,'candidate_sha':args.candidate_sha,'cases':[],'errors':[]}
    env=dict(os.environ,PYSCF_NUMA='off',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE',OMP_MAX_ACTIVE_LEVELS='1',PYSCF_DFT_DENSE_BLAS='1',PYSCF_DFT_MGGA_PACK_SIMD='1',PYSCF_JK_BLOCK_BOUND='0',PYSCF_JK_BLOCK_BOUND_STATS='0',PYSCF_DFT_MGGA_COALESCE='0',PYSCF_DFT_VMAT_TILE_SCREEN='0')
    for n in (4,8):
        for budget in (64,256):
            case={'n':n,'threads':2,'budget_mb':budget,'records':[]}
            summary['cases'].append(case)
            for repeat in range(3):
                modes=['legacy','control','fast'];modes=modes[repeat:]+modes[:repeat]
                for mode in modes:
                    root=args.baseline_root if mode=='legacy' else args.candidate_root
                    cfg=dict(root=str(Path(root).resolve()),sha=BASELINE if mode=='legacy' else args.candidate_sha,
                             n=n,threads=2,budget_mb=budget,mode=mode,repeat=repeat,
                             reference=str((Path(args.reference_dir)/f'MN15-n{n}-t2-0-baseline.npz').resolve()),
                             output=str(output/f'n{n}-m{budget}-{repeat}-{mode}.json'))
                    config=Path(cfg['output']).with_suffix('.config.json');config.write_text(json.dumps(cfg))
                    try:
                        with config.with_suffix('.log').open('w') as log:
                            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],cwd=root,env={**env,'PYTHONPATH':root},stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
                        case['records'].append(json.loads(Path(cfg['output']).read_text()))
                    except Exception as exc:
                        summary['errors'].append(f'{config.name}: {exc}')
                        (output/'summary.json').write_text(json.dumps(summary,indent=2));raise
            case['median_sequence_seconds']={mode:statistics.median(r['sequence_seconds'] for r in case['records'] if r['mode']==mode) for mode in modes}
            print(json.dumps({k:v for k,v in case.items() if k!='records'}),flush=True)
            (output/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(output)


if __name__=='__main__':main()
