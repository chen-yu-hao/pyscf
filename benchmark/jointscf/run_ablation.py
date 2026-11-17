#!/usr/bin/env python
'''
Ablation experiments for JointSCF schedule='transferable':
  1. Mini-batch size (molecule_batch in {1, 4, 8, 'full'})
  2. Correction rank (correction_rank in {0, 1, 2, None})
  3. Shared optimizer (optimizer in {'lbfgs', 'adam'})

Runs each configuration in a fresh single-thread process via run_benchmark.run_one
and writes benchmark/jointscf/ablation_results.json.
'''

import os
import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_benchmark


def _summarize(run, ref_e):
    e = np.asarray(run['e_tot'], dtype=float)
    de = e - ref_e
    abs_de = np.abs(de)
    return {
        'nconf': int(run['nconf']),
        'wall': float(run['wall']),
        'nfock_mean': float(np.mean(run['nfock'])),
        'nfock_shared_mean': float(np.mean(run['nfock_shared'])),
        'nfock_polish_mean': float(np.mean(run['nfock_polish'])),
        'converged': int(sum(run['converged'])),
        'mean_abs_de': float(abs_de.mean()),
        'median_abs_de': float(np.median(abs_de)),
        'p90_abs_de': float(np.percentile(abs_de, 90)),
        'max_abs_de': float(abs_de.max()),
        'min_signed_de': float(de.min()),
        'e_err_est_mean': float(np.mean(run['e_err_est'])),
        'maxrss_mb': run.get('maxrss_mb'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref-json', default=os.path.join(HERE, 'results.json'))
    parser.add_argument('--out', default=os.path.join(HERE, 'ablation_results.json'))
    args = parser.parse_args()

    refs = {}
    if os.path.exists(args.ref_json):
        with open(args.ref_json) as f:
            rdata = json.load(f)
        for key, run in rdata.get('runs', {}).items():
            case, method, _ = key.split('/')
            if method == 'reference':
                refs[case] = np.asarray(run['e_tot'], dtype=float)

    needed_cases = ('h2o-64', 'h2o-scan2d-64', 'h2o-stretched-64',
                    'c2h4-path-48', 'c2h4-48', 'h2co-40')
    for c in needed_cases:
        if c not in refs:
            r = run_benchmark.run_one(
                c, 'reference', dict(kind='pool', omp=1, nproc=4, conv_tol=1e-12))
            refs[c] = np.asarray(r['e_tot'], dtype=float)

    data = {'environment': run_benchmark.environment_info(),
            'batch_size': [], 'correction_rank': [], 'optimizer': []}
    if os.path.exists(args.out):
        with open(args.out) as f:
            data = json.load(f)

    def _done(section, tag):
        return any(item['tag'] == tag for item in data[section])

    def _save():
        with open(args.out + '.tmp', 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(args.out + '.tmp', args.out)

    # 1. Mini-batch size ablation
    for case in ('h2o-64', 'h2o-scan2d-64', 'c2h4-path-48', 'c2h4-48'):
        for polish in (False, True):
            for mb in (1, 4, 8, 'full'):
                tag = f'{case}/polish={polish}/mb={mb}'
                if _done('batch_size', tag):
                    continue
                spec = dict(kind='joint', schedule='transferable', polish=polish,
                            molecule_batch=mb, omp=1, workers=1)
                t0 = time.time()
                run = run_benchmark.run_one(case, tag, spec)
                rec = {'tag': tag, 'case': case, 'polish': polish, 'molecule_batch': mb}
                rec.update(_summarize(run, refs[case]))
                data['batch_size'].append(rec)
                _save()
                print(f'[batch_size] {tag:36s} wall={rec["wall"]:6.2f}s '
                      f'nfock={rec["nfock_mean"]:5.2f} (sh={rec["nfock_shared_mean"]:4.2f}+'
                      f'po={rec["nfock_polish_mean"]:4.2f}) max|dE|={rec["max_abs_de"]:.2e} '
                      f'({time.time()-t0:.0f}s)', flush=True)

    # 2. Correction rank r_Delta ablation
    for case in ('h2o-64', 'h2o-stretched-64', 'c2h4-48', 'h2co-40'):
        for polish in (False, True):
            for r_delta in (0, 1, 2, None):
                tag = f'{case}/polish={polish}/r_delta={r_delta}'
                if _done('correction_rank', tag):
                    continue
                spec = dict(kind='joint', schedule='transferable', polish=polish,
                            correction_rank=r_delta, omp=1, workers=1)
                t0 = time.time()
                run = run_benchmark.run_one(case, tag, spec)
                rec = {'tag': tag, 'case': case, 'polish': polish, 'correction_rank': r_delta}
                rec.update(_summarize(run, refs[case]))
                data['correction_rank'].append(rec)
                _save()
                print(f'[r_delta]    {tag:36s} wall={rec["wall"]:6.2f}s '
                      f'nfock={rec["nfock_mean"]:5.2f} (sh={rec["nfock_shared_mean"]:4.2f}+'
                      f'po={rec["nfock_polish_mean"]:4.2f}) mean|dE|={rec["mean_abs_de"]:.2e} '
                      f'max|dE|={rec["max_abs_de"]:.2e} ({time.time()-t0:.0f}s)', flush=True)

    # 3. Optimizer ablation
    for case in ('h2o-64', 'h2o-stretched-64', 'c2h4-48'):
        for polish in (False, True):
            for opt in ('lbfgs', 'adam'):
                tag = f'{case}/polish={polish}/opt={opt}'
                if _done('optimizer', tag):
                    continue
                spec = dict(kind='joint', schedule='transferable', polish=polish,
                            optimizer=opt, omp=1, workers=1)
                t0 = time.time()
                run = run_benchmark.run_one(case, tag, spec)
                rec = {'tag': tag, 'case': case, 'polish': polish, 'optimizer': opt}
                rec.update(_summarize(run, refs[case]))
                data['optimizer'].append(rec)
                _save()
                print(f'[optimizer]  {tag:36s} wall={rec["wall"]:6.2f}s '
                      f'nfock={rec["nfock_mean"]:5.2f} (sh={rec["nfock_shared_mean"]:4.2f}+'
                      f'po={rec["nfock_polish_mean"]:4.2f}) mean|dE|={rec["mean_abs_de"]:.2e} '
                      f'max|dE|={rec["max_abs_de"]:.2e} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
