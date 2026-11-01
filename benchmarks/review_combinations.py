"""Staged, audited cold-SCF search on fixed PySCF source revisions.

Screen every relevant Boolean kernel combination, then search cache budgets and
thread allocations. Screening samples never count toward finalist repetitions.
Every sample is a fresh process and retains its raw arrays and provenance.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from review_upstream import FLAGS
from review_combo_audit import audit_record


BITS = (
    ('D', 'PYSCF_DFT_DENSE_BLAS'),
    ('S', 'PYSCF_DFT_MGGA_PACK_SIMD'),
    ('T', 'PYSCF_DFT_VMAT_TILE_SCREEN'),
    ('C', 'PYSCF_DFT_MGGA_COALESCE'),
    ('R', 'PYSCF_DFT_MGGA_SCATTER_RUNS'),
    ('J', 'PYSCF_JK_BLOCK_BOUND'),
)
SCHEMA = 1


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hardware():
    affinity = sorted(os.sched_getaffinity(0))
    quotas = []
    for path in (Path('/sys/fs/cgroup/cpu.max'),):
        if path.is_file():
            fields = path.read_text().split()
            if len(fields) == 2 and fields[0] != 'max':
                quotas.append(int(fields[0]) / int(fields[1]))
    quota_file = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period_file = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota_file.is_file() and period_file.is_file():
        quota = int(quota_file.read_text())
        if quota > 0:
            quotas.append(quota / int(period_file.read_text()))
    usable = min([len(affinity)] + [max(1, math.floor(q)) for q in quotas])
    model = next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines()
                  if x.startswith('model name')), 'unknown')
    return dict(affinity=affinity, cpu_quotas=quotas, usable_cpus=usable, cpu_model=model)


def profile(mask=0, ao=0, eri=None, threads=2, blas=1, source='candidate', build='portable'):
    return dict(core_mask=int(mask), ao_cache_mb=ao, cache_options=eri,
                threads=int(threads), blas_threads=int(blas), source_kind=source,
                build_profile=build)


def identity(value):
    keys = ('core_mask', 'ao_cache_mb', 'cache_options', 'threads', 'blas_threads',
            'source_kind', 'build_profile')
    return json.dumps({key: value[key] for key in keys}, sort_keys=True)


def label(value):
    eri = value['cache_options']
    code = 'off' if eri is None else (
        ('filter' if eri['saturation_filter'] else 'fast' if eri['fast_lookup'] else 'legacy')
        + str(eri['max_mb']))
    return (f"{value['source_kind']}-{value['build_profile']}-m{value['core_mask']:02d}"
            f"-ao{value['ao_cache_mb']}-eri{code}-t{value['threads']}b{value['blas_threads']}")


def dedupe(values):
    unique = {}
    for value in values:
        unique.setdefault(identity(value), copy.deepcopy(value))
    return list(unique.values())


def ericache(mb, fast, filtered):
    return dict(max_mb=mb, min_primitive_work=64, fast_lookup=fast,
                saturation_filter=filtered)


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.hardware = hardware()
        self.base_threads = min(2, self.hardware['usable_cpus'])
        self.roots = {}
        for source in [args.roots] + ([args.native_roots] if args.native_roots else []):
            entry = json.loads(Path(source).read_text())
            self.roots[entry['build_profile']] = entry
        self.case = f'{args.xc}-n{args.n}-def2-TZVPP-grid6'
        self.records = []
        self.failures = []
        self.references = {}
        self.serial = 0
        self.seen_screen = {}
        write_json(self.out / 'hardware.json', self.hardware)
        print('COMBO_HARDWARE ' + json.dumps(self.hardware), flush=True)

    def make_config(self, variant, phase, repeat):
        build = variant['build_profile']
        root = self.roots[build][variant['source_kind']]
        flags = dict(FLAGS)
        flags['PYSCF_DFT_VMAT_MAX_MEMORY_MB'] = '256'
        for bit, (_, key) in enumerate(BITS):
            flags[key] = str(int(bool(variant['core_mask'] & (1 << bit))))
        flags.update(OMP_NUM_THREADS=str(variant['threads']))
        for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS'):
            flags[key] = str(variant['blas_threads'])
        name = f'{self.serial:04d}-{phase}-{label(variant)}-r{repeat}'
        reference = self.references.get(build)
        reference_arrays = str(reference.with_suffix('.npz')) if reference else None
        return dict(root=root['root'], sha=root['sha'], output=str(self.out / (name + '.json')),
                    n=self.args.n, xc=self.args.xc, basis='def2-TZVPP', grid_level=6,
                    max_memory=2300, force_direct=False, flags=flags,
                    reference_arrays=reference_arrays, case=self.case,
                    variant=label(variant), phase=phase, repeat=repeat, **variant)

    def run(self, variant, phase, repeat=0, required=False):
        if variant['threads'] * variant['blas_threads'] > self.hardware['usable_cpus']:
            raise ValueError('Requested thread allocation exceeds CPU affinity/quota')
        self.serial += 1
        cfg = self.make_config(variant, phase, repeat)
        config_file = Path(cfg['output']).with_suffix('.config.json')
        write_json(config_file, cfg)
        env = dict(os.environ, **cfg['flags'])
        worker = Path(__file__).with_name('review_combo_worker.py')
        started = time.monotonic()
        try:
            process = subprocess.run([sys.executable, str(worker), '--config', str(config_file)],
                                     env=env, text=True, capture_output=True, timeout=2400)
            Path(cfg['output']).with_suffix('.log').write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f'Worker exit {process.returncode}: ' + process.stderr[-3000:])
            raw_path = Path(cfg['output'])
            raw = json.loads(raw_path.read_text())
            reference = self.references.get(variant['build_profile'], raw_path)
            audited = audit_record(str(raw_path), str(reference), expected_cfg=cfg)
            if variant['build_profile'] not in self.references:
                if variant['source_kind'] != 'upstream':
                    raise ValueError('The first record in a build profile must be upstream')
                self.references[variant['build_profile']] = raw_path
                if variant['build_profile'] != 'portable' and 'portable' in self.references:
                    portable = json.loads(self.references['portable'].read_text())
                    if abs(raw['energy_hartree'] - portable['energy_hartree']) > 1e-8:
                        raise AssertionError('Compiler-profile energy mismatch')
                    if raw['grid_sha256'] != portable['grid_sha256']:
                        raise AssertionError('Compiler-profile grid mismatch')
            item = dict(case=self.case, phase=phase, repeat=repeat, variant=copy.deepcopy(variant),
                        label=label(variant), json=raw_path.name,
                        seconds=raw['scf_seconds'], energy=raw['energy_hartree'],
                        rss_mib=raw['max_rss_kib_at_scf_end'] / 1024,
                        cycles=raw['cycles'], used_incore_eri=raw['used_incore_eri'],
                        scope_status=raw['scope_status'], audit=audited,
                        wall_seconds=time.monotonic() - started)
            self.records.append(item)
            if phase not in ('final', 'confirm', 'build-final'):
                self.seen_screen[identity(variant)] = item
            print('COMBO_RECORD ' + json.dumps(item, allow_nan=False), flush=True)
            write_json(self.out / 'records.json', self.records)
            return item
        except Exception as exc:
            failed = dict(case=self.case, phase=phase, variant=variant, config=cfg,
                          error=str(exc), traceback=traceback.format_exc())
            self.failures.append(failed)
            write_json(self.out / 'failures.json', self.failures)
            print('COMBO_FAILURE ' + json.dumps(failed), flush=True)
            if required or not self.references:
                raise
            return None

    def probe(self, variant, phase):
        return self.seen_screen.get(identity(variant)) or self.run(variant, phase)

    def baseline(self):
        return profile(threads=self.base_threads, source='upstream')

    def reference(self):
        return profile(mask=1 if self.args.xc == 'PBE' else 3, threads=self.base_threads)

    def candidate_records(self):
        return [r for r in self.seen_screen.values() if r['variant']['source_kind'] == 'candidate']

    def finish(self, name, value):
        value.update(schema=SCHEMA, case=self.case, xc=self.args.xc, n=self.args.n,
                     hardware=self.hardware, roots=self.roots,
                     record_count=len(self.records), failed_count=len(self.failures),
                     failures=self.failures)
        write_json(self.out / name, value)
        print('COMBO_SUMMARY ' + json.dumps(value, allow_nan=False), flush=True)
        return value


def screen(experiment):
    ex = experiment
    ex.run(ex.baseline(), 'reference', required=True)
    reference = ex.reference()
    masks = list(range(64)) if ex.args.xc == 'MN15' else [0, 1, 4, 5, 32, 33, 36, 37]
    masks = [mask for index, mask in enumerate(masks) if index % ex.args.shards == ex.args.shard]
    random.Random(20261101 + ex.args.shard).shuffle(masks)
    before = ex.run(reference, 'anchor-before', required=True)
    measured = []
    for mask in masks:
        result = ex.run(profile(mask=mask, threads=ex.base_threads), 'core-screen')
        if result:
            measured.append(result)
    after = ex.run(reference, 'anchor-after', required=True)
    denominator = statistics.median([before['seconds'], after['seconds']])
    candidates = [dict(variant=r['variant'], seconds=r['seconds'], json=r['json'],
                       normalized_seconds=r['seconds'] / denominator) for r in measured]
    return ex.finish('screen_summary.json', dict(stage='screen', shard=ex.args.shard,
                     shards=ex.args.shards, masks=masks, candidates=candidates,
                     anchor_seconds=[before['seconds'], after['seconds']],
                     anchor_drift_percent=100 * abs(after['seconds'] / before['seconds'] - 1),
                     selection_only=True))


def input_summaries(directory, filename, xc=None):
    values = [json.loads(path.read_text()) for path in Path(directory).rglob(filename)]
    if xc is not None:
        values = [v for v in values if v['xc'] == xc]
    if not values:
        raise ValueError(f'No {filename} for {xc} in {directory}')
    return values


def thread_allocations(cpus):
    candidates = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2)]
    return [(omp, blas) for omp, blas in candidates if omp * blas <= cpus]


def aggregate_final(ex, phase):
    groups = {}
    for item in ex.records:
        if item['phase'] == phase:
            groups.setdefault(identity(item['variant']), []).append(item)
    summaries = []
    for items in groups.values():
        if len(items) != 3:
            continue
        times = [r['seconds'] for r in items]
        memory = [r['rss_mib'] for r in items]
        summaries.append(dict(variant=items[0]['variant'], label=items[0]['label'],
                              median_seconds=statistics.median(times), seconds=times,
                              minimum_seconds=min(times), maximum_seconds=max(times),
                              median_rss_mib=statistics.median(memory), rss_mib=memory,
                              cycles=[r['cycles'] for r in items],
                              scope_status=[r['scope_status'] for r in items],
                              used_incore_eri=[r['used_incore_eri'] for r in items],
                              records=[r['json'] for r in items]))
    summaries.sort(key=lambda item: item['median_seconds'])
    candidates = [r for r in summaries if r['variant']['source_kind'] == 'candidate']
    if not candidates:
        raise AssertionError('No candidate completed all three independent finalist repetitions')
    return dict(finalists=summaries, winner=candidates[0],
                ranked_candidates=candidates,
                uncertainty='Three independent samples; screening samples excluded. '
                'Overlapping ranges or small differences do not establish a stable ranking.')


def measure_finalists(ex, variants, phase):
    variants = dedupe(variants)
    # Required upstream reference is first before any candidate in each profile.
    variants.sort(key=lambda p: (p['build_profile'] != 'portable', p['source_kind'] != 'upstream', label(p)))
    for repeat in range(3):
        ordered = variants[repeat:] + variants[:repeat]
        for variant in ordered:
            ex.run(variant, phase, repeat=repeat, required=variant['source_kind'] == 'upstream')
    result = aggregate_final(ex, phase)
    result['requested_variants'] = variants
    result['complete_variants'] = len(result['finalists'])
    result['expected_variants'] = len(variants)
    return result


def tune(ex):
    screens = input_summaries(ex.args.inputs, 'screen_summary.json', ex.args.xc)
    actual = sorted(mask for part in screens for mask in part['masks'])
    expected = list(range(64)) if ex.args.xc == 'MN15' else [0, 1, 4, 5, 32, 33, 36, 37]
    if actual != expected:
        raise AssertionError('Missing or duplicate core-screen shards')
    ranked = sorted([row for part in screens for row in part['candidates']],
                    key=lambda item: item['normalized_seconds'])
    ex.run(ex.baseline(), 'reference', required=True)
    seeds = dedupe([r['variant'] for r in ranked[:4]] + [ex.reference(),
                   profile(threads=ex.base_threads)])
    for value in seeds:
        value['threads'] = ex.base_threads
        ex.probe(value, 'same-host-core')
    core_rank = sorted(ex.candidate_records(), key=lambda r: r['seconds'])
    if not core_rank:
        raise AssertionError('No valid core candidate')
    top_core = copy.deepcopy(core_rank[0]['variant'])
    runner_core = copy.deepcopy(core_rank[min(1, len(core_rank)-1)]['variant'])
    standalone = []
    for mb in (256, 768, 1024):
        value = copy.deepcopy(top_core); value['ao_cache_mb'] = mb
        item = ex.probe(value, 'ao-screen')
        if item:
            standalone.append(item)
    for options in (ericache(256, False, False), ericache(64, True, False),
                    ericache(256, True, False), ericache(64, True, True),
                    ericache(256, True, True)):
        value = copy.deepcopy(top_core); value['cache_options'] = options
        item = ex.probe(value, 'eri-screen')
        if item:
            standalone.append(item)
    aos = sorted([r for r in standalone if r['variant']['ao_cache_mb']], key=lambda r:r['seconds'])
    eris = sorted([r for r in standalone if r['variant']['cache_options']], key=lambda r:r['seconds'])
    if aos and eris:
        pairs = [(0, 0)] + ([(1, 0)] if len(aos)>1 else []) + ([(0, 1)] if len(eris)>1 else [])
        for ai, ei in pairs:
            value = copy.deepcopy(top_core)
            value['ao_cache_mb'] = aos[ai]['variant']['ao_cache_mb']
            value['cache_options'] = eris[ei]['variant']['cache_options']
            ex.probe(value, 'combined-cache')
    current = sorted(ex.candidate_records(), key=lambda r:r['seconds'])
    for winner in current[:2]:
        value = copy.deepcopy(runner_core)
        value['ao_cache_mb'] = winner['variant']['ao_cache_mb']
        value['cache_options'] = winner['variant']['cache_options']
        ex.probe(value, 'cache-core-interaction')
    current = sorted(ex.candidate_records(), key=lambda r:r['seconds'])
    for selected in current[:2]:
        for omp, blas in thread_allocations(ex.hardware['usable_cpus']):
            value = copy.deepcopy(selected['variant'])
            value['threads'], value['blas_threads'] = omp, blas
            ex.probe(value, 'thread-screen')
    upstream_probes = []
    for omp, blas in thread_allocations(ex.hardware['usable_cpus']):
        value = ex.baseline(); value['threads'], value['blas_threads'] = omp, blas
        item = ex.probe(value, 'upstream-thread-screen')
        if item:
            upstream_probes.append(item)
    candidates = sorted(ex.candidate_records(), key=lambda r:r['seconds'])
    nocache = next(r for r in candidates if not r['variant']['ao_cache_mb'] and r['variant']['cache_options'] is None)
    best_upstream = min(upstream_probes, key=lambda r:r['seconds'])['variant']
    finalists = ([r['variant'] for r in candidates[:2]] + [nocache['variant'], ex.reference(),
                 profile(threads=ex.base_threads), ex.baseline(), best_upstream])
    result = measure_finalists(ex, finalists, 'final')
    result.update(stage='tune', core_screen_count=len(actual),
                  core_screen_shards=len(screens),
                  best_nocache=next(r for r in result['ranked_candidates']
                                   if not r['variant']['ao_cache_mb'] and r['variant']['cache_options'] is None),
                  reference_config=ex.reference(), default_config=profile(threads=ex.base_threads),
                  best_upstream=min([r for r in result['finalists']
                                     if r['variant']['source_kind']=='upstream'],key=lambda r:r['median_seconds']))
    return ex.finish('tune_summary.json', result)


def confirm(ex):
    tunes = input_summaries(ex.args.inputs, 'tune_summary.json')
    matched = next(v for v in tunes if v['xc'] == ex.args.xc)
    if ex.args.n == 4:
        # Retest both functional winners on each functional. This is the
        # evidence for a common recipe, rather than extrapolating across XC.
        selections = [v['winner']['variant'] for v in tunes]
        selections += [matched['best_nocache']['variant'], matched['reference_config']]
    else:
        selections = [matched['winner']['variant'], matched['best_nocache']['variant'],
                      matched['reference_config']]
    variants = [ex.baseline(), matched['best_upstream']['variant'],
                profile(threads=ex.base_threads)] + selections
    if ex.args.n == 1:
        # Dense small-system kernels and thread overhead need their own
        # measurements; do not assume the larger-system winner transfers.
        small_masks = [0, 1] if ex.args.xc == 'PBE' else [0, 1, 2, 3]
        for omp, blas in thread_allocations(ex.hardware['usable_cpus']):
            variants.extend(profile(mask=mask,threads=omp,blas=blas) for mask in small_masks)
            variants.append(profile(threads=omp,blas=blas,source='upstream'))
            for mb in (256, 768):
                variants.append(profile(mask=1 if ex.args.xc == 'PBE' else 3,
                                        ao=mb,threads=omp,blas=blas))
    skipped = [v for v in variants
               if v['threads'] * v['blas_threads'] > ex.hardware['usable_cpus']]
    # A recipe exceeding this runner's capacity is not silently downscaled.
    variants = [v for v in variants if v['threads'] * v['blas_threads'] <= ex.hardware['usable_cpus']]
    for omp, blas in sorted({(v['threads'],v['blas_threads']) for v in variants}):
        variants.append(profile(threads=omp,blas=blas,source='upstream'))
    if ex.args.native_roots:
        # Native-vs-portable occurs on this same machine, using separate
        # worktrees and matched upstream dependencies inside each profile.
        for value in dedupe([matched['winner']['variant'], matched['best_nocache']['variant']]):
            native = copy.deepcopy(value); native['build_profile'] = 'native'
            if native['threads'] * native['blas_threads'] > ex.hardware['usable_cpus']:
                skipped.append(native)
                continue
            variants.append(native)
            variants.append(profile(threads=value['threads'],blas=value['blas_threads'],
                                    source='upstream',build='native'))
        native_upstream = copy.deepcopy(matched['best_upstream']['variant'])
        native_upstream['build_profile'] = 'native'
        if native_upstream['threads'] * native_upstream['blas_threads'] <= ex.hardware['usable_cpus']:
            variants.append(native_upstream)
        else:
            skipped.append(native_upstream)
    result = measure_finalists(ex, variants, 'confirm')
    result.update(stage='confirm', measured_native=bool(ex.args.native_roots),
                  skipped_hardware_incompatible_profiles=skipped,
                  holdout_size=ex.args.n != 4,
                  selection_scope='Winners from the n=4 tuning stage; all confirm samples are fresh')
    return ex.finish('confirm_summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen','tune','confirm'), required=True)
    parser.add_argument('--roots', required=True)
    parser.add_argument('--native-roots')
    parser.add_argument('--output', required=True)
    parser.add_argument('--inputs')
    parser.add_argument('--xc', choices=('PBE','MN15'), required=True)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error('Invalid shard')
    ex = Experiment(args)
    result = {'screen':screen,'tune':tune,'confirm':confirm}[args.stage](ex)
    # Failed options are retained and excluded from the recommendation. A
    # report must disclose them; a partial final matrix cannot claim success.
    if args.stage != 'screen' and result['complete_variants'] != result['expected_variants']:
        raise SystemExit('Incomplete finalist matrix; inspect failures.json')


if __name__ == '__main__':
    main()
