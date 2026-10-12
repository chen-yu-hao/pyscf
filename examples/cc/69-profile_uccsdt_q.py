#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

'''Measure UCCSDT(Q) wall time and native-inclusive resident memory on Linux.

Each thread setting runs in a fresh process: one cold correction followed by
warm repeats, with ERIs already in memory. Inputs are synthetic, not molecular
or literature benchmarks. RSS is sampled by the parent; ru_maxrss also includes
input preparation and is reported separately. A sampled peak is a lower bound.

Example (from a built checkout):
  python examples/cc/69-profile_uccsdt_q.py --nocc 3 --nvir 24 \
      --threads 1 2 4 8 --cpu-affinity 0-7 --output profile.json

Replay an archived Python/C pair with --phase baseline --baseline-root DIR.
DIR must contain pyscf/cc/uccsdt_q.py and pyscf/lib/libccsdt.so.
'''

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import queue
import resource
import statistics
import subprocess
import sys
import threading
import time


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rss_bytes(pid):
    try:
        with open('/proc/%d/status' % pid) as stream:
            for line in stream:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def io_bytes():
    with open('/proc/self/io') as stream:
        counters = dict(line.split(':', 1) for line in stream)
    return {name: int(counters[name]) for name in ('read_bytes', 'write_bytes')}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def child(args):
    if args.cpu_affinity:
        cpus = set()
        for part in args.cpu_affinity.split(','):
            bounds = list(map(int, part.split('-')))
            cpus.update(range(bounds[0], bounds[-1]+1))
        os.sched_setaffinity(0, cpus)
    import numpy as np
    import pyscf
    from pyscf import lib
    from pyscf.cc import uccsdt

    repo = Path(__file__).resolve().parents[2]
    source_root = args.baseline_root if args.phase == 'baseline' else repo
    module = load_module('_profile_uccsdt_q', source_root / 'pyscf/cc/uccsdt_q.py')
    if args.phase == 'baseline':
        # Let the archived module itself configure the ctypes ABI, then retain
        # its cached CDLL. This never changes the installed library on disk.
        original_loader = lib.load_library
        native = ctypes.CDLL(str(source_root / 'pyscf/lib/libccsdt.so'))
        try:
            lib.load_library = lambda name: native if name == 'libccsdt' else original_loader(name)
            module._native_library()
        finally:
            lib.load_library = original_loader
    native = module._native_library()
    lib.num_threads(args.threads[0])
    factory_path = source_root / 'examples/cc/68-benchmark_uccsdt_q.py'
    factory = load_module('_profile_input_factory', factory_path)
    integral_seconds = 0.
    original_einsum = np.einsum

    def timed_einsum(*a, **kw):
        nonlocal integral_seconds
        start = time.perf_counter()
        result = original_einsum(*a, **kw)
        integral_seconds += time.perf_counter() - start
        return result

    prep_start = time.perf_counter()
    try:
        np.einsum = timed_einsum
        mycc, eris = factory.make_inputs(args.nocc, args.nvir)
    finally:
        np.einsum = original_einsum
    conversion_start = time.perf_counter()
    if args.representation == 'full':
        names = ['t2c_%s_%df_%s%s' % (kind, fold, space, spin)
                 for spin in 'ab' for space in 'ov' for fold in (6, 2)
                 for kind in ('map', 'mask')]
        for name, array in zip(names, uccsdt.setup_tri2block_t3_uhf(mycc)):
            setattr(mycc, name, array)
        mycc.cc_order = 3
        mycc.tamps[2] = uccsdt.tamps_tri2full_uhf(mycc, mycc.tamps[2])
        mycc.do_tri_max_t = False
        for name in names:
            delattr(mycc, name)
        del array
    conversion_seconds = time.perf_counter() - conversion_start
    gc.collect()
    mycc.max_memory = args.max_memory
    native_path = Path(native._name).resolve()
    metadata = {
        'phase': args.phase, 'python': platform.python_version(), 'numpy': np.__version__,
        'pyscf': pyscf.__version__, 'pyscf_source': pyscf.__file__,
        'source_root': str(source_root), 'native_path': str(native_path),
        'native_sha256': sha256(native_path),
        'source_sha256': {name: sha256(source_root / name) for name in
                          ('pyscf/cc/uccsdt_q.py', 'pyscf/lib/ccsdt/uccsdt_q.c',
                           'examples/cc/68-benchmark_uccsdt_q.py')},
        'profiler_sha256': sha256(__file__),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'thread_environment': {name: os.environ.get(name) for name in
                               ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'requested_threads': args.threads[0], 'seed': 3049,
        'nocc': [args.nocc]*2, 'nvir': [args.nvir]*2,
        'representation': args.representation, 'forced_block_size': args.block_size,
        'max_memory_mb': mycc.max_memory, 'integrals_preexisting_at_correction': True,
        'input_preparation_wall_seconds': time.perf_counter()-prep_start,
        'synthetic_integrals_wall_seconds': integral_seconds,
        'representation_conversion_wall_seconds': conversion_seconds,
        'input_resident_rss_bytes': rss_bytes(os.getpid()),
        'input_array_bytes': {
            'eri': sum(a.nbytes for a in (eris.pppp, eris.pPpP, eris.PPPP)),
            'amplitudes': sum(a.nbytes for blocks in mycc.tamps for a in blocks),
        },
        'preparation_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    }
    emit('metadata', metadata=metadata)
    original_block_size = module._native_block_size
    original_batch = module._native_batch
    original_table = module._native_term_table
    stats = {}

    def measure_table(*a, **kw):
        start = time.perf_counter()
        result = original_table(*a, **kw)
        stats['formula_table_wall_seconds'] += time.perf_counter()-start
        return result

    def measure_block_size(cc, nocc, nvir, records, offsets, nthreads):
        block = args.block_size or original_block_size(cc, nocc, nvir, records, offsets, nthreads)
        sizes = np.asarray((*nocc, *nvir), dtype=np.int64)
        workspace = native.UCCSDTQ_workspace(sizes.ctypes.data, records.ctypes.data,
                                           offsets.ctypes.data, block)
        stats.update(block_size=block, native_workspace_per_thread_bytes=int(workspace),
                     total_tasks=module._task_count(nocc, nvir, block))
        return block

    def measure_batch(*a, **kw):
        start = time.perf_counter()
        result = original_batch(*a, **kw)
        stats['native_batches_wall_seconds'] += time.perf_counter()-start
        stats['native_batches'] += 1
        stats['actual_threads'] = max(stats['actual_threads'], result[2])
        return result

    module._native_term_table = measure_table
    module._native_block_size = measure_block_size
    module._native_batch = measure_batch
    first_energy = None
    for index in range(args.repeats+1):
        stage = 'cold' if index == 0 else 'warm_%d' % index
        stats.clear()
        stats.update(native_batches_wall_seconds=0., formula_table_wall_seconds=0.,
                     native_batches=0, actual_threads=0)
        initial_rss = rss_bytes(os.getpid())
        emit('ready', stage=stage, rss_before_bytes=initial_rss)
        if sys.stdin.readline().strip() != 'go':
            raise RuntimeError('Missing parent sampling handshake')
        before_io = io_bytes()
        before_usage = resource.getrusage(resource.RUSAGE_SELF)
        start_cpu, start = time.process_time(), time.perf_counter()
        energy = np.asarray(module.kernel(mycc, eris=eris, verbose=0))
        elapsed = time.perf_counter()-start
        cpu_seconds = time.process_time()-start_cpu
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        after_io = io_bytes()
        if first_energy is None:
            first_energy = energy
        delta = float(np.max(np.abs(energy-first_energy)))
        np.testing.assert_allclose(energy, first_energy, atol=1e-11, rtol=1e-10)
        emit('result', stage=stage, result={
            **stats, 'wall_seconds': elapsed, 'cpu_seconds': cpu_seconds,
            'io_bytes': {name: after_io[name]-before_io[name] for name in before_io},
            'minor_page_faults': after_usage.ru_minflt-before_usage.ru_minflt,
            'major_page_faults': after_usage.ru_majflt-before_usage.ru_majflt,
            'energies_eh': energy.tolist(), 'finite': bool(np.isfinite(energy).all()),
            'max_abs_cold_energy_difference_eh': delta,
            'rss_before_bytes': initial_rss, 'rss_after_bytes': rss_bytes(os.getpid()),
            'ru_maxrss_total_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'native_workspace_total_bytes': stats['native_workspace_per_thread_bytes']*stats['actual_threads'],
        })


def profile_process(args, threads):
    command = [sys.executable, str(Path(__file__).resolve()), '--child',
               '--phase', args.phase, '--nocc', str(args.nocc), '--nvir', str(args.nvir),
               '--threads', str(threads), '--representation', args.representation,
               '--repeats', str(args.repeats), '--max-memory', str(args.max_memory)]
    for option, value in (('--baseline-root', args.baseline_root),
                          ('--cpu-affinity', args.cpu_affinity), ('--block-size', args.block_size)):
        if value is not None:
            command.extend((option, str(value)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OMP_NUM_THREADS=str(threads), OMP_PROC_BIND='false', OMP_PLACES='cores')
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env, bufsize=1)
    events, errors = queue.Queue(), []

    def read_events():
        for line in process.stdout:
            events.put(json.loads(line))

    def read_errors():
        errors.extend(process.stderr.readlines())

    reader = threading.Thread(target=read_events, daemon=True)
    err_reader = threading.Thread(target=read_errors, daemon=True)
    reader.start()
    err_reader.start()
    result = {'measurements': [], 'command': command}
    active, samples, peak = None, 0, 0
    deadline = time.monotonic()+args.timeout
    try:
        while process.poll() is None or not events.empty() or reader.is_alive():
            if time.monotonic() > deadline:
                raise TimeoutError('Profile exceeded %.1f seconds: %s' % (args.timeout, command))
            rss = rss_bytes(process.pid)
            if active:
                samples += 1
                peak = max(peak, rss)
            try:
                event = events.get(timeout=args.sample_ms/1000)
            except queue.Empty:
                continue
            if event['event'] == 'metadata':
                result['metadata'] = event['metadata']
            elif event['event'] == 'ready':
                active = event['stage']
                samples, peak = 1, max(rss, event['rss_before_bytes'])
                deadline = time.monotonic()+args.timeout
                process.stdin.write('go\n')
                process.stdin.flush()
            elif event['event'] == 'result':
                row = event['result']
                peak = max(peak, row['rss_after_bytes'])
                row.update(stage=active, sampled_peak_rss_bytes=peak, rss_samples=samples,
                           sampled_rss_growth_bytes=max(0, peak-row['rss_before_bytes']))
                result['measurements'].append(row)
                active = None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join()
        err_reader.join()
    if process.returncode:
        raise RuntimeError('Child failed (%d):\n%s' % (process.returncode, ''.join(errors)))
    result['stderr'] = ''.join(errors)
    warm = result['measurements'][1:]
    result['median_warm_wall_seconds'] = statistics.median(row['wall_seconds'] for row in warm)
    result['median_warm_cpu_seconds'] = statistics.median(row['cpu_seconds'] for row in warm)
    result['max_sampled_correction_peak_rss_bytes'] = max(
        row['sampled_peak_rss_bytes'] for row in result['measurements'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baseline', 'current'), default='current')
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--nocc', type=int, default=3)
    parser.add_argument('--nvir', type=int, default=12)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--representation', choices=('compact', 'full'), default='compact')
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--max-memory', type=float, default=4000.)
    parser.add_argument('--repeats', type=int, default=3, help='warm repeats after one cold run')
    parser.add_argument('--sample-ms', type=float, default=2.)
    parser.add_argument('--timeout', type=float, default=120., help='maximum seconds per stage')
    parser.add_argument('--cpu-affinity', help='Linux CPU list, e.g. 0-7 or 0,2,4,6')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.nocc, args.nvir, args.repeats, *args.threads) < 1:
        parser.error('dimensions, repeat count and threads must be positive')
    if args.block_size is not None and args.block_size < 1:
        parser.error('block size must be positive')
    if args.phase == 'baseline' and args.baseline_root is None:
        parser.error('--phase baseline requires --baseline-root')
    if args.child:
        return child(args)
    report = {
        'description': 'Synthetic UCCSDT(Q), preexisting ERIs, fresh process per thread count',
        'rss_sampling_interval_ms': args.sample_ms,
        'memory_note': 'Sampled /proc VmRSS includes native malloc but may miss short peaks; '
                       'ru_maxrss_total includes startup/preparation/warmups. Neither is tracemalloc.',
        'io_note': 'Linux process physical I/O counters during correction only. Inputs are '
                   'already resident; zero reads do not validate out-of-core performance.',
        'cases': [],
    }
    for threads in args.threads:
        result = profile_process(args, threads)
        report['cases'].append(result)
        reference = report['cases'][0]['measurements'][0]['energies_eh']
        result['max_abs_thread_energy_difference_eh'] = max(
            abs(value-expected) for row in result['measurements']
            for value, expected in zip(row['energies_eh'], reference))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
        if not all(math.isclose(value, expected, abs_tol=1e-11, rel_tol=1e-10)
                   for row in result['measurements']
                   for value, expected in zip(row['energies_eh'], reference)):
            raise AssertionError('UCCSDT(Q) energy changed across thread counts')
        print('threads=%d warm=%.6fs peakRSS=%.3f MB' %
              (threads, result['median_warm_wall_seconds'],
               result['max_sampled_correction_peak_rss_bytes']/1e6), file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
