#!/usr/bin/env python3
"""Sequential fresh-process CPKS benchmark campaign with immutable evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_ticks(cpus):
    wanted = {f'cpu{i}' for i in cpus}
    return {fields[0]: list(map(int, fields[1:]))
            for line in Path('/proc/stat').read_text().splitlines()
            if (fields := line.split()) and fields[0] in wanted}


def siblings(cpus):
    result = set(cpus)
    for cpu in cpus:
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
        for item in path.read_text().strip().split(','):
            if '-' in item:
                a, b = map(int, item.split('-'))
                result.update(range(a, b+1))
            else:
                result.add(int(item))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', nargs='+', required=True, metavar='LABEL=CHECKOUT')
    parser.add_argument('--python', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).with_name('fixtures'))
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('response_protocol.json'))
    parser.add_argument('--affinity', required=True, help='Ordered comma-separated physical CPU IDs')
    parser.add_argument('--threads', type=int, nargs='+', default=[8, 24])
    parser.add_argument('--repeats', '--repeat', type=int, default=3)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Use a fresh output namespace; previous evidence is immutable')
    protocol = json.loads(args.protocol.read_text())
    cases = [case for case in protocol['systems'] if args.cases is None or case['name'] in args.cases]
    affinity = list(map(int, args.affinity.split(',')))
    if not cases or args.repeats < 1 or max(args.threads) > len(affinity):
        raise ValueError('Empty cases, invalid repeat count, or insufficient physical CPUs')
    versions = []
    for spec in args.versions:
        label, source = spec.split('=', 1)
        source = str(Path(source).resolve())
        commit = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
        versions.append({'label': label, 'source': source, 'commit': commit})
    if len({v['label'] for v in versions}) != len(versions):
        raise ValueError('Duplicate version labels')
    worker = Path(__file__).with_name('response_benchmark.py')
    args.output.mkdir(parents=True)
    batch = {'schema': 1, 'status': 'running', 'host': os.uname().nodename,
             'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'runner_sha256': sha256(__file__), 'worker_sha256': sha256(worker),
             'protocol_sha256': sha256(args.protocol), 'protocol': protocol,
             'versions': versions, 'cases': cases, 'threads': args.threads,
             'repeats': args.repeats, 'affinity_order': affinity,
             'expected': len(cases)*len(args.threads)*len(versions)*args.repeats,
             'completed': 0, 'rows': [], 'arguments': vars(args).copy(),
             'order': 'sequential fresh processes, version order rotated by replicate'}
    batch['arguments'] = {key: str(value) if isinstance(value, Path) else value
                          for key, value in batch['arguments'].items()}
    ledger = args.output / 'batch.json'
    def save():
        tmp = ledger.with_suffix('.tmp')
        tmp.write_text(json.dumps(batch, indent=2)+'\n')
        tmp.replace(ledger)
    save()
    try:
        for case in cases:
            fixture = args.fixtures / case['fixture']
            fixture_hash = sha256(fixture)
            for threads in args.threads:
                cpus = affinity[:threads]
                monitor = siblings(cpus)
                for repeat in range(args.repeats):
                    offset = repeat % len(versions)
                    for version in versions[offset:]+versions[:offset]:
                        name = f"{case['name']}-t{threads}-{version['label']}-r{repeat}"
                        result = args.output / (name+'.json')
                        env = os.environ.copy()
                        env.pop('LD_LIBRARY_PATH', None)
                        env.update(PYTHONPATH=version['source'], PYSCF_EXT_PATH='/dev/null',
                                   OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
                                   MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
                                   GOMP_SPINCOUNT='0', NUMPY_MADVISE_HUGEPAGE='0')
                        command = ['taskset', '-c', ','.join(map(str, cpus)), args.python,
                                   str(worker), 'run', '--source', version['source'],
                                   '--input', str(fixture), '--output', str(result)]
                        row = {'case': case['name'], 'threads': threads, 'version': version['label'],
                               'commit': version['commit'], 'repeat': repeat,
                               'fixture': str(fixture), 'fixture_sha256': fixture_hash,
                               'result': str(result), 'command': command, 'affinity': cpus,
                               'monitored_cpus': monitor, 'cpu_ticks_before': cpu_ticks(monitor),
                               'clock_ticks_per_second': os.sysconf('SC_CLK_TCK')}
                        usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                        started = time.perf_counter()
                        print('START '+name, flush=True)
                        try:
                            with result.with_suffix('.log').open('w') as stream:
                                child = subprocess.run(command, env=env, stdout=stream,
                                                       stderr=subprocess.STDOUT, timeout=args.timeout)
                            row['returncode'] = child.returncode
                        except subprocess.TimeoutExpired:
                            row['returncode'] = None
                            row['timed_out'] = True
                        row['process_wall_s'] = time.perf_counter()-started
                        usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                        row['child_cpu_s'] = usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime
                        row['cpu_ticks_after'] = cpu_ticks(monitor)
                        row['result_exists'] = result.exists()
                        batch['rows'].append(row)
                        if row['returncode'] != 0 or not result.exists():
                            raise RuntimeError('Response child failed: '+name)
                        observed = json.loads(result.read_text())
                        if (observed['source']['commit'] != version['commit'] or
                            observed['fixture_sha256'] != fixture_hash or
                            not observed['scientific_accepted']):
                            raise RuntimeError('Response child provenance/acceptance failed: '+name)
                        batch['completed'] += 1
                        save()
                        print('DONE '+name, flush=True)
        batch['status'] = 'finished_unvalidated'
    except BaseException as exc:
        batch['status'] = 'failed'
        batch['failure'] = repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
