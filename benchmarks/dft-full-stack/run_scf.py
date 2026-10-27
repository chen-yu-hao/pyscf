"""Run the old full-SCF protocol on four frozen versions, serially."""
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def cpu_sample(cpus):
    values = {}
    for line in Path('/proc/stat').read_text().splitlines():
        f = line.split()
        if f[0].startswith('cpu') and f[0][3:].isdigit() and int(f[0][3:]) in cpus:
            ticks = list(map(int, f[1:9]))
            values[f[0]] = {'total': sum(ticks), 'busy': sum(ticks) - ticks[3] - ticks[4]}
    return values


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    versions = json.loads((ROOT / 'versions.json').read_text())
    output = ROOT / 'runs'
    output.mkdir()
    batch = {'status': 'running', 'expected': protocol['expected'], 'completed': 0,
             'rows': [], 'protocol': protocol, 'versions': versions, 'python': sys.executable,
             'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        tmp = output / 'batch.tmp'
        tmp.write_text(json.dumps(batch, indent=2) + '\n')
        tmp.replace(output / 'batch.json')
    save()
    mask = protocol['cpu_mask']
    observed_cpus = set(mask + protocol['smt_siblings_monitored'])
    try:
        for case_index, spec in enumerate(protocol['cases']):
            for repeat in range(protocol['repeats']):
                shift = (case_index + (0, 3, 7)[repeat]) % len(versions)
                for version in versions[shift:] + versions[:shift]:
                    name = f"{spec['id']}-{version['label']}-r{repeat}"
                    tmpdir = ROOT / 'tmp' / name
                    tmpdir.mkdir()
                    result, log = output / (name + '.json'), output / (name + '.log')
                    env = os.environ.copy()
                    for k in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'PYSCF_EXT_PATH'):
                        env.pop(k, None)
                    env.update(PYTHONPATH=version['root'], PYSCF_EXT_PATH='/dev/null',
                               TMPDIR=str(tmpdir), PYSCF_TMPDIR=str(tmpdir),
                               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
                               GOMP_SPINCOUNT='0', NUMPY_MADVISE_HUGEPAGE='0', PYTHONDONTWRITEBYTECODE='1')
                    cmd = ['taskset', '-c', ','.join(map(str, mask)), sys.executable,
                           str(ROOT / 'pyscf_case.py'), '--case', spec['id'], '--cores', '8',
                           '--xc', protocol['xc'], '--cutoff', str(protocol['grid']['pyscf_ao_cutoff']),
                           '--output', str(result)]
                    row = {'case': spec['id'], 'engine': version['label'], 'repeat': repeat,
                           'cores': 8, 'affinity': mask, 'result': str(result), 'log': str(log),
                           'command': cmd, 'scratch': str(tmpdir),
                           'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           'cpu_ticks_before': cpu_sample(observed_cpus)}
                    batch['rows'].append(row)
                    save()
                    usage0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    started = time.perf_counter()
                    print('START', name, flush=True)
                    with log.open('w') as stream:
                        child = subprocess.run(cmd, env=env, cwd=tmpdir, stdout=stream,
                                               stderr=subprocess.STDOUT, timeout=1800)
                    row['process_wall_s'] = time.perf_counter() - started
                    usage1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                    row.update(returncode=child.returncode,
                               process_cpu_seconds=usage1.ru_utime + usage1.ru_stime - usage0.ru_utime - usage0.ru_stime,
                               cpu_ticks_after=cpu_sample(observed_cpus),
                               ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    if child.returncode or not result.is_file():
                        raise RuntimeError('Failed ' + name)
                    values = json.loads(result.read_text())
                    row.update(energy=values['energy'], cycles=values['cycles'],
                               kernel_wall_s=values['kernel_wall_s'], strict=values['strict'])
                    if not values['strict']:
                        raise RuntimeError('SCF not converged ' + name)
                    batch['completed'] += 1
                    save()
                    print('DONE', batch['completed'], '/', batch['expected'], name,
                          f"{row['process_wall_s']:.3f}s", 'cycles', values['cycles'], flush=True)
        batch['status'] = 'finished_unvalidated'
    except BaseException as error:
        batch.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
