"""Serial, alternating-order grid-build and full SCF comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_xc_campaign import cpu_ticks
from xc_benchmark import checkout_info, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--versions', nargs=2, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir()
    protocol = json.loads(a.protocol.read_text())
    versions = []
    for spec in a.versions:
        label, path = spec.split('=', 1)
        source = checkout_info(Path(path).resolve())
        assert source['commit'] == protocol['source_commits'][label]
        versions.append({'label': label, 'source': source['root'], 'commit': source['commit']})
    worker = Path(__file__).with_name('end_to_end.py')
    batch = {'status': 'running', 'expected': protocol['expected_processes'], 'completed': 0,
             'rows': [], 'versions': versions, 'protocol': protocol, 'repeats': 3,
             'host': os.uname().nodename, 'runner_sha256': digest(Path(__file__)),
             'worker_sha256': digest(worker), 'protocol_sha256': digest(a.protocol)}
    def save():
        temp = a.output / 'batch.tmp'
        temp.write_text(json.dumps(batch, indent=2) + '\n')
        temp.replace(a.output / 'batch.json')
    save()
    try:
        for cell in protocol['cells']:
            cpus = protocol['affinity_order'][:cell['threads']]
            for repeat in range(3):
                for v in versions[repeat % 2:] + versions[:repeat % 2]:
                    name = '-'.join([cell['mode'], cell['case'], cell.get('xc', '').replace(',', '_'),
                                     't' + str(cell['threads']), v['label'], 'r' + str(repeat)])
                    result = a.output / (name + '.json')
                    cmd = [sys.executable, str(worker), '--source', v['source'], '--commit', v['commit'],
                           '--protocol', str(a.protocol), '--mode', cell['mode'], '--case', cell['case'],
                           '--threads', str(cell['threads']), '--affinity', ','.join(map(str, cpus)),
                           '--fixtures', str(a.fixtures), '--output', str(result)]
                    if 'xc' in cell:
                        cmd += ['--xc', cell['xc']]
                    row = {**cell, 'version': v['label'], 'repeat': repeat, 'result': str(result),
                           'command': cmd, 'cpu_ticks_before': cpu_ticks(cpus)}
                    batch['rows'].append(row)
                    save()
                    print('START', name, flush=True)
                    start = time.perf_counter()
                    with result.with_suffix('.log').open('w') as log:
                        child = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                    row.update(process_wall_seconds=time.perf_counter() - start,
                               returncode=child.returncode, cpu_ticks_after=cpu_ticks(cpus))
                    if child.returncode or not result.is_file():
                        raise RuntimeError('Failed ' + name)
                    batch['completed'] += 1
                    save()
                    print('DONE', batch['completed'], '/', batch['expected'], name,
                          round(row['process_wall_seconds'], 3), flush=True)
        batch['status'] = 'finished_unvalidated'
    except BaseException as error:
        batch.update(status='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
