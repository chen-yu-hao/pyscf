"""Build each cumulative stage, reusing only unchanged native modules."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/lyl/soft/pyscf-split-20261026/base')
PYTHON = '/home/lyl/soft/conda-envs/pyscf-new/bin/python'
CMAKE = '/home/lyl/soft/conda-envs/pyscf-new/bin/cmake'
CPUS = '20-23'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()


def main():
    stack = json.loads((ROOT / 'stack.json').read_text())
    previous = BASE
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH', None)
    for stage in stack['stages']:
        started = time.perf_counter()
        repo, label = Path(stage['source']), stage['label']
        assert git(repo, 'rev-parse', 'HEAD') == stage['head']
        assert not git(repo, 'status', '--porcelain', '--untracked-files=no')
        out = ROOT / 'builds' / label
        out.mkdir(parents=True)
        native, reference = repo / 'pyscf/lib', previous / 'pyscf/lib'
        shutil.copytree(reference / 'deps', native / 'deps', dirs_exist_ok=True)
        for file in reference.glob('*.so*'):
            shutil.copy2(file, native / file.name)
        shutil.copy2(reference / 'config.h', native / 'config.h')
        changed = git(repo, 'diff', '--name-only', stage.get('parent', stage['head']), stage['head'], '--', 'pyscf/lib').splitlines()
        targets = set()
        for file in changed:
            if file == 'pyscf/lib/dft/libxc_itrf.c':
                targets.add('xc_itrf')
            elif file.startswith('pyscf/lib/gto/'):
                targets.add('cgto')
            elif file.startswith('pyscf/lib/dft/'):
                targets.add('dft')
            else:
                raise RuntimeError('Unexpected native source change ' + file)
        commands = []
        if targets:
            commands.append(['taskset', '-c', CPUS, CMAKE, '-S', str(native), '-B', str(out / 'cmake'),
                             '-DCMAKE_C_COMPILER=/usr/bin/gcc', '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
                             '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG',
                             '-DBUILD_MARCH_NATIVE=OFF', '-DBUILD_LIBCINT=OFF', '-DBUILD_LIBXC=OFF',
                             '-DBUILD_XCFUN=OFF', '-DENABLE_OPENMP=ON', '-DENABLE_LIBXC=ON',
                             '-DENABLE_XCFUN=ON', '-DBLAS_LIBRARIES=' + str(native / 'libopenblas-r0-f650aae0.3.3.so')])
            commands.append(['taskset', '-c', CPUS, CMAKE, '--build', str(out / 'cmake'),
                             '--parallel', '4', '--target', *sorted(targets)])
            for index, command in enumerate(commands):
                with (out / f'build-{index}.log').open('w') as stream:
                    subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        audit_env = dict(env, PYTHONPATH=str(repo), PYSCF_EXT_PATH='/dev/null',
                         OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        code = "import json,pyscf; from pyscf import dft,gto; p=sorted({l.split()[-1] for l in open('/proc/self/maps') if '.so' in l and '/pyscf/lib/' in l}); print(json.dumps({'pyscf':pyscf.__file__,'libxc':dft.libxc.__version__,'native_maps':p}))"
        audit = subprocess.run(['taskset', '-c', CPUS, PYTHON, '-c', code], env=audit_env,
                               capture_output=True, text=True, check=True)
        info = json.loads(audit.stdout)
        assert Path(info['pyscf']).is_relative_to(repo)
        assert all(Path(p).is_relative_to(repo) for p in info['native_maps'])
        assert sum(Path(p).name == 'libcint.so' for p in info['native_maps']) == 1
        sources = git(repo, 'ls-files', 'pyscf/lib').splitlines()
        manifest = {'label': label, 'head': stage['head'], 'status': 'ready', 'source': str(repo),
                    'source_hashes': {p: sha(repo / p) for p in sources if Path(p).suffix in ('.c', '.h', '.txt', '.in')},
                    'native_hashes': {str(p.relative_to(repo)): sha(p) for p in native.rglob('*.so*') if p.is_file()},
                    'reused_native_from': str(previous), 'rebuilt_targets': sorted(targets),
                    'native_source_changes': changed, 'commands': commands, 'audit': info,
                    'compiler': 'GCC 11.3.1', 'cmake': '4.3.1', 'libxc': info['libxc'],
                    'c_flags': '-Wall -msse3 -O3 -DNDEBUG -std=gnu99 -fPIC -fopenmp',
                    'duration_s': time.perf_counter() - started}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('BUILT', label, stage['head'], sorted(targets), round(manifest['duration_s'], 2), flush=True)
        previous = repo


if __name__ == '__main__':
    main()
