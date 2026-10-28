"""Default NUMA XC partitions with optional measured selection for RKS.

Unsupported configurations, failed binding and insufficient memory use native
integration. Optional auto mode additionally checks speed and numerical values.
No molecule, density or grid data is persisted beyond the SCF worker lifetime.
"""
import hashlib
import atexit
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import weakref

import numpy as np
import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.dft._numa_topology import discover, local_grid_partitions
from pyscf.dft._numa_worker import PartitionedXC

_states = weakref.WeakKeyDictionary()
_plans = weakref.WeakKeyDictionary()
_registry_lock = threading.RLock()
libxc = numint.libxc  # also allow native XCFun-only installations to import


def _quota():
    """Conservatively obey cgroup v2 CPU quotas, including ancestor limits.

    A v1 CPU controller is declined for now: never assume affinity represents
    a container's quota. Unknown/unreadable active cgroup constraints disable
    the optimization rather than expanding the allocation.
    """
    limits = []
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    if any('cpu' in line.split(':')[1].split(',') for line in lines):
        return 0
    entries = [line.split(':', 2)[2] for line in lines if line.startswith('0::')]
    root = Path('/sys/fs/cgroup')
    if not (root / 'cgroup.controllers').exists():
        return math.inf  # host without a mounted v2 controller
    if not entries or '..' in Path(entries[0]).parts:
        return 0
    directory = root / entries[0].lstrip('/')
    if not directory.exists():
        return 0
    while True:
        path = directory / 'cpu.max'
        if path.exists():
            count, period = path.read_text().split()
            if count != 'max':
                limits.append(int(count) / int(period))
        if directory == root:
            break
        directory = directory.parent
    return min(limits, default=math.inf)


def _partitions():
    for name in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE'):
        if int(os.getenv(name, '1')) > 1:
            return [], []  # do not replicate a worker team for every MPI rank
    initial = set(pyscf._initial_cpu_affinity)
    current = set(os.sched_getaffinity(0))
    if not initial:
        return [], []
    # A singleton can be OpenMP's pinned master. Never expand an explicit
    # later multi-CPU narrowing of the original process allocation.
    allowed = initial if len(current) == 1 else initial & current
    topo = discover(affinity=allowed)
    quota = _quota()
    budget = min(lib.num_threads(), len(topo.cores))
    if os.getenv('SLURM_CPUS_PER_TASK'):
        budget = min(budget, int(os.environ['SLURM_CPUS_PER_TASK']))
    if math.isfinite(quota):
        budget = min(budget, int(quota))
    groups = local_grid_partitions(topo, [c.cpu for c in topo.cores])
    if len(groups) < 2 or budget < 4:
        return [], []
    # Balanced round-robin allocation avoids filling one socket first.
    chosen = [[] for _ in groups]
    for index in range(max(map(lambda g: len(g[0]), groups))):
        for i, (cpus, _) in enumerate(groups):
            if index < len(cpus) and sum(map(len, chosen)) < budget:
                chosen[i].append(cpus[index])
    result = [(cpus, groups[i][1]) for i, cpus in enumerate(chosen) if len(cpus) >= 2]
    if len(result) < 2:
        return [], []
    return result, sorted(allowed)


def _memory_headroom():
    """Available MB after host and cgroup-v2 limits, with a 25% reserve."""
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    if available is None:
        return 0.
    root = Path('/sys/fs/cgroup')
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::') and (root / 'cgroup.controllers').exists():
            relative = line.split(':', 2)[2]
            if '..' in Path(relative).parts:
                return 0.
            directory = root / relative.lstrip('/')
            if not directory.exists():
                return 0.
            while True:
                maximum, current = directory / 'memory.max', directory / 'memory.current'
                if maximum.exists() and current.exists():
                    value = maximum.read_text().strip()
                    if value != 'max':
                        available = min(available, max(0, int(value) - int(current.read_text())))
                if directory == root:
                    break
                directory = directory.parent
    return .75 * available / 1e6


def _supported(ks, mol, dm, hermi):
    # Standard real closed-shell RKS only. Do not change callbacks, XCFun,
    # custom Libxc parameters, response/periodic integration or range overrides.
    from pyscf.dft.rks import RKS
    ni = ks._numint
    return (type(ks) is RKS and type(ni) is numint.NumInt and mol is ks.mol
            and mol.spin == 0 and hermi == 1 and not mol.cart
            and isinstance(dm, np.ndarray) and dm.ndim == 2 and dm.dtype == np.float64
            and getattr(dm, 'mo_coeff', None) is not None
            and ni.libxc is libxc and libxc.__name__ == 'pyscf.dft.libxc' and ni.omega is None
            and not libxc._CUSTOM_FUNC_R and not libxc._CUSTOM_FUNC_U
            and not any(callable(v) for v in ni.__dict__.values())
            and isinstance(ks.xc, str) and not ks.do_nlc()
            and ni.nr_rks.__func__ is numint.nr_rks
            and mol.nao >= 200 and len(ks.grids.weights) >= 100000
            and ks.max_cycle >= 8 and not ks.converged)


def _signature(ks, mol):
    digest = hashlib.sha256()
    for arr in (mol._atm, mol._bas, mol._env, ks.grids.coords,
                ks.grids.weights, ks.grids.non0tab):
        if arr is None:
            digest.update(b'None')
        else:
            digest.update(str((arr.shape, arr.dtype.str)).encode())
            digest.update(np.ascontiguousarray(arr).data)
    digest.update(repr((ks.xc, ks.grids.cutoff, ks._numint.cutoff,
                        mol.cart, lib.num_threads(), ks.max_memory,
                        tuple(sorted(os.sched_getaffinity(0))))).encode())
    return digest.digest()


class _State:
    def __init__(self, key):
        self.key = key
        self.pool = None
        self.directory = None
        self.disabled = False
        self.lock = threading.RLock()

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def __del__(self):
        self.close()


def close(ks):
    """Release this calculation's workers; normal kernel/reset do this too."""
    with _registry_lock:
        state = _states.pop(ks, None)
        _plans.pop(ks, None)
    if state is not None:
        with state.lock:
            state.close()


def _close_all():
    for state in list(_states.values()):
        state.close()


atexit.register(_close_all)


def _equal(reference, candidate):
    return all(np.allclose(x, y, rtol=1e-10, atol=1e-9, equal_nan=False)
               for x, y in zip(reference, candidate))


def _get_plan(ks):
    # No grid hashing or worker initialization on a UMA/native-only machine.
    # Cache topology per calculation, but invalidate allocation/quota changes.
    resource_key = (tuple(sorted(os.sched_getaffinity(0))), lib.num_threads(),
                    _quota(), os.getenv('SLURM_CPUS_PER_TASK'),
                    tuple(os.getenv(k) for k in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE')))
    with _registry_lock:
        saved = _plans.get(ks)
        if saved is None or saved[0] != resource_key:
            groups, allowed = _partitions()
            _plans[ks] = (resource_key, groups, allowed)
        return _plans[ks][1:]


def nr_rks(ks, mol, dm, hermi, max_memory):
    native = lambda: ks._numint.nr_rks(mol, ks.grids, ks.xc, dm, max_memory=max_memory)
    mode = os.getenv('PYSCF_NUMA', 'on').lower()
    if mode not in ('on', 'auto') or os.name != 'posix' or not hasattr(os, 'sched_getaffinity'):
        close(ks)
        return native()
    if not _supported(ks, mol, dm, hermi):
        close(ks)
        return native()
    try:
        groups, allowed = _get_plan(ks)
    except (OSError, ValueError):
        close(ks)
        return native()
    if not groups:
        # Preserve the lightweight negative topology cache for later SCF calls.
        with _registry_lock:
            previous = _states.pop(ks, None)
        if previous is not None:
            previous.close()
        return native()
    key = (mode, tuple((tuple(cpus), node) for cpus, node in groups), _signature(ks, mol))
    with _registry_lock:
        state = _states.get(ks)
        if state is not None and state.key != key:
            state.close()
            state = None
        if state is None:
            state = _State(key)
            _states[ks] = state
    with state.lock:
        if state.disabled:
            return native()
        if state.pool is not None:
            try:
                return state.pool.evaluate(dm)
            except (OSError, EOFError, RuntimeError, ValueError) as error:
                state.close()
                state.disabled = True
                logger.warn(ks, 'NUMA XC worker failed; using native integration: %s', error)
                return native()
        reference = None
        baseline = None
        if mode == 'auto':
            # Optional calibration; direct-on mode never duplicates this XC call.
            stamp = time.perf_counter()
            reference = native()
            baseline = time.perf_counter() - stamp
        def fallback():
            return reference if reference is not None else native()
        state.disabled = True
        if (baseline is not None and baseline < .5) or max_memory < 1024:
            return fallback()
        try:
            executable = shutil.which('numactl')
            if executable is None:
                return fallback()
            for cpus, node in groups:
                subprocess.run([executable, '--all', '--physcpubind=' + ','.join(map(str, cpus)),
                                '--membind=' + str(node), '/bin/true'], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            # Reserve room for worker interpreters, final matrices and native
            # thread workspaces. max_memory is a hint, never an OS hard limit.
            reserve = len(groups) * (192 + mol.nao ** 2 * 8 / 1e6 * 6)
            usable = min(max_memory, _memory_headroom()) - reserve
            if usable / len(groups) < 256:
                return fallback()
            state.directory = tempfile.TemporaryDirectory(prefix='pyscf-numa-')
            stamp = time.perf_counter()
            state.pool = PartitionedXC(mol, ks.grids, ks.xc, groups,
                                      allowed_cpus=allowed, memory_mb=usable,
                                      ni_cutoff=ks._numint.cutoff,
                                      log_prefix=str(Path(state.directory.name) / 'xc'))
            setup = time.perf_counter() - stamp
            stamp = time.perf_counter()
            candidate = state.pool.evaluate(dm)
            elapsed = time.perf_counter() - stamp
            if mode == 'on':
                if not all(np.all(np.isfinite(x)) for x in candidate):
                    raise ValueError('Non-finite NUMA XC result')
                state.disabled = False
                logger.info(ks, 'NUMA XC enabled: %s cores/group (no calibration)',
                            [len(cpus) for cpus, _ in groups])
                return candidate
            # Calibration overhead must be recoverable in six later calls;
            # reject marginal improvements and always check the full matrix.
            if (_equal(reference, candidate) and elapsed < .8 * baseline
                    and 6 * (baseline - elapsed) > setup + elapsed):
                state.disabled = False
                logger.info(ks, 'NUMA XC enabled: %s cores/group; native %.3fs, partition %.3fs',
                            [len(cpus) for cpus, _ in groups], baseline, elapsed)
            else:
                state.close()
                logger.info(ks, 'NUMA XC retained native integration (%.3fs vs %.3fs)', baseline, elapsed)
        except (OSError, EOFError, RuntimeError, ValueError, ImportError, MemoryError,
                subprocess.SubprocessError) as error:
            state.close()
            logger.info(ks, 'NUMA XC unavailable; retained native integration: %s', error)
        return fallback()
