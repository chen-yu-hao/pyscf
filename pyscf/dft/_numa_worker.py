"""Experimental CPU-only RKS XC grid partitions in persistent NUMA workers.

This is an opt-in benchmark adapter, not a production NumInt replacement.
Workers start through exec (never fork after OpenMP initialization), own their
grid/mask arrays, and receive the current density and optional MO tags each call.
Only Ne, Exc and the final Vxc matrix cross back to the coordinating process.
"""
from multiprocessing.connection import Connection
from pathlib import Path
import os
import socket
import shutil
import subprocess
import sys
import time
import traceback

def environment(cpus):
    env = dict(os.environ)
    # Propagate the calling interpreter's active package resolution. No fixed
    # conda environment, source checkout, CPU ids or NUMA node ids are used.
    env['PYTHONPATH'] = os.pathsep.join(str(p) for p in sys.path if p)
    env.update(OMP_NUM_THREADS=str(len(cpus)), OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE',
               OMP_PROC_BIND='true', OMP_WAIT_POLICY='PASSIVE',
               PYSCF_NUMA='off',
               OMP_PLACES=','.join('{' + str(c) + '}' for c in cpus))
    return env


def import_numerics():
    import numpy as np
    from pyscf import dft, gto, lib
    from threadpoolctl import threadpool_limits
    limiter = threadpool_limits(limits=1, user_api='blas')
    return np, dft, gto, lib, limiter


def worker(fd):
    conn = Connection(fd)
    try:
        start = time.perf_counter()
        np, dft, gto, lib, limiter = import_numerics()
        import pyscf
        import resource
        from threadpoolctl import threadpool_info
        conn.send({'imported': True})
        phases = {}

        def instrument(obj, name):
            original = getattr(obj, name)

            def timed(*args, **kwargs):
                stamp = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    phases[name] = phases.get(name, 0.) + time.perf_counter() - stamp

            setattr(obj, name, timed)

        def configure(cfg):
            if str(Path(pyscf.__file__).resolve()) != cfg['pyscf_file']:
                raise RuntimeError('Worker resolved a different PySCF environment')
            lib.num_threads(len(cfg['cpus']))
            mol = gto.loads(cfg['mol'])
            mol.verbose = 0
            mol.max_memory = cfg['memory_mb']
            grids = dft.gen_grid.Grids(mol)
            grids.cutoff = cfg['grid_cutoff']
            # These allocations are first-touched after numactl memory binding.
            grids.coords = np.array(cfg.pop('coords'), order='C', copy=True)
            grids.weights = np.array(cfg.pop('weights'), order='C', copy=True)
            grids.non0tab = np.array(cfg.pop('mask'), order='C', copy=True)
            ni = dft.numint.NumInt()
            ni.cutoff = cfg['ni_cutoff']
            if cfg['block_size'] is not None:
                original_loop = ni.block_loop

                def block_loop(*args, **kwargs):
                    kwargs['blksize'] = cfg['block_size']
                    return original_loop(*args, **kwargs)

                ni.block_loop = block_loop
            # Initialize XC caches before repeated timing, equally for all layouts.
            ni._xc_type(cfg['xc'])
            for name in ('eval_ao', 'eval_rho1', 'eval_rho2', 'eval_xc_eff'):
                instrument(ni, name)
            return mol, grids, ni

        # Module-level function: wrap once per process, not per configuration.
        instrument(dft.numint.libdft, 'VXCxc_vmat')
        cfg = conn.recv()
        mol, grids, ni = configure(cfg)
        affinity = sorted(os.sched_getaffinity(0))
        assert affinity == [cfg['cpus'][0]], (affinity, cfg['cpus'])
        conn.send({'ready': True, 'pid': os.getpid(), 'affinity': affinity,
                   'pyscf_file': str(Path(pyscf.__file__).resolve()),
                   'startup_seconds': time.perf_counter() - start,
                   'grid_points': len(grids.weights), 'pools': threadpool_info()})
        while True:
            request = conn.recv()
            if request is None:
                break
            if 'reload' in request:
                # New molecule/grid (e.g. the next geometry step) for this
                # running worker: no interpreter start, import or CPU binding.
                stamp = time.perf_counter()
                mol = grids = ni = None
                cfg = request['reload']
                mol, grids, ni = configure(cfg)
                conn.send({'ready': True, 'pid': os.getpid(),
                           'reload_seconds': time.perf_counter() - stamp,
                           'grid_points': len(grids.weights)})
                continue
            dm = request['density']
            if request.get('mo_coeff') is not None:
                dm = lib.tag_array(dm, mo_coeff=request['mo_coeff'],
                                   mo_occ=request['mo_occ'])
            phases.clear()
            stamp = time.perf_counter()
            cpu = time.process_time()
            ne, exc, vxc = ni.nr_rks(mol, grids, cfg['xc'], dm, hermi=1,
                                    max_memory=cfg['memory_mb'])
            metrics = {'seconds': time.perf_counter() - stamp,
                       'cpu_seconds': time.process_time() - cpu,
                       'phases': dict(phases),
                       'max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       'thread_affinities': sorted({tuple(os.sched_getaffinity(int(t.name)))
                                                   for t in Path('/proc/self/task').iterdir()})}
            conn.send({'nelec': float(ne), 'exc': float(exc), 'vxc': vxc,
                       'metrics': metrics})
    except BaseException:
        try:
            conn.send({'error': traceback.format_exc()})
        finally:
            raise
    finally:
        conn.close()


class PartitionedXC:
    """A persistent pool for one fixed molecule/grid, real RKS, hermi=1 only.

    partitions is a list of (physical CPU ids, optional memory node). Total workspace is
    divided between workers. Grid boundaries preserve the 56-point screen bins
    and 8-point sparse-kernel alignment. Geometry/grid changes use reload().
    """

    def __init__(self, mol, grids, xc, partitions, *, memory_mb=4000,
                 block_size=None, ni_cutoff=1e-13, log_prefix='xc-worker', allowed_cpus=None):
        # Fail before starting workers or transmitting a large grid when the
        # optional dependency is missing from the calling environment.
        try:
            import threadpoolctl
        except ImportError as error:
            raise ImportError('PartitionedXC requires threadpoolctl in the active environment') from error
        from pyscf.dft.gen_grid import BLKSIZE
        if not partitions or memory_mb <= 0:
            raise ValueError('A positive memory budget and nonempty partitions are required')
        if allowed_cpus is None:
            raise ValueError('Pass the CPU allocation captured before OpenMP narrowed affinity')
        if block_size is not None and (block_size < BLKSIZE or block_size % BLKSIZE):
            raise ValueError('block_size must be a positive multiple of 56')
        cpus = [c for group, _ in partitions for c in group]
        if any(not group for group, _ in partitions):
            raise ValueError('Worker groups must not be empty')
        if len(cpus) != len(set(cpus)):
            raise ValueError('Worker CPU groups must not overlap')
        if not set(cpus) <= set(allowed_cpus):
            raise ValueError('Worker CPUs exceed the caller\'s allocation')
        self.partitions = partitions
        self.block_size = block_size
        self.processes = []
        self.connections = []
        self.logs = []
        self.ready = []
        self.metrics = []
        configs = self._configs(mol, grids, xc, memory_mb, ni_cutoff)
        start = time.perf_counter()
        try:
            for i, (group, node) in enumerate(partitions):
                parent, child = socket.socketpair()
                log_path = Path(f'{log_prefix}-{i}.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('x')
                self.logs.append(log)
                try:
                    # The coordinator is pinned to one CPU. --all permits an
                    # exec child to choose its own explicitly allocated group.
                    numa = shutil.which('numactl')
                    if numa is None:
                        if node is not None:
                            raise RuntimeError('NUMA binding unavailable; retain the native integrator')
                        cmd = [sys.executable, str(Path(__file__).resolve()), str(child.fileno()),
                               ','.join(map(str, group))]
                    else:
                        cmd = [numa, '--all', '--physcpubind=' + ','.join(map(str, group)),
                           *(['--membind=' + str(node)] if node is not None else []),
                           sys.executable, str(Path(__file__).resolve()),
                               str(child.fileno())]
                    process = subprocess.Popen(cmd, env=environment(group),
                                               pass_fds=(child.fileno(),), stdout=log,
                                               stderr=subprocess.STDOUT)
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.processes.append(process)
                self.connections.append(Connection(parent.detach()))
            for conn in self.connections:
                if not conn.poll(60):
                    raise RuntimeError('NUMA worker did not initialize within 60 seconds')
                self._receive(conn)
            for conn, cfg in zip(self.connections, configs):
                conn.send(cfg)
            for conn in self.connections:
                self.ready.append(self._receive(conn))
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.perf_counter() - start

    def _configs(self, mol, grids, xc, memory_mb, ni_cutoff):
        """Per-worker molecule and grid slices; sets nao and cuts."""
        import numpy as np
        import pyscf
        from pyscf.dft.gen_grid import BLKSIZE, ALIGNMENT_UNIT
        if mol.spin != 0:
            raise NotImplementedError('Prototype supports closed-shell RKS only')
        if grids.coords is None or grids.non0tab is None:
            raise ValueError('Build the complete grid and masks before partitioning')
        if len(grids.weights) % ALIGNMENT_UNIT:
            raise ValueError('This experiment requires sparse-kernel grid alignment')
        if memory_mb <= 0:
            raise ValueError('A positive memory budget is required')
        partitions = self.partitions
        ngrids = len(grids.weights)
        shares = np.cumsum([len(group) for group, _ in partitions])
        cuts = [0] + [(ngrids * int(share) // int(shares[-1])) // BLKSIZE * BLKSIZE
                      for share in shares[:-1]] + [ngrids]
        if any(b <= a for a, b in zip(cuts, cuts[1:])):
            raise ValueError('Too few grid points for partitions')
        configs = []
        for i, (group, node) in enumerate(partitions):
            lo, hi = cuts[i:i + 2]
            configs.append({'mol': mol.dumps(), 'pyscf_file': str(Path(pyscf.__file__).resolve()),
                            'coords': grids.coords[lo:hi],
                            'weights': grids.weights[lo:hi],
                            'mask': grids.non0tab[lo // BLKSIZE:(hi + BLKSIZE - 1) // BLKSIZE],
                            'grid_cutoff': grids.cutoff, 'ni_cutoff': ni_cutoff,
                            'xc': xc, 'cpus': group, 'memory_node': node,
                            'memory_mb': memory_mb / len(partitions),
                            'block_size': self.block_size})
        self.nao = mol.nao
        self.cuts = cuts
        return configs

    def reload(self, mol, grids, xc, *, memory_mb, ni_cutoff=1e-13):
        """Give the running workers a new molecule/grid, keeping processes,
        CPU and memory binding. Used between geometry-optimization steps."""
        if not self.connections:
            raise RuntimeError('Worker pool is closed')
        start = time.perf_counter()
        configs = self._configs(mol, grids, xc, memory_mb, ni_cutoff)
        for conn, cfg in zip(self.connections, configs):
            conn.send({'reload': cfg})
        for conn in self.connections:
            if not conn.poll(120):
                raise RuntimeError('NUMA worker did not reload within 120 seconds')
            self._receive(conn)
        self.reload_seconds = time.perf_counter() - start
        return self.reload_seconds

    @staticmethod
    def _receive(conn):
        reply = conn.recv()
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def evaluate(self, dm):
        import numpy as np
        density = np.asarray(dm)
        if density.dtype != np.float64 or density.shape != (self.nao, self.nao):
            raise ValueError('Expected a real float64 single RKS density matrix')
        start = time.perf_counter()
        payload = {'density': density,
                   'mo_coeff': getattr(dm, 'mo_coeff', None),
                   'mo_occ': getattr(dm, 'mo_occ', None)}
        for conn in self.connections:
            conn.send(payload)
        results = [self._receive(conn) for conn in self.connections]
        nelec = sum(r['nelec'] for r in results)
        exc = sum(r['exc'] for r in results)
        vxc = results[0]['vxc']
        for r in results[1:]:
            vxc += r['vxc']
        elapsed = time.perf_counter() - start
        metrics = {'seconds': elapsed, 'workers': [r['metrics'] for r in results],
                   'max_worker_seconds': max(r['metrics']['seconds'] for r in results),
                   'sum_worker_peak_rss_mib': sum(r['metrics']['max_rss_mib'] for r in results)}
        # Includes sending density/MOs, returning Vxc, reduction and scheduling.
        metrics['coordination_seconds'] = elapsed - metrics['max_worker_seconds']
        self.metrics.append(metrics)
        return nelec, exc, vxc

    def close(self):
        for conn in self.connections:
            try:
                conn.send(None)
            except (OSError, EOFError):
                pass
        for conn in self.connections:
            conn.close()
        for process in self.processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in self.logs:
            log.close()
        self.connections = []
        self.processes = []
        self.logs = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == '__main__':
    if len(sys.argv) == 3:
        # NUMA metadata/numactl may be absent on UMA servers. Set affinity before
        # importing NumPy or any OpenMP library, using the validated allocation.
        os.sched_setaffinity(0, {int(c) for c in sys.argv[2].split(',')})
    worker(int(sys.argv[1]))
