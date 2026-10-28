"""Topology discovery and bounded plans, before importing any OpenMP runtime.

No CPU ids, NUMA ids, socket counts or vendor-specific core sizes are built in.
An allocation is the launching process's affinity, intersected with online CPUs.
SMT siblings count as one physical core. A node without local memory uses the
nearest allowed memory-bearing node; NUMA absence is a valid single-node case.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import os


def parse_list(value):
    result = set()
    for piece in value.strip().split(','):
        if not piece:
            continue
        bounds = piece.split('-')
        lo, hi = int(bounds[0]), int(bounds[-1])
        if lo > hi:
            raise ValueError(value)
        result.update(range(lo, hi + 1))
    return result


@dataclass(frozen=True)
class Core:
    cpu: int
    socket: int
    core: int
    node: int


@dataclass
class Topology:
    cores: list
    memory_bytes: dict
    distances: dict
    allowed_cpus: list
    allowed_memory_nodes: list

    def nearest_memory(self, node):
        candidates = [n for n, size in self.memory_bytes.items()
                      if size > 0 and n in self.allowed_memory_nodes]
        if not candidates:
            return None
        return min(candidates, key=lambda n: (self.distances.get(node, {}).get(n, 10 if n == node else 20), n))

    def ordered(self, socket=None):
        cores = [c for c in self.cores if socket is None or c.socket == socket]
        return sorted(cores, key=lambda c: (
            c.socket, self.distances.get(c.node, {}).get(self.nearest_memory(c.node), 10),
            c.node, c.core, c.cpu))

    def to_dict(self):
        return asdict(self)


def discover(sys_root=Path('/sys/devices/system'), status_path=Path('/proc/self/status'), affinity=None):
    allowed = set(os.sched_getaffinity(0) if affinity is None else affinity)
    online_path = sys_root / 'cpu/online'
    if online_path.exists():
        allowed &= parse_list(online_path.read_text())
    mem_nodes = None
    for line in status_path.read_text().splitlines():
        if line.startswith('Mems_allowed_list:'):
            mem_nodes = parse_list(line.split(':', 1)[1])
    memory, distances, cpu_nodes = {}, {}, {}
    for node_dir in sorted((sys_root / 'node').glob('node[0-9]*')):
        node = int(node_dir.name[4:])
        for cpu in parse_list((node_dir / 'cpulist').read_text()):
            cpu_nodes[cpu] = node
        for line in (node_dir / 'meminfo').read_text().splitlines():
            if 'MemTotal:' in line:
                memory[node] = int(line.split('MemTotal:')[1].split()[0]) * 1024
        distance_file = node_dir / 'distance'
        distances[node] = {i: int(value) for i, value in enumerate(distance_file.read_text().split())} if distance_file.exists() else {}
    if not memory:
        # No NUMA files: permit unbound local allocation rather than inventing a
        # node for numactl. CPU planning still works without libnuma/numactl.
        memory, distances = {}, {}
    seen, cores = set(), []
    for cpu in sorted(allowed):
        directory = sys_root / f'cpu/cpu{cpu}/topology'
        socket_path, core_path = directory / 'physical_package_id', directory / 'core_id'
        socket_id = int(socket_path.read_text()) if socket_path.exists() else 0
        core_id = int(core_path.read_text()) if core_path.exists() else cpu
        key = (socket_id, core_id)
        if key not in seen:
            cores.append(Core(cpu, socket_id, core_id, cpu_nodes.get(cpu, 0)))
            seen.add(key)
    return Topology(cores, memory, distances, sorted(allowed),
                    sorted(mem_nodes if mem_nodes is not None else memory))


def plans(topology, budget, layout, preferred_socket=None):
    if budget < 2 or budget > len(topology.cores):
        raise ValueError('Physical-core budget outside allocation')
    sockets = sorted({c.socket for c in topology.cores})
    socket_id = sockets[0] if preferred_socket is None else preferred_socket
    primary = topology.ordered(socket_id)
    if layout == 'single':
        if len(primary) < budget:
            raise ValueError('Requested single-socket allocation does not fit')
        allocation = primary[:budget]
    elif layout == 'dual':
        others = [c for c in topology.ordered() if c.socket != socket_id]
        share = max(1, budget // 3)
        if len(primary) < budget - share or len(others) < share:
            raise ValueError('Requested two-socket allocation does not fit')
        allocation = primary[:budget - share] + others[:share]
    else:
        raise ValueError(layout)
    # This fraction proposes a measured candidate, never an assumed optimum.
    xc_count = max(1, budget // 3)
    if layout == 'dual':
        xc = allocation[-xc_count:]
        jk = allocation[:-xc_count]
    else:
        # XC gets memory-local cores; J/K gets the remaining disjoint cores.
        xc, jk = allocation[:xc_count], allocation[xc_count:]
    baseline_xc = allocation[:xc_count]
    return {'layout': layout, 'budget': budget,
            'allocation': [c.cpu for c in allocation],
            'baseline_xc': [c.cpu for c in baseline_xc],
            'baseline_memory': topology.nearest_memory(baseline_xc[0].node),
            'xc': [c.cpu for c in xc], 'xc_memory': topology.nearest_memory(xc[0].node),
            'jk': [c.cpu for c in jk], 'jk_memory': topology.nearest_memory(jk[0].node),
            'notes': 'Candidate split requires timing against the serial baseline; not a universal thread recommendation'}


def choose_measured(baseline, candidate, *, setup_seconds, expected_calls,
                    numerical_ok, relative_margin=.10):
    """Conservative opt-in gate; all paired samples must clear the margin.

    Account for startup, require >=3 finite timing pairs, and refuse tuning when
    the numerical check failed. This gate makes no claim about unseen hardware.
    """
    import math
    import statistics
    if (not numerical_ok or len(baseline) != len(candidate) or len(baseline) < 3
            or expected_calls <= 0 or not math.isfinite(setup_seconds) or setup_seconds < 0
            or not 0 <= relative_margin < 1):
        return False
    if any(not math.isfinite(v) or v <= 0 for v in baseline + candidate):
        return False
    if any(c > b * (1 - relative_margin) for b, c in zip(baseline, candidate)):
        return False
    saving = statistics.median(baseline) - statistics.median(candidate)
    return expected_calls * saving > setup_seconds


def local_grid_partitions(topology, allocation):
    """Candidate groups with local memory; return [] on UMA/unsupported layouts.

    Memoryless remote cores remain available for J/K. They are not silently
    treated as extra memory-local XC capacity. Unequal groups receive grid
    shares proportional to their core counts in PartitionedXC.
    """
    allowed = set(allocation)
    if not allowed <= set(topology.allowed_cpus):
        raise ValueError('Allocation exceeds available CPUs')
    groups = {}
    for core in topology.cores:
        if core.cpu not in allowed:
            continue
        node = topology.nearest_memory(core.node)
        if node is not None and core.node == node:
            groups.setdefault(node, []).append(core.cpu)
    if len(groups) < 2:
        return []
    return [(sorted(cpus), node) for node, cpus in sorted(groups.items())]
