import tempfile
import unittest
from pathlib import Path

from topology import Core, Topology, choose_measured, discover, parse_list, plans, local_grid_partitions


def server(sockets, per_socket, nodes_per_socket=1, memoryless=False):
    cores, memory, distances = [], {}, {}
    nodes = sockets * nodes_per_socket
    for node in range(nodes):
        memory[node] = 0 if memoryless and node % nodes_per_socket != 1 else 32 * 1024 ** 3
        distances[node] = {other: (10 if node == other else 16 if node // nodes_per_socket == other // nodes_per_socket else 32)
                           for other in range(nodes)}
    for socket in range(sockets):
        for core in range(per_socket):
            node = socket * nodes_per_socket + core // (per_socket // nodes_per_socket)
            # Deliberately nonconsecutive CPU identifiers.
            cores.append(Core(3 * (socket * per_socket + core) + 7, socket, core, node))
    return Topology(cores, memory, distances, [c.cpu for c in cores], list(memory))


class TopologyTests(unittest.TestCase):
    def assert_plan(self, topology, plan):
        allocated = set(plan['allocation'])
        self.assertEqual(len(allocated), plan['budget'])
        self.assertTrue(allocated <= set(topology.allowed_cpus))
        self.assertFalse(set(plan['xc']) & set(plan['jk']))
        self.assertEqual(set(plan['xc']) | set(plan['jk']), allocated)
        for key in ('xc_memory', 'jk_memory', 'baseline_memory'):
            node = plan[key]
            if node is not None:
                self.assertGreater(topology.memory_bytes[node], 0)
                self.assertIn(node, topology.allowed_memory_nodes)

    def test_single_uma_socket(self):
        topo = server(1, 32)
        for budget in (8, 16, 24, 32):
            self.assert_plan(topo, plans(topo, budget, 'single'))

    def test_two_uniform_sockets(self):
        topo = server(2, 32)
        for budget in (8, 16, 24, 48):
            self.assert_plan(topo, plans(topo, budget, 'dual'))

    def test_many_numa_nodes_with_memory(self):
        topo = server(2, 64, 4)
        self.assert_plan(topo, plans(topo, 24, 'dual', preferred_socket=1))

    def test_memoryless_nodes(self):
        topo = server(2, 32, 4, memoryless=True)
        plan = plans(topo, 24, 'dual', preferred_socket=1)
        self.assert_plan(topo, plan)
        self.assertEqual(plan['baseline_memory'], 5)
        self.assertEqual(plan['xc_memory'], 1)

    def test_restricted_memory_allocation(self):
        topo = server(2, 32)
        topo.allowed_memory_nodes = [0]
        self.assert_plan(topo, plans(topo, 24, 'dual'))

    def test_missing_numa_metadata(self):
        topo = server(1, 16)
        topo.memory_bytes = {}
        topo.allowed_memory_nodes = []
        self.assertIsNone(plans(topo, 8, 'single')['xc_memory'])

    def test_budget_and_single_socket_limits(self):
        topo = server(2, 16)
        for budget, layout in ((1, 'single'), (33, 'dual'), (24, 'single')):
            with self.assertRaises(ValueError):
                plans(topo, budget, layout)

    def test_discovery_respects_affinity_and_smt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'cpu').mkdir()
            (root / 'cpu/online').write_text('0-7')
            for cpu in range(8):
                directory = root / f'cpu/cpu{cpu}/topology'
                directory.mkdir(parents=True)
                (directory / 'core_id').write_text(str(cpu % 4))
                (directory / 'physical_package_id').write_text('0')
            node = root / 'node/node0'
            node.mkdir(parents=True)
            (node / 'cpulist').write_text('0-7')
            (node / 'distance').write_text('10')
            (node / 'meminfo').write_text('Node 0 MemTotal:  32000000 kB\n')
            status = root / 'status'
            status.write_text('Mems_allowed_list:\t0\n')
            topo = discover(root, status, affinity={2, 3, 4, 6, 7, 8})
            self.assertEqual([c.cpu for c in topo.cores], [2, 3, 4])
            self.assertEqual(topo.allowed_cpus, [2, 3, 4, 6, 7])
            self.assertEqual(parse_list('0-2,7,10-11'), {0, 1, 2, 7, 10, 11})

    def test_uma_keeps_native_integrator(self):
        topo = server(1, 32)
        self.assertEqual(local_grid_partitions(topo, topo.allowed_cpus[:24]), [])

    def test_partitions_omit_memoryless_remote_cores(self):
        topo = server(2, 32, 4, memoryless=True)
        plan = plans(topo, 24, 'dual', preferred_socket=1)
        partitions = local_grid_partitions(topo, plan['allocation'])
        self.assertEqual(sorted(len(cpus) for cpus, _ in partitions), [8, 8])
        self.assertEqual(sorted(node for _, node in partitions), [1, 5])

    def test_unequal_groups_and_restricted_allocation(self):
        topo = server(2, 16)
        allocation = topo.allowed_cpus[:3] + topo.allowed_cpus[16:21]
        groups = local_grid_partitions(topo, allocation)
        self.assertEqual([len(cpus) for cpus, _ in groups], [3, 5])
        with self.assertRaises(ValueError):
            local_grid_partitions(topo, [max(topo.allowed_cpus) + 100])


class SelectionTests(unittest.TestCase):
    def choose(self, baseline, candidate, **kw):
        return choose_measured(baseline, candidate, setup_seconds=kw.get('setup', 1),
                               expected_calls=kw.get('calls', 10), numerical_ok=kw.get('ok', True))

    def test_clear_amortized_gain(self):
        self.assertTrue(self.choose([20, 21, 20], [14, 15, 14]))

    def test_noise_or_one_regression_rejects(self):
        self.assertFalse(self.choose([20, 21, 20], [19, 19, 19]))
        self.assertFalse(self.choose([20, 21, 20], [14, 22, 14]))

    def test_setup_or_accuracy_rejects(self):
        self.assertFalse(self.choose([20, 21, 20], [14, 15, 14], setup=1000))
        self.assertFalse(self.choose([20, 21, 20], [14, 15, 14], ok=False))

    def test_invalid_measurements_reject(self):
        for baseline, candidate in (([20], [14]), ([20, 20, 20], [14, 0, 14]),
                                    ([20, 20, 20], [14, float('nan'), 14])):
            self.assertFalse(self.choose(baseline, candidate))


if __name__ == '__main__':
    unittest.main(verbosity=2)
