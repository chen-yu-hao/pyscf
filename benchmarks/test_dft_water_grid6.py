"""Harness-only checks; these do not substitute for an actual SCF run."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
spec = importlib.util.spec_from_file_location('water_bench', Path(__file__).with_name('dft_water_grid6.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class HarnessTests(unittest.TestCase):
    def test_alternation_fixed_reference_and_separate_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            def launch(cfg, timeout):
                calls.append(cfg)
                array_path = str(Path(cfg['result']).with_suffix('.npz'))
                np.savez(array_path, dm=np.eye(2), vxc=np.eye(2), ne=10., exc=-1.)
                wall = 10. if cfg['variant']=='baseline' else 5.
                if cfg['profile']:
                    wall = 1000.
                return dict(variant=cfg['variant'], n_water=cfg['n'], xc=cfg['xc'],
                            threads=cfg['threads'], scf_seconds=wall, profiled=cfg['profile'],
                            converged=True, grid_sha256='same', energy_hartree=-76., arrays=array_path)
            argv = ['bench', '--baseline-root', temp, '--candidate-root', temp,
                    '--sizes', '1', '--threads', '1', '--xc', 'PBE', '--repeat', '3',
                    '--profile', '--output', temp]
            with mock.patch.object(sys, 'argv', argv), mock.patch.object(bench, 'launch', launch):
                bench.main()
            report = json.loads((Path(temp)/'summary.json').read_text())
            self.assertEqual(report['errors'], [])
            self.assertEqual(report['cases'][0]['speedup'], 2.)
            self.assertEqual([c['variant'] for c in calls],
                             ['baseline', 'candidate', 'candidate', 'baseline',
                              'baseline', 'candidate', 'candidate', 'baseline'])
            self.assertIsNone(calls[0]['reference'])
            self.assertEqual(len({c['reference'] for c in calls[1:]}), 1)
            self.assertEqual(sum(c['profile'] for c in calls), 2)

    def test_verification_rejects_bad_grid_convergence_and_energy(self):
        good = dict(converged=True, grid_sha256='same', energy_hartree=-76.)
        for changes in (dict(converged=False), dict(grid_sha256='different'),
                        dict(energy_hartree=-75.)):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                bench.verify(good, dict(good, **changes))


if __name__ == '__main__':
    unittest.main()
