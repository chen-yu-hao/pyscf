"""Summarize accepted full-SCF timings of the actual GitHub DFT branch."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    accepted = json.loads((ROOT / 'acceptance.json').read_text())
    assert accepted['accepted'] and accepted['accepted_runs'] == 18 and accepted['accepted_pairs'] == 9
    rows = []
    comparisons = []
    for system in accepted['summaries']:
        for label, metrics in system['versions'].items():
            row = {k: system[k] for k in ('case', 'method', 'basis', 'nao')}
            row.update(version=label, grid_points=metrics['grid_points'], cycles=metrics['cycles'],
                       polish_cycles=metrics['polish_cycles'])
            for field in ('process_wall_s', 'kernel_wall_s', 'peak_rss_mib'):
                row[field + '_median'] = metrics[field]['median']
                row[field + '_range'] = metrics[field]['range']
            rows.append(row)
        comparisons.append({k: system[k] for k in ('case', 'method', 'basis', 'nao')} |
                           system['ratios']['upstream/DFT'])
    for name, data in [('measurements.csv', rows), ('comparison.csv', comparisons)]:
        with (ROOT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    lines = ['# Complete SCF benchmark of the GitHub DFT branch', '',
             '**18/18 fresh-process runs and 9/9 cross-version numerical pairs passed.** '
             'All timings were measured anew in this campaign.', '',
             '- Candidate: [DFT](https://github.com/chen-yu-hao/pyscf/tree/DFT), '
             '`72e5b26c91964acbf3186c5348f77a0f7faf44dd`.',
             '- Upstream baseline: `44313dfb457b93358918c391fc66285fa44153ef`.', '',
             'This measures the GitHub branch itself. The earlier AO64 filesystem snapshot is a distinct '
             'implementation and is not the candidate in this table. Both checkouts have clean tracked '
             'sources and verified native builds. Compiler GCC 11.3.1, CMake 4.3.1, LibXC 7.0.0, '
             'standard Release `-O3 -DNDEBUG -msse3 -fopenmp`, without fast-math.', '',
             '## Common protocol', '',
             'MN15, the same fixed Bohr geometries and spherical cc-pVDZ water-cluster / cc-pVTZ O2 bases '
             'as the historical full-SCF table. Each atom requests 99x590 points, no pruning and no '
             'density-based point removal; grid cutoff 1e-10, numerical-integration cutoff 1e-8. '
             'Conventional direct J/K with direct_scf_tol=1e-13, no density fitting, no symmetry, '
             'Hückel guess, 2000 MB memory setting and DIIS space 12. The input worker is byte-identical '
             'to the historical worker.', '',
             'Strict final convergence checks density RMS <1e-9, maximum density change <1e-7, energy '
             'change <1e-7 Ha and orbital gradient <3e-5, including the extra cycle. One additional '
             'polishing SCF is included in time if the first strict check fails. Normal DM/MO dispatch '
             'is retained.', '',
             'Local shared Xeon Platinum 8176, physical CPUs 1,3,5,7,9,12,13,22 on socket 0, OMP8/BLAS1. '
             'The second socket was heavily occupied; selected cores and their SMT siblings had low '
             'sampled utilization. Both versions run serially on the same mask, alternating order '
             'across cases and replicates. OpenMP passive wait and NUMPY_MADVISE_HUGEPAGE=0.', '',
             '## Complete-process timings', '',
             'Primary timing is parent subprocess start-to-exit, including imports, the entire SCF, '
             'normal output and result checks. Each value is the median of three independent processes; '
             'brackets show observed min/max ranges, not confidence intervals. Speedup is upstream/DFT.', '',
             '| System | Upstream (s) | DFT branch (s) | Speedup |',
             '|---|---:|---:|---:|']
    for s in accepted['summaries']:
        a, b = [s['versions'][v]['process_wall_s'] for v in ('upstream', 'DFT')]
        lines.append(f"| {s['case']} {s['method']} | {a['median']:.4f} [{a['range'][0]:.4f}–{a['range'][1]:.4f}] | "
                     f"{b['median']:.4f} [{b['range'][0]:.4f}–{b['range'][1]:.4f}] | "
                     f"{s['ratios']['upstream/DFT']['process_wall_s']:.3f}x |")
    lines += ['', '## Kernel time and memory', '',
              '`mf.kernel()` includes grid setup, numerical integration, J/K, diagonalization and any '
              'polishing pass. Its time is kept separate from process time. Peak RSS is the Linux '
              'process high-water mark through SCF and checks, not a phase-local allocation.', '',
              '| System | Version | Kernel median [min/max] (s) | Peak RSS median [min/max] (MiB) | Cycles | Polishing cycles | Points |',
              '|---|---|---:|---:|---|---|---:|']
    for s in accepted['summaries']:
        for label, m in s['versions'].items():
            k, rss = m['kernel_wall_s'], m['peak_rss_mib']
            lines.append(f"| {s['case']} | {label} | {k['median']:.4f} [{k['range'][0]:.4f}/{k['range'][1]:.4f}] | "
                         f"{rss['median']:.1f} [{rss['range'][0]:.1f}/{rss['range'][1]:.1f}] | "
                         f"{m['cycles']} | {m['polish_cycles']} | {m['grid_points']} |")
    e = accepted['max_errors']
    lines += ['', '## Acceptance and scope', '',
              'All baseline/candidate pairs have exactly matching ordered grid coordinates, weights '
              'and point counts: water4 700920, O2 116824, water12 2102760. Every run passes strict '
              'convergence, finite FP64 DM, overlap electron count and OMP8/BLAS1 checks.', '',
              f"Maximum energy difference: {e['energy_delta_hartree']:.3e} Ha; "
              f"DM RMS: {e['density_rms']:.3e}; maximum DM element: {e['density_max']:.3e}; "
              f"spin-square difference: {e['spin_square_delta']:.3e}. "
              'Gates are 1e-8 Ha, 1e-9, 1e-7 and 1e-8 respectively. Repeat energy ranges must be '
              'below 1e-9 Ha.', '',
              'Source and native identities are frozen before measurement and checked again after '
              'all runs; mapped libraries must originate from the selected checkout. CPU and sibling '
              'counters are retained per run. This is a shared host: three repeats and core affinity '
              'do not establish exclusive execution or a population confidence interval. These results '
              'cover the stated three MN15 workloads and eight-thread setting.', '',
              '## Artifacts and reproduction', '',
              'Use versions.json for exact source commits, source/native hashes and runtime paths. '
              'provenance/ records original build manifests and the aggregate build configuration. '
              'Use the same interpreter/dependencies for both builds. Copy the benchmark directory to '
              'a fresh namespace, choose eight idle physical cores, and update cpu_mask and '
              'smt_siblings_monitored in protocol.json if needed. Create tmp/ and run `python run.py`, '
              '`python accept.py`, then `python report.py`. The driver requires a fresh runs/ directory. '
              'finish.py is the original serial acceptance/cleanup controller.', '',
              'measurements.csv contains all medians, ranges, memory and cycles; comparison.csv '
              'contains process and kernel ratios. acceptance.json retains every numerical pair. '
              'Raw density arrays and worker logs remain locally in '
              '`/home/lyl/soft/pyscf-dft-scf-20261027/runs`; the public text archive omits binaries. '
              'Registered temporary scratch is removed only after acceptance.', '',
              'OpenAI Codex substantially assisted benchmark setup, validation and reporting. '
              'No upstream pull request was created.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print('Wrote REPORT.md, measurements.csv and comparison.csv')


if __name__ == '__main__':
    main()
