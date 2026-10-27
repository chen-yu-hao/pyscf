"""Render the accepted same-batch three-version 16-thread SCF benchmark."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    a = json.loads((ROOT / 'acceptance.json').read_text())
    assert a['accepted'] and a['accepted_runs'] == 27 and a['accepted_pairs'] == 27
    rows, comparisons = [], []
    for s in a['summaries']:
        for label, m in s['versions'].items():
            row = {k: s[k] for k in ('case', 'method', 'basis', 'nao')}
            row.update(version=label, threads=16, cycles=m['cycles'], polish_cycles=m['polish_cycles'],
                       grid_points=m['grid_points'])
            for field in ('process_wall_s', 'kernel_wall_s', 'peak_rss_mib'):
                row[field + '_median'] = m[field]['median']
                row[field + '_range'] = m[field]['range']
            rows.append(row)
        row = {k: s[k] for k in ('case', 'method', 'basis', 'nao')}
        for pair, metrics in s['ratios'].items():
            row.update({pair + ':' + k: v for k, v in metrics.items()})
        comparisons.append(row)
    for name, data in [('measurements.csv', rows), ('comparison.csv', comparisons)]:
        with (ROOT / name).open('w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(data)
    lines = ['# Three-version complete SCF comparison at 16 threads', '',
             '**27/27 fresh-process runs and 27/27 pairwise numerical comparisons passed.** '
             'Every number in the primary comparison was measured in this same campaign.', '',
             '| Version | Tested commit |', '|---|---|',
             '| Upstream | `44313dfb457b93358918c391fc66285fa44153ef` |',
             '| DFT | `72e5b26c91964acbf3186c5348f77a0f7faf44dd` |',
             '| DFT-combined | `a8cbddc874d7846dedfdc482944fc299d76f2409` |', '',
             'The two candidates are the actual GitHub branches. Each clean checkout uses its own '
             'verified native libraries. GCC 11.3.1, CMake 4.3.1, LibXC 7.0.0, standard Release '
             '`-O3 -DNDEBUG -msse3 -fopenmp`, no fast-math.', '',
             '## Protocol', '',
             'MN15, fixed Bohr geometries; water4/water12 RKS with spherical cc-pVDZ and triplet O2 UKS '
             'with spherical cc-pVTZ. Each atom uses an unpruned 99x590 grid, with no density-based '
             'point removal. AO cutoff 1e-10, integration cutoff 1e-8, conventional direct J/K '
             'with direct_scf_tol=1e-13, no density fitting, no symmetry, Hückel initial guess, '
             '2000 MB memory setting and DIIS space 12.', '',
             'Strict final convergence: density RMS <1e-9, maximum density change <1e-7, energy '
             'change <1e-7 Ha and orbital gradient <3e-5, including the extra cycle. At most one '
             'additional polishing SCF is included in timing if the first strict check fails. '
             'The worker is identical to the prior 8-thread worker; its --cores argument is 16.', '',
             'Local shared Xeon Platinum 8176. Sixteen OpenMP threads bind to physical CPUs 32–47 '
             'on socket 1; SMT siblings 88–103 are monitored. BLAS remains single-threaded. '
             'Busy CPUs 14–21 on the other socket were avoided. Processes run serially with version '
             'order rotated by case and replicate. Three fresh processes per case/version; passive '
             'OpenMP wait and NUMPY_MADVISE_HUGEPAGE=0.', '',
             '## Complete-process timing', '',
             'Parent subprocess start-to-exit includes imports, the entire SCF, normal output and '
             'result checks. Values are medians of three processes; brackets are observed min/max '
             'ranges, not confidence intervals. Speedups use the same-batch upstream baseline.', '',
             '| System | Upstream (s) | DFT (s) | DFT-combined (s) | Upstream/DFT | Upstream/combined |',
             '|---|---:|---:|---:|---:|---:|']
    for s in a['summaries']:
        cells = [s['case'] + ' ' + s['method']]
        for label in ('upstream', 'DFT', 'DFT-combined'):
            m = s['versions'][label]['process_wall_s']
            cells.append(f"{m['median']:.4f} [{m['range'][0]:.4f}–{m['range'][1]:.4f}]")
        cells += [f"{s['ratios'][p]['process_wall_s']:.3f}x" for p in ('upstream/DFT', 'upstream/DFT-combined')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Kernel time, memory and iterations', '',
              '`mf.kernel()` includes grid setup, integration, J/K, diagonalization and polishing. '
              'Its timing is separate from whole-process timing. Peak RSS is the Linux process '
              'high-water mark through SCF and checks, not a phase-local allocation.', '',
              '| System | Version | Kernel median [min/max] (s) | Peak RSS median [min/max] (MiB) | Cycles | Polishing | Points |',
              '|---|---|---:|---:|---|---|---:|']
    for s in a['summaries']:
        for label, m in s['versions'].items():
            k, rss = m['kernel_wall_s'], m['peak_rss_mib']
            lines.append(f"| {s['case']} | {label} | {k['median']:.4f} [{k['range'][0]:.4f}/{k['range'][1]:.4f}] | "
                         f"{rss['median']:.1f} [{rss['range'][0]:.1f}/{rss['range'][1]:.1f}] | "
                         f"{m['cycles']} | {m['polish_cycles']} | {m['grid_points']} |")
    e = a['max_errors']
    lines += ['', '## Acceptance', '',
              'Every pair of versions is compared at each replicate. All 27 comparisons require '
              'exactly matching ordered grid coordinate/weight digests and point counts: water4 '
              '700920, O2 116824, water12 2102760. All runs pass strict convergence, finite FP64 DM, '
              'overlap electron count and effective OMP16/BLAS1 checks.', '',
              f"Maximum cross-version energy difference: {e['energy_delta_hartree']:.3e} Ha; "
              f"DM RMS: {e['density_rms']:.3e}; maximum DM element: {e['density_max']:.3e}; "
              f"spin-square difference: {e['spin_square_delta']:.3e}. "
              'The respective gates are 1e-8 Ha, 1e-9, 1e-7 and 1e-8. Repeat energy ranges must be '
              'below 1e-9 Ha. Source commits, tracked source hashes and native library identities '
              'are frozen before timing and checked again after the campaign.', '',
              'The host is shared. Core affinity and three repeats do not establish exclusive execution '
              'or population confidence intervals. All regressions remain in the tables. These results '
              'cover the three stated MN15 cases. Earlier 8-thread results used different CPU masks '
              'and batches: [DFT 8-thread report](../dft-branch-scf) and '
              '[combined 8-thread report](../dft-scf-matched). Cross-batch 8/16 time ratios are '
              'descriptive references rather than controlled parallel-scaling measurements.', '',
              '## Artifacts and reproduction', '',
              'versions.json binds all three checkouts to exact commits and binary identities; '
              'provenance/ contains build records. measurements.csv retains medians, ranges, RSS '
              'and cycles; comparison.csv contains all process/kernel ratios. acceptance.json '
              'retains every numerical pair, and runs/ contains each result and the ledger.', '',
              'Use matching builds/dependencies and the same interpreter for all versions. Copy this '
              'benchmark to a fresh directory, choose sixteen idle physical cores on one socket, '
              'update cpu_mask and smt_siblings_monitored in protocol.json, and create tmp/. Run '
              '`python run.py`, `python accept.py`, then `python report.py`. The driver requires a '
              'fresh runs/ namespace. finish.py is the serial acceptance/cleanup controller. '
              'Raw density arrays and worker stdout remain locally in '
              '`/home/lyl/soft/pyscf-scf-16t-20261027/runs`; public evidence omits these binaries '
              'and compiled libraries. Registered temporary scratch is removed after acceptance.', '',
              'OpenAI Codex substantially assisted benchmark preparation, validation and reporting. '
              'No upstream pull request was created.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print('Wrote REPORT.md, measurements.csv and comparison.csv')


if __name__ == '__main__':
    main()
