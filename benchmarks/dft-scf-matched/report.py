"""Render the accepted matched-protocol full-SCF comparison."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {'legacy-original': 'Legacy original 2.14.0', 'upstream': 'Upstream 44313dfb4',
          'combined': 'Three changes a8cbddc87', 'full-optimized': 'Full AO64 snapshot'}


def main():
    accepted = json.loads((ROOT / 'acceptance.json').read_text())
    assert accepted['accepted'] and accepted['accepted_runs'] == 36 and accepted['accepted_pairs'] == 54
    rows = []
    for system in accepted['summaries']:
        for label, metrics in system['versions'].items():
            row = {k: system[k] for k in ('case', 'method', 'basis', 'nao')}
            row.update(version=label, grid_points=metrics['grid_points'], cycles=metrics['cycles'],
                       polish_cycles=metrics['polish_cycles'])
            for field in ('process_wall_s', 'kernel_wall_s', 'peak_rss_mib'):
                row[field + '_median'] = metrics[field]['median']
                row[field + '_range'] = metrics[field]['range']
            rows.append(row)
    with (ROOT / 'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    for system in accepted['summaries']:
        row = {k: system[k] for k in ('case', 'method', 'basis', 'nao')}
        for ratio, boundaries in system['ratios'].items():
            for boundary, value in boundaries.items():
                row[ratio + ':' + boundary] = value
        comparisons.append(row)
    with (ROOT / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparisons)
    lines = ['# Matched-protocol complete SCF comparison', '',
             '**All 36 fresh-process runs and all 54 cross-version numerical pairs passed.** '
             'These are newly measured full-SCF results using the protocol of the historical three-system table.', '',
             '| Version | Identity |', '|---|---|',
             '| Legacy original | Frozen PySCF 2.14.0 runtime used by the historical benchmark |',
             '| Current upstream | `44313dfb457b93358918c391fc66285fa44153ef` |',
             '| Three changes | `DFT-combined`, `a8cbddc874d7846dedfdc482944fc299d76f2409` |',
             '| Full optimized | Frozen `runtime-ao64-recovered`: round3 plus AO_TILE=64 and stable grid sorting |', '',
             'The four versions use the same interpreter and input script. `pyscf_case.py` is byte-identical '
             'to the historical worker. The two current source checkouts share a baseline and matching build '
             'configuration; their ratio isolates the three selected changes. The two legacy packages reproduce '
             'the historical software comparison. Package/build and baseline differences also contribute when '
             'comparing a current checkout with a legacy package.', '',
             '## Identical scientific protocol', '',
             'MN15, fixed Bohr geometries and spherical basis functions, 99x590 atom grids without pruning '
             'or density-based grid removal; AO cutoff 1e-10 and numerical-integration cutoff 1e-8. '
             'Conventional direct J/K, no density fitting, direct_scf_tol=1e-13, no symmetry, Hückel guess, '
             '2000 MB memory setting and DIIS space 12. Strict final convergence checks density RMS <1e-9, '
             'maximum density change <1e-7, energy change <1e-7 Ha and orbital gradient <3e-5, including '
             'the extra cycle. One additional polishing SCF is included in time if the first strict check fails. '
             'Natural DM/MO dispatch is retained.', '',
             'The test runs locally on the shared Xeon Platinum 8176 host, serially, using physical CPUs 5–12 '
             'with OMP8/BLAS1. CPU4 and sibling60 were avoided after the utilization check. Three independent '
             'processes per case/version; version order rotates with case and replicate. OpenMP passive wait '
             'and NUMPY_MADVISE_HUGEPAGE=0 match the historical environment. No performance result was '
             'copied from an earlier experiment.', '',
             '## Whole-process timing', '',
             'Primary timing is parent subprocess start-to-exit, including imports, complete SCF, normal output '
             'and output checks. The statistic is the median of three processes; brackets are observed min/max '
             'ranges, not confidence intervals. Speed ratios are baseline/candidate, with values below 1 '
             'indicating a slowdown.', '',
             '| System | Legacy original (s) | Upstream (s) | Three changes (s) | Full optimized (s) | Upstream/three | Legacy/full |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for system in accepted['summaries']:
        cells = [system['case'] + ' ' + system['method']]
        for label in LABELS:
            m = system['versions'][label]['process_wall_s']
            cells.append(f"{m['median']:.4f} [{m['range'][0]:.4f}–{m['range'][1]:.4f}]")
        cells += [f"{system['ratios']['upstream/combined']['process_wall_s']:.3f}x",
                  f"{system['ratios']['legacy-original/full-optimized']['process_wall_s']:.3f}x"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'The historical complete-package acceleration is reproduced under a common protocol. '
              'The three selected changes recover part of the full-SCF benefit, especially for water12, '
              'while the O2 timing ranges overlap and show no demonstrated benefit. The complete package '
              'remains faster than the three-change package on these systems. This comparison measures '
              'their combined behavior; it does not assign the remaining benefit to individual excluded kernels.', '',
              '## Kernel timing, memory and iteration counts', '',
              'Kernel time covers the entire `mf.kernel()`, including grid setup, numerical integration, '
              'J/K, diagonalization and any polishing pass. It is reported separately from whole-process time. '
              'Peak RSS is the Linux process high-water mark through the SCF and result checks, not a phase-local allocation.', '',
              '| System | Version | Kernel median [min/max] (s) | Peak RSS median [min/max] (MiB) | Cycles | Polishing cycles | Grid points |',
              '|---|---|---:|---:|---|---|---:|']
    for system in accepted['summaries']:
        for label, m in system['versions'].items():
            k, rss = m['kernel_wall_s'], m['peak_rss_mib']
            lines.append(f"| {system['case']} | {LABELS[label]} | {k['median']:.4f} [{k['range'][0]:.4f}/{k['range'][1]:.4f}] | "
                         f"{rss['median']:.1f} [{rss['range'][0]:.1f}/{rss['range'][1]:.1f}] | "
                         f"{m['cycles']} | {m['polish_cycles']} | {m['grid_points']} |")
    e = accepted['max_errors']
    lines += ['', '## Scientific and runtime acceptance', '',
              'Every pair of versions is checked at each replicate, giving 54 comparisons. All versions '
              'have exactly matching ordered grid coordinate/weight digests and actual point counts: '
              'water4 700920, O2 116824 and water12 2102760.', '',
              f"Largest cross-version energy difference: **{e['energy_delta_hartree']:.3e} Ha**; "
              f"DM RMS: **{e['density_rms']:.3e}**; maximum DM element: **{e['density_max']:.3e}**; "
              f"UKS spin-square difference: **{e['spin_square_delta']:.3e}**. "
              'The respective gates are 1e-8 Ha, 1e-9, 1e-7 and 1e-8. Electron counts, strict final '
              'convergence and finite FP64 density matrices are checked for every run. Repeat energy '
              'ranges must remain below 1e-9 Ha.', '',
              'Source identities and native library hashes are frozen in versions.json and checked again '
              'after the campaign. Every worker records the imported package, mapped native libraries '
              'and effective OpenMP/BLAS thread counts. The two current checkouts must retain their '
              'recorded clean git commits. CPU accounting and SMT-sibling utilization are retained '
              'per run; affinity on a shared host does not establish exclusive execution.', '',
              '## Reproduction and provenance', '',
              'Use the source/runtime paths in versions.json with matching dependencies, choose eight '
              'currently idle physical cores, and update the CPU mask and sibling list in a fresh copy '
              'of protocol.json if necessary. Run `python run.py`, `python accept.py`, then `python report.py` '
              'with `/home/lyl/soft/conda-envs/pyscf-new/bin/python` or the equivalent shared interpreter. '
              'The driver requires a fresh runs/ namespace and a writable tmp/ directory. The input worker '
              'uses protocol.json from its own directory. Binary density matrices and temporary files '
              'are retained locally under `/home/lyl/soft/pyscf-scf-matched-20261027` until acceptance; '
              'temporary scratch is removed only after all runs are accepted.', '',
              'measurements.csv contains every version/system timing range, memory and cycle count; '
              'comparison.csv contains all reported ratios. acceptance.json includes all numerical pairs. '
              'runs/ contains individual results and the execution ledger; provenance/ retains the '
              'earlier build identities. Full density arrays remain in the local runs/ directory and '
              'are regenerated by the worker.', '',
              'OpenAI Codex substantially assisted benchmark setup, automation, validation and reporting. '
              'No upstream pull request was created.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print('Wrote REPORT.md, measurements.csv and comparison.csv')


if __name__ == '__main__':
    main()
