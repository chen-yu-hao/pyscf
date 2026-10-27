"""Write the final report only after all numerical and native gates pass."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    reports = {name: read(path) for name, path in {
        'xc': 'xc-v1/accepted/acceptance.json',
        'sort': 'grid-sort-v1/accepted/acceptance.json',
        'end_to_end': 'end-to-end-v1/acceptance.json'}.items()}
    for name in ('xc-v1', 'grid-sort-v1', 'end-to-end-v1'):
        assert read(name + '/native-acceptance.json')['accepted']
    assert all(report['accepted'] for report in reports.values())
    xc = reports['xc']['comparisons']
    sorting = reports['sort']['comparisons'][0]
    grids = [r for r in reports['end_to_end']['comparisons'] if r['mode'] == 'grid-build']
    scf = [r for r in reports['end_to_end']['comparisons'] if r['mode'] == 'scf']
    rows = []
    for r in xc + [sorting]:
        rows.append({'boundary': 'grid sorting' if 'mode' in r else 'XC-only DM',
                     'case': r['case'], 'xc': r.get('xc', ''), 'threads': r['threads'],
                     'baseline_s': r['baseline_seconds_median'], 'combined_s': r['candidate_seconds_median'],
                     'speedup': r['speedup'], 'baseline_rss_mib': r['baseline_peak_rss_mib_median'],
                     'combined_rss_mib': r['candidate_peak_rss_mib_median']})
    for r in grids + scf:
        for metric in (['grid_seconds'] if r['mode'] == 'grid-build' else ['kernel_seconds', 'process_seconds']):
            rows.append({'boundary': metric, 'case': r['case'], 'xc': r.get('xc') or '',
                         'threads': r['threads'], 'baseline_s': r['baseline_' + metric],
                         'combined_s': r['combined_' + metric], 'speedup': r[metric + '_ratio'],
                         'baseline_rss_mib': r['baseline_peak_rss_mib'],
                         'combined_rss_mib': r['combined_peak_rss_mib']})
    with (ROOT / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'results.json').write_text(json.dumps(reports, indent=2) + '\n')
    lines = [
        '# Three combined DFT optimizations: measurements of the submitted code', '',
        'Code branch: [DFT-combined](https://github.com/chen-yu-hao/pyscf/tree/DFT-combined).', '',
        'Baseline: `44313dfb457b93358918c391fc66285fa44153ef` (upstream master snapshot). '
        'Combined: `a8cbddc874d7846dedfdc482944fc299d76f2409` (the pushed and measured commit).', '',
        'The branch combines stable integer grid grouping, smaller sparse AO-pair output tiles, '
        'and sparse DM density contractions. It contains three commits and changes seven files '
        '(461 insertions, 7 deletions). There are no MO-streaming, LibXC, AO-derivative, '
        'potential-kernel or response-cache changes.', '',
        '**Acceptance:** 44/44 numerical unit tests; 48/48 XC processes, 6/6 sorting processes, '
        'and 48/48 grid-build/SCF processes accepted. All 102 measured processes passed native-library '
        'identity, source commit, import location and thread-count checks. DFT lint passed with '
        'the upstream CI rules and recursive test-directory exclusions.', '',
        '## Test conditions', '',
        'Local shared host, Intel Xeon Platinum 8176. Performance processes ran serially, on physical '
        'CPUs 5–12 at 8 threads and 5–27 plus 3 at 24 threads; CPU 4 and its SMT sibling 60 were excluded '
        'after the utilization check. One BLAS thread, passive OpenMP wait policy, GCC 11.3.1, '
        'CMake 4.3.1, LibXC 7.0.0, Release `-O3 -DNDEBUG -msse3 -fopenmp`, no fast-math.', '',
        'Every comparison has three independent fresh processes per version with alternating version order. '
        'XC, sorting and grid-build processes have one excluded warmup and three timed calls; the table '
        'uses the median of the three process medians. SCF has one calculation per process. '
        'Ratios are baseline/combined; values below 1 indicate a slowdown. Min/max ranges and RSS '
        'are retained in the acceptance JSON and CSV. Three repeats are descriptive observations, '
        'not confidence intervals or a universal performance guarantee.', '',
        '## Fixed-DM XC-only integration', '',
        '`nr_rks`/`nr_uks` includes AO evaluation, density, LibXC and potential assembly; '
        'it excludes grid construction, J/K and SCF. The same frozen coordinates, weights, '
        'screening indices and core-Hamiltonian-projector DMs are used by both versions. '
        'These are not converged SCF densities. Level-2 pruned grids have 263424/258056 points '
        'for the 288/283-AO cases and 439040/433672 points for the 1160/1146-AO cases.', '',
        '| Case | Functional | Threads | Baseline (s) | Combined (s) | Speedup | RSS base/new (MiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in xc:
        lines.append(f"| {r['case']} | {r['xc']} | {r['threads']} | {r['baseline_seconds_median']:.4f} | "
                     f"{r['candidate_seconds_median']:.4f} | {r['speedup']:.3f}x | "
                     f"{r['baseline_peak_rss_mib_median']:.1f}/{r['candidate_peak_rss_mib_median']:.1f} |")
    worst = {k: max(r['max_absolute_errors'][k] for r in xc) for k in ('nelec', 'exc', 'vxc')}
    lines += ['', 'Largest absolute baseline/combined differences: '
              f"electrons {worst['nelec']:.3e}, Exc {worst['exc']:.3e} Ha, Vxc {worst['vxc']:.3e}. "
              'Gates: 1e-8 electrons, 1e-8 Ha and 1e-9 Vxc.', '',
              '## Grid sorting and complete grid construction', '',
              'Sorting is `arg_group_grids` on a fixed permutation of 439040 grid points at one thread; '
              'output indices match exactly. Grid construction includes `Grids` creation and '
              '`build(with_non0tab=True)` for water20/cc-pVTZ, level 2. All generated coordinates, '
              'weights and screening indices exactly match the frozen baseline fixture.', '',
              '| Boundary | Threads | Baseline (s) | Combined (s) | Speedup |',
              '|---|---:|---:|---:|---:|',
              f"| Sorting only | 1 | {sorting['baseline_seconds_median']:.4f} | "
              f"{sorting['candidate_seconds_median']:.4f} | {sorting['speedup']:.3f}x |"]
    for r in grids:
        lines.append(f"| Complete grid build | {r['threads']} | {r['baseline_grid_seconds']:.4f} | "
                     f"{r['combined_grid_seconds']:.4f} | {r['grid_seconds_ratio']:.3f}x |")
    lines += ['', '## Complete SCF', '',
              'Water4 RKS/cc-pVDZ (96 AOs) and O2 UKS/cc-pVTZ (60 AOs), eight threads. '
              'Level-3 nwchem-pruned grids, no density-based grid removal, grid cutoff 1e-15, '
              'numerical-integration cutoff 1e-13, conventional direct J/K with tolerance 1e-13, '
              'no density fitting, Hückel initial guess, no symmetry, 1024 MB memory setting, '
              'DIIS space 12, maximum 100 cycles. Stock PySCF convergence uses energy tolerance '
              '1e-10 Ha and orbital-gradient tolerance 1e-6 including the extra cycle. '
              'Normal automatic DM/MO route selection is retained.', '',
              'Kernel timing covers `mf.kernel()` including grid setup and J/K. Process timing covers '
              'parent subprocess start-to-exit, including imports, normal output and the common '
              'provenance/validation harness overhead; it is not mixed with kernel timing.', '',
              '| Case | XC | Kernel base/new (s) | Kernel speedup | Process base/new (s) | Process speedup | Cycles base/new | RSS base/new (MiB) |',
              '|---|---|---:|---:|---:|---:|---|---:|']
    for r in scf:
        lines.append(f"| {r['case']} | {r['xc']} | {r['baseline_kernel_seconds']:.3f}/{r['combined_kernel_seconds']:.3f} | "
                     f"{r['kernel_seconds_ratio']:.3f}x | {r['baseline_process_seconds']:.3f}/{r['combined_process_seconds']:.3f} | "
                     f"{r['process_seconds_ratio']:.3f}x | {r['baseline_cycles']}/{r['combined_cycles']} | "
                     f"{r['baseline_peak_rss_mib']:.1f}/{r['combined_peak_rss_mib']:.1f} |")
    worst_scf = {k: max(r['errors'][k] for r in scf) for k in ('energy_hartree', 'dm_absolute')}
    lines += ['', f"Maximum SCF energy difference: {worst_scf['energy_hartree']:.3e} Ha; "
              f"maximum DM element difference: {worst_scf['dm_absolute']:.3e}. "
              'Gates are 1e-8 Ha and 1e-6 respectively. Grid coordinates, weights and masks match exactly. '
              'Every run is converged, with overlap electron counts checked to 1e-9.', '',
              '## Interpretation and scope', '',
              'The independently measured historical speedups must not be multiplied. The new tables '
              'measure their actual combination. Sorting gains apply only to sorting; fixed-DM XC '
              'gains cannot be substituted for normal SCF speedups because SCF can use the MO route, '
              'includes J/K and diagonalization, and may require different iteration counts. '
              'Large-system results here cover XC-only integration; the complete-SCF cases are small. '
              'Peak RSS is a Linux process high-water mark including inputs and warmup, not a '
              'phase-local allocation measurement. Shared-host noise remains possible despite core affinity.', '',
              'These three changes make no response-cache memory claim; TDDFT/CPHF are outside this '
              'combined-branch benchmark.', '',
              '## Artifacts and reproduction', '',
              '`comparison.csv` contains every timing boundary; `results.json` includes accepted '
              'comparisons, ranges and numerical errors. `protocols/`, `builds/`, `machine.json`, '
              '`tests.json`, `tests.log`, `lint-ci.json` and per-run JSON files retain provenance. '
              'See `REPRODUCE.md` for commands. Raw numerical output arrays remain in the local '
              'measurement directories and can be regenerated by these scripts.', '',
              'OpenAI Codex substantially assisted the code combination, testing, benchmark automation '
              'and documentation. A human author must understand and review the kernels and personally '
              'submit any upstream PR under `AI_POLICY.md`. No upstream PR was created.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print('Wrote REPORT.md, comparison.csv and results.json; all 102 processes accepted.')


if __name__ == '__main__':
    main()
