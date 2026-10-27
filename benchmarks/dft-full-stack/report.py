"""Render accepted cumulative SCF evidence and factual notes for human PR authors."""
import csv
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
POLICY = 'https://github.com/pyscf/pyscf/blob/master/AI_POLICY.md'
DESCRIPTIONS = {
 '01-grid': ('Preserve molecular grid grouping with stable integer keys',
   'Grid grouping currently performs a three-column unique operation before stable sorting. '
   'Encode the box IDs as mixed-radix int64 keys and sort them stably, preserving the exact existing index order. '
   'Check integer range before NumPy arithmetic and retain the original fallback.'),
 '02-libxc': ('Pack and evaluate each LibXC grid slice in one parallel region',
   'Repeated packing, evaluation and merging regions add synchronization to LibXC calls. '
   'Give each thread ownership of one grid slice through packing, component evaluation and output assembly. '
   'Retain the functional-component accumulation order at each point, including empty thread slices.'),
 '03-ao': ('Reuse primitive exponentials in first AO derivatives',
   'First AO derivatives repeatedly load primitive exponential factors during contraction. '
   'Reuse a short primitive-local grid buffer while retaining primitive accumulation order, '
   'the exponential function, screening and first-derivative semantics.'),
 '04-density-dm': ('Fuse screened density-matrix contractions for numerical integration',
   'Screened density evaluation creates repeated AO-by-DM intermediates. '
   'Compute density, gradients and tau in local screened blocks, reusing the relevant AO components and masks. '
   'Keep unsupported dtype, layout, Laplacian and nonsymmetric-gradient cases on their existing paths. '
   'Bound the dispatch mask to the current grid block rather than its unprocessed tail.'),
 '05-density-mo': ('Stream MO density, gradient and tau contractions',
   'Normal SCF can evaluate density from occupied MOs rather than a bare density matrix. '
   'Form orbitals in short grid blocks and immediately contract density, gradient and tau. '
   'Compress AO columns only when their value and all first derivatives are exactly zero; '
   'retain signed occupations, nonfinite propagation, resource checks and existing fallbacks.'),
 '06-sparse-tiles': ('Use finer sparse AO-pair output tasks with actual-width scratch',
   'Coarse output tiles limit sparse AO-pair task granularity. '
   'Target 32-AO tiles while keeping each shell intact and each output AO pair owned by one task. '
   'Allocate scratch using actual partition width so shells wider than the target remain supported.'),
 '07-potential': ('Fuse molecular XC potential and kinetic-density matrix assembly',
   'Potential assembly repeatedly scales AO arrays and contracts separate channels. '
   'Pack the LDA/GGA/MGGA channels together and assemble the potential with a 64-AO C kernel. '
   'Retain signed weights, Hermitian factors, partial blocks and decline-before-output-write fallbacks. '
   'This step requires the actual-width sparse scratch correction in the preceding tiles step.'),
 '08-cache-blocks': ('Build XC caches without overlapping AO and derivative buffers',
   'AO buffers and large XC derivative arrays can be live at the same time during cache construction. '
   'Finish density construction before creating blocked derivative outputs. Public return types remain full arrays; '
   'custom and overridden callbacks retain their whole-grid invocation semantics.'),
 '09-response': ('Store invariant XC derivatives for internal response actions',
   'Expanded Cartesian XC Hessians increase persistent response-cache storage. '
   'Store invariant derivatives and apply the exact chain rule, including sigma geometric Hessian terms, '
   'when building response weights. Keep public cache APIs and unsupported-input fallbacks; '
   'retain the parent fused AO assembly when integrating the compact response path.'),
}


def load(name):
    return json.loads((ROOT / name).read_text())


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def evidence_label(parent, current):
    if current['range'][1] < parent['range'][0]:
        return 'candidate observed range is faster'
    if current['range'][0] > parent['range'][1]:
        return 'candidate observed range is slower'
    return 'observed ranges overlap'


def main():
    data, stack = load('acceptance.json'), load('stack.json')
    response = load('response/runs/acceptance.json')
    dm = load('dm/runs/accepted/acceptance.json') if (ROOT / 'dm/runs/accepted/acceptance.json').is_file() else None
    assert data['accepted'] and data['accepted_runs'] == 90 and data['accepted_pairs'] == 405
    assert response['accepted'] and response['completed'] == 18
    assert load('response/runs/native-acceptance.json')['accepted']
    if dm is not None:
        assert dm['accepted'] and dm['completed'] == 48
        assert load('dm/runs/native-acceptance.json')['accepted']
    storage = {}
    for row in load('response/runs/batch.json')['rows']:
        observed = json.loads(Path(row['result']).read_text())
        storage.setdefault(row['case'], set()).add((observed['coulomb_eri_cached'], observed['coulomb_eri_bytes']))
    assert all(len(values) == 1 for values in storage.values()), 'Conventional Coulomb storage differs between response stages'
    (ROOT / 'response/runs/coulomb-storage.json').write_text(json.dumps(
        {case: list(values) for case, values in storage.items()}, indent=2) + '\n')
    stages, systems = stack['stages'], data['summaries']
    tests = {r['label']: r for r in load('tests.json')}
    measurements, increments = [], []
    for system in systems:
        for stage in stages:
            label = stage['label']
            m = system['versions'][label]
            row = {'case': system['case'], 'stage': label, 'head': stage['head'],
                   'grid_points': m['grid_points'], 'cycles': m['cycles'], 'polish_cycles': m['polish_cycles']}
            for field in ('process_wall_s', 'kernel_wall_s', 'peak_rss_mib'):
                row[field + '_median'] = m[field]['median']
                row[field + '_range'] = m[field]['range']
                row[field + '_samples'] = m[field]['samples']
            measurements.append(row)
        baseline = system['versions'][stages[0]['label']]
        for parent, stage in zip(stages, stages[1:]):
            a, b = system['versions'][parent['label']], system['versions'][stage['label']]
            increments.append({'case': system['case'], 'stage': stage['label'], 'head': stage['head'],
                'parent_stage': parent['label'], 'parent_head': parent['head'],
                'parent_seconds': a['process_wall_s']['median'], 'candidate_seconds': b['process_wall_s']['median'],
                'parent_range': a['process_wall_s']['range'], 'candidate_range': b['process_wall_s']['range'],
                'incremental_speedup': a['process_wall_s']['median'] / b['process_wall_s']['median'],
                'cumulative_speedup': baseline['process_wall_s']['median'] / b['process_wall_s']['median'],
                'parent_rss_mib': a['peak_rss_mib']['median'], 'candidate_rss_mib': b['peak_rss_mib']['median'],
                'parent_cycles': a['cycles'], 'candidate_cycles': b['cycles'],
                'observation': evidence_label(a['process_wall_s'], b['process_wall_s'])})
    write_csv(ROOT / 'measurements.csv', measurements)
    write_csv(ROOT / 'increments.csv', increments)
    full = stages[-1]
    lines = ['# Full implementation retained as a cumulative PR stack', '',
       '**90/90 full-SCF processes, 405/405 cross-version numerical pairs, and 18/18 CPKS processes accepted.**', '',
       f"Full branch: [DFT-full-stack](https://github.com/chen-yu-hao/pyscf/tree/DFT-full-stack), `{full['head']}`. "
       f"Its complete tracked tree is identical to the original full aggregate `{stack['target']}` "
       f"(tree `{stack['final_tree']}`). All five core C files also exactly match the earlier AO64 full implementation. "
       'No optimization family was removed based on an isolated benchmark.', '',
       '## Full-SCF endpoint', '',
       'MN15; the same frozen Bohr geometries, basis functions, 99x590 unpruned grids, screening thresholds, '
       'FP64 arithmetic, conventional direct J/K and strict convergence checks are used for every stage. '
       'Each cell has three fresh processes; timings below are medians of parent subprocess start-to-exit. '
       'No historical timing is substituted. CPU14–21 were selected after other load appeared on CPU5–12; '
       'OMP8/BLAS1, one physical core per thread, serial execution, shared Xeon Platinum 8176 host.', '',
       '| System | Baseline seconds | Full seconds | Full speedup | Baseline/full RSS MiB | Cycles baseline/full |',
       '|---|---:|---:|---:|---:|---|']
    for system in systems:
        a, b = system['versions'][stages[0]['label']], system['versions'][full['label']]
        lines.append(f"| {system['case']} {system['method']}/{system['basis']} | {a['process_wall_s']['median']:.4f} | "
          f"{b['process_wall_s']['median']:.4f} | {a['process_wall_s']['median']/b['process_wall_s']['median']:.3f}x | "
          f"{a['peak_rss_mib']['median']:.1f}/{b['peak_rss_mib']['median']:.1f} | {a['cycles']}/{b['cycles']} |")
    lines += ['', '## Incremental full-SCF gains', '',
       'Every row compares one stage with its immediate cumulative parent. Ratios below1 indicate an observed '
       'slowdown. Min/max ranges, all samples, cycles and RSS are preserved in increments.csv and measurements.csv. '
       'These are order-dependent merge increments, not isolated or leave-one-out contributions; they must not '
       'be replaced by kernel speedups or multiplied across different campaigns. Overlapping ranges do not '
       'establish a stable speed difference. Three repeats are descriptive, not confidence intervals.', '',
       '| Stage | Change | Water4 RKS increment | O2 UKS increment | Water12 RKS increment | Water12 cumulative |',
       '|---|---|---:|---:|---:|---:|']
    for stage in stages[1:]:
        rows = {r['case']: r for r in increments if r['stage'] == stage['label']}
        lines.append(f"| {stage['label']} | {DESCRIPTIONS[stage['label']][0]} | "
          + ' | '.join(f"{rows[c]['incremental_speedup']:.3f}x" for c in ('water4', 'o2', 'water12'))
          + f" | {rows['water12']['cumulative_speedup']:.3f}x |")
    lines += ['', 'All retained steps remain in the final implementation, including weak or negative observations. '
       'The cache and compact-response steps target response work and storage; their ordinary ground-state SCF '
       'measurements are recorded without claiming a ground-state acceleration.', '',
       '## Dependencies and PR preparation', '',
       'The branches form a cumulative review sequence. The hard source dependencies are actual-width sparse '
       'scratch before fused potential assembly, and blocked cache construction before compact response. '
       'Other branch ordering is the chosen measurement/merge order, not a mathematical prerequisite. '
       'The density proposal is split into separate DM and MO steps.', '',
       '| Branch | Direct parent | Head | Review diff |', '|---|---|---|---|']
    drafts = ROOT / 'drafts'
    drafts.mkdir(exist_ok=True)
    index = ['# Factual PR preparation notes', '',
       'Upstream target: pyscf/pyscf:master. These are factual notes for a human author to review, understand '
       'and rewrite, not externally submitted maintainer messages. The full implementation remains available '
       'throughout review. The first step can target upstream master; later steps should be submitted after '
       'their predecessor is merged, then rebased and revalidated against the actual resulting head. '
       'Fork compare links below show only the current child diff. Creating a cumulative head against master '
       'before its parent lands also includes pending prerequisites.', '',
       '| Note | Branch | Parent | Review | Upstream creation link after prerequisite merges |',
       '|---|---|---|---|---|']
    for parent, stage in zip(stages, stages[1:]):
        label, branch = stage['label'], stage['branch']
        parent_ref = parent.get('branch') or stack['base']
        review = f'https://github.com/chen-yu-hao/pyscf/compare/{parent_ref}...{branch}'
        create = f'https://github.com/pyscf/pyscf/compare/master...chen-yu-hao:pyscf:{branch}?expand=1&title={quote(DESCRIPTIONS[label][0])}'
        lines.append(f"| [{branch}](https://github.com/chen-yu-hao/pyscf/tree/{branch}) | {parent['label']} | `{stage['head']}` | [diff]({review}) |")
        index.append(f'| [{label}]({label}.md) | {branch} | {parent_ref} | [diff]({review}) | [create]({create}) |')
        notes = [f'# {DESCRIPTIONS[label][0]}', '',
           'Internal factual notes: human review, understanding and rewriting remain required before submission.', '',
           DESCRIPTIONS[label][1], '',
           f"Head: `{stage['head']}`. Direct parent: `{parent['head']}`. [Review diff]({review}).", '',
           'Full-SCF changes below are measured after all preceding stack changes, on the submitted code. '
           'They include grid setup, numerical integration, conventional J/K, diagonalization, convergence '
           'and process startup. Ratios are parent/candidate; lower than1 means slower.', '',
           '| MN15 system | Parent seconds [min/max] | Candidate seconds [min/max] | Increment | Observation |',
           '|---|---:|---:|---:|---|']
        for r in [r for r in increments if r['stage'] == label]:
            notes.append(f"| {r['case']} | {r['parent_seconds']:.4f} [{r['parent_range'][0]:.4f}/{r['parent_range'][1]:.4f}] | "
              f"{r['candidate_seconds']:.4f} [{r['candidate_range'][0]:.4f}/{r['candidate_range'][1]:.4f}] | "
              f"{r['incremental_speedup']:.3f}x | {r['observation']} |")
        t = tests[label]
        notes += ['', f"The stage-specific numerical suite ran {t['tests']} tests with {t['skipped']} skips. "
           'The full combined head independently ran109 tests:108 passed and one pre-existing backend skip. '
           'All90 SCF runs and405 cross-version pairs passed identical-grid, convergence, electron-count, '
           'energy, DM and spin checks. Raw ranges and RSS are in the evidence tables.', '',
           'Changed files:', '']
        names = subprocess.check_output(['git', 'diff', '--name-only', parent['head'], stage['head']],
                                        cwd=stack['repo'], text=True).splitlines()
        notes += ['- ' + name for name in names]
        if label == '04-density-dm' and dm is not None:
            notes += ['', 'The direct DM API also has48 accepted fresh processes comparing this head with03-ao. '
               'Unlike normal SCF, frozen bare-DM calls select the density-matrix evaluator directly. '
               'These XC-only timings exclude grid construction, J/K and SCF and do not substitute for '
               'the full-SCF increments above. The frozen core-Hamiltonian-projector DMs are not converged SCF densities.', '',
               '| Fixed-DM case | XC | Threads | Parent/candidate XC seconds | Speedup | Parent/candidate RSS MiB |',
               '|---|---|---:|---:|---:|---:|']
            for c in dm['comparisons']:
                notes.append(f"| {c['case']} | {c['xc']} | {c['threads']} | "
                   f"{c['baseline_seconds_median']:.4f}/{c['candidate_seconds_median']:.4f} | {c['speedup']:.3f}x | "
                   f"{c['baseline_peak_rss_mib_median']:.1f}/{c['candidate_peak_rss_mib_median']:.1f} |")
        if label in ('08-cache-blocks', '09-response'):
            before, after = ('baseline', 'cacheblocks') if label == '08-cache-blocks' else ('cacheblocks', 'compact')
            notes += ['', 'Response validation uses separate frozen converged TPSS/cc-pVDZ fixtures: '
               'water4 RKS and water3+OH UKS, level3 grids,8threads, three Cartesian-field RHSs. '
               'The timed interval includes cache construction plus the complete CPKS solve; it excludes SCF '
               'and checks the unpreconditioned residual and polarizability tensor.', '',
               '| Response case | Parent/candidate total seconds | Parent/candidate sampled peak MiB | Parent/candidate cache bytes |',
               '|---|---:|---:|---:|']
            for case in ('water4-rks', 'oh-water3-uks'):
                a = next(g for g in response['groups'] if g['case'] == case and g['version'] == before)
                b = next(g for g in response['groups'] if g['case'] == case and g['version'] == after)
                notes.append(f"| {case} | {a['response_wall_s']['median']:.4f}/{b['response_wall_s']['median']:.4f} | "
                   f"{a['sampled_peak_rss_mib']['median']:.1f}/{b['sampled_peak_rss_mib']['median']:.1f} | {a['cache_bytes']}/{b['cache_bytes']} |")
            notes += ['', 'Cache payload bytes are not whole-process RSS. Sampled RSS at5ms can miss brief transients.']
            notes += ['', 'Conventional Coulomb ERI storage is recorded per run and matches across compared stages; '
                      'see response/runs/coulomb-storage.json.']
        notes += ['', 'OpenAI Codex substantially assisted implementation, tests, benchmark automation and documentation. '
           f'The human author must review and understand the kernels and personally submit under [AI_POLICY.md]({POLICY}).', '']
        (drafts / (label + '.md')).write_text('\n'.join(notes))
    (drafts / 'INDEX.md').write_text('\n'.join(index) + '\n')
    if dm is not None:
        lines += ['', '## Direct-parent DM API coverage', '',
           '48 additional fresh processes compare03-ao with04-density-dm using identical frozen bare DMs, '
           'coordinates, weights and screening masks. This covers LDA/VWN, PBE and MN15 on288/283-AO '
           'RKS/UKS cases at8threads plus MN15 on1160/1146-AO cases at16threads. These are core-Hamiltonian '
           'projector densities, not converged SCF densities. Timing covers nr_rks/nr_uks only, with one '
           'excluded warmup and three timed calls per process. The reported statistic is the median of '
           'three process medians. All numerical outputs, source/native identities and effective threads '
           'are accepted. The16-core affinity was selected from free physical cores after the SCF/CPKS '
           'campaigns; these jobs did not run concurrently.', '',
           '| Fixed-DM case | XC | Threads | Parent seconds | Candidate seconds | XC speedup |',
           '|---|---|---:|---:|---:|---:|']
        for c in dm['comparisons']:
            lines.append(f"| {c['case']} | {c['xc']} | {c['threads']} | {c['baseline_seconds_median']:.4f} | "
                         f"{c['candidate_seconds_median']:.4f} | {c['speedup']:.3f}x |")
    e = data['max_errors']
    lines += ['', '## Numerical acceptance and reproduction', '',
       f"Largest cross-version energy difference: {e['energy_delta_hartree']:.3e} Ha; DM RMS {e['density_rms']:.3e}; "
       f"maximum DM element {e['density_max']:.3e}; UKS spin-square difference {e['spin_square_delta']:.3e}. "
       'Gates are1e-8 Ha,1e-9,1e-7 and1e-8 respectively. All ordered grid coordinates, weights and point counts '
       'match; every run must pass the same strict final convergence and overlap electron counts. '
       'Source commits, native files, imported library origins and effective thread counts were checked.', '',
       'Stage-specific suites together invoke109 tests with one existing skip; the full combined suite '
       'independently invokes the same109 tests with one existing skip. These overlapping counts must not '
       'be added as218 distinct tests. All CI-style lint commands pass. Build flags are '
       '`-O3 -DNDEBUG -msse3 -fopenmp`, GCC11.3.1, CMake4.3.1, LibXC7.0.0, without fast-math.', '',
       'The fullSCF worker is byte-identical to the historical worker. In a fresh evidence directory, build '
       'the source stages with build.py, run test.py, then run_scf.py. finish.py accepts all90 SCFs, '
       'runs18 response jobs, validates response native identity, renders this report and cleans registered '
       'scratch. Two converged reference fixtures are generated once from00-base with '
       'response/response_benchmark.py prepare --source SOURCE --waters 4 --spin 0 (or --spin 1) '
       '--output FIXTURE.npz, with PYTHONPATH=SOURCE; use the matching interpreter and '
       'current idle cores recorded in protocol.json. Paths are recorded execution provenance and should '
       'be adapted for a new machine; preserve the adapted protocol. Raw density arrays and response fixtures '
       'are retained locally in /home/lyl/soft/pyscf-stack-bench-20261027 and can be regenerated.', '',
       '[CPKS time/memory report](response/runs/REPORT.md), [SCF increments](increments.csv), '
       '[all measured samples](measurements.csv), [PR fact notes](drafts/INDEX.md), '
       '[kernel review guide](REVIEW_GUIDE.md). '
       'No upstream PR has been created. [AI_POLICY.md](' + POLICY + ') states: '
       '"A human must review any generated code and submit a pull request". '
       'OpenAI Codex substantially assisted implementation, testing, benchmark automation and documentation.', '']
    if dm is not None:
        lines += ['Additional DM API inputs, all timings, min/max ranges and RSS are in '
                  '[dm/runs/accepted/acceptance.json](dm/runs/accepted/acceptance.json). '
                  'Numerical gates are1e-8 electrons,1e-8 Ha in Exc and1e-9 in Vxc. '
                  'Existing raw density fixtures are reused in place; the scripts can regenerate them from00-base.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    print('Rendered full report,27 incremental rows,30 measured stage cells and9 PR fact drafts.', flush=True)


if __name__ == '__main__':
    main()
