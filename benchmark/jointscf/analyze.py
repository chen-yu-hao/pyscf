#!/usr/bin/env python
'''
Tables and figures of the JointSCF benchmark.

Usage:
    python analyze.py [results.json]

Writes tables.md and fig_*.png next to the results.
'''

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

CASE_LABELS = {
    'h2o-16': 'H2O ×16, random',
    'h2o-32': 'H2O ×32, random',
    'h2o-64': 'H2O ×64, random',
    'h2o-128': 'H2O ×128, random',
    'h2o-stretched-64': 'H2O ×64, stretched',
    'h2o-scan2d-64': 'H2O ×64, 2D scan',
    'c2h4-48': 'C2H4 ×48, random',
    'c2h4-path-48': 'C2H4 ×48, path',
    'h2co-40': 'H2CO ×40, random',
}
CASE_LEVEL = {
    'h2o-16': 'PBE/def2-SVP', 'h2o-32': 'PBE/def2-SVP', 'h2o-64': 'PBE/def2-SVP',
    'h2o-128': 'PBE/def2-SVP', 'h2o-stretched-64': 'PBE/def2-SVP',
    'h2o-scan2d-64': 'PBE/def2-SVP', 'c2h4-48': 'PBE/def2-SVP',
    'c2h4-path-48': 'PBE/def2-SVP', 'h2co-40': 'B3LYP/def2-TZVP',
}
CASE_ORDER = ['h2o-16', 'h2o-32', 'h2o-64', 'h2o-128', 'h2o-scan2d-64',
              'h2o-stretched-64', 'c2h4-path-48', 'c2h4-48', 'h2co-40']


def load(path):
    with open(path) as f:
        results = json.load(f)
    data = {}
    for key, run in results['runs'].items():
        case, method, _ = key.split('/')
        data.setdefault(case, {}).setdefault(method, []).append(run)
    return results['environment'], data

def wall(runs):
    '''median, min, max of the solve time'''
    t = np.array([r['wall'] for r in runs])
    return np.median(t), t.min(), t.max()

def fmt_time(runs):
    med, lo, hi = wall(runs)
    if len(runs) == 1:
        return '%.2f' % med
    return '%.2f (%.2f–%.2f)' % (med, lo, hi)

def nfock(runs):
    return float(np.mean(runs[0]['nfock']))

def find(methods, prefix, suffix=''):
    '''The method with n workers, e.g. direct_4p'''
    for name in methods:
        if name.startswith(prefix) and name.endswith(suffix) and name != prefix + '1' + suffix:
            body = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
            if body.isdigit():
                return name
    return None


def make_tables(env, data, out):
    cases = [c for c in CASE_ORDER if c in data]
    lines = []
    w = lines.append
    w('## Environment\n')
    w('| | |\n|---|---|')
    for key in ('cpu', 'ncpu', 'mem_gb', 'platform', 'python', 'numpy', 'scipy', 'pyscf', 'git'):
        if key in env:
            w(f'| {key} | {env[key]} |')
    w('')

    first = data[cases[0]]
    n = None
    for name in first:
        if name.startswith('direct_') and name.endswith('p'):
            n = name[len('direct_'):-1]
    d1, dnt, dnp = 'direct_1t', f'direct_{n}t', f'direct_{n}p'
    s1, snw = 'sync_1w', f'sync_{n}w'
    c1, cnw = 'cascade_1w', f'cascade_{n}w'

    w('## Table 1. Fock builds per configuration and accuracy\n')
    w('All methods use the same convergence criteria (conv_tol = 1e-9 Ha, '
      'gradient 3.2e-5).  E_ref: independent calculations converged to 1e-12 Ha.\n')
    w('| ensemble | level of theory | N_AO | direct | JointSCF sync | JointSCF cascade | '
      'direct / cascade | converged | max\\|E_direct − E_ref\\| | max\\|E_cascade − E_ref\\| |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if d1 not in m or c1 not in m:
            continue
        ref = np.array(m['reference'][0]['e_tot']) if 'reference' in m else None
        e_d = np.array(m[d1][0]['e_tot'])
        e_c = np.array(m[c1][0]['e_tot'])
        err_d = '%.1e' % abs(e_d - ref).max() if ref is not None else '-'
        err_c = '%.1e' % abs(e_c - ref).max() if ref is not None else '-'
        nconf = m[d1][0]['nconf']
        conv = '%d/%d' % (sum(m[c1][0]['converged']), nconf)
        sync = '%.2f' % nfock(m[s1]) if s1 in m else '-'
        w(f"| {CASE_LABELS[case]} | {CASE_LEVEL[case]} | {m[d1][0]['nao']} | "
          f"{nfock(m[d1]):.2f} | {sync} | {nfock(m[c1]):.2f} | "
          f"{nfock(m[d1]) / nfock(m[c1]):.2f}× | {conv} | {err_d} | {err_c} |")
    w('')

    w('## Table 2. Wall time on one core (seconds; median and range of the repeats)\n')
    w('| ensemble | direct | JointSCF sync | JointSCF cascade | speed-up | '
      'direct + AO cache | cascade + AO cache | speed-up (both with cache) | '
      'cascade + cache vs direct |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if d1 not in m or c1 not in m:
            continue
        td, tc = wall(m[d1])[0], wall(m[c1])[0]
        row = [CASE_LABELS[case], fmt_time(m[d1]),
               fmt_time(m[s1]) if s1 in m else '-', fmt_time(m[c1]), '%.2f×' % (td / tc)]
        if d1 + '_cache' in m and c1 + '_cache' in m:
            tdc, tcc = wall(m[d1 + '_cache'])[0], wall(m[c1 + '_cache'])[0]
            row += [fmt_time(m[d1 + '_cache']), fmt_time(m[c1 + '_cache']),
                    '%.2f×' % (tdc / tcc), '%.2f×' % (td / tcc)]
        else:
            row += ['-'] * 4
        w('| ' + ' | '.join(row) + ' |')
    w('')

    w(f'## Table 3. Wall time on {n} cores (seconds; median and range of the repeats)\n')
    w(f'direct ({n} threads): one calculation at a time with {n} OpenMP threads '
      '(the default behaviour of PySCF).  '
      f'direct ({n} processes): {n} independent single-thread calculations at a time.  '
      'The start-up of the worker processes is not included (Table 4).\n')
    w(f'| ensemble | direct ({n} threads) | direct ({n} processes) | JointSCF sync | '
      f'JointSCF cascade | speed-up vs {n} processes | direct ({n} proc.) + cache | '
      'cascade + cache | speed-up (both with cache) | cascade + cache vs direct 1 core |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if dnp not in m or cnw not in m:
            continue
        tp, tc = wall(m[dnp])[0], wall(m[cnw])[0]
        row = [CASE_LABELS[case], fmt_time(m[dnt]) if dnt in m else '-', fmt_time(m[dnp]),
               fmt_time(m[snw]) if snw in m else '-', fmt_time(m[cnw]), '%.2f×' % (tp / tc)]
        if dnp + '_cache' in m and cnw + '_cache' in m:
            tpc, tcc = wall(m[dnp + '_cache'])[0], wall(m[cnw + '_cache'])[0]
            row += [fmt_time(m[dnp + '_cache']), fmt_time(m[cnw + '_cache']),
                    '%.2f×' % (tpc / tcc)]
            row.append('%.1f×' % (wall(m[d1])[0] / tcc) if d1 in m else '-')
        else:
            row += ['-'] * 4
        w('| ' + ' | '.join(row) + ' |')
    w('')

    w('## Table 4. Start-up of the worker processes and peak memory\n')
    w(f'| ensemble | start-up direct ({n} proc.) / s | start-up JointSCF ({n} workers) / s | '
      'peak RSS direct 1 core / MB | peak RSS cascade 1 worker / MB | '
      'peak RSS cascade 1 worker + cache / MB |')
    w('|---|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if dnp not in m or cnw not in m:
            continue
        def rss(name):
            if name in m and m[name][0].get('maxrss_mb'):
                return '%.0f' % m[name][0]['maxrss_mb']
            return '-'
        w(f"| {CASE_LABELS[case]} | {np.median([r['startup'] for r in m[dnp]]):.2f} | "
          f"{np.median([r['startup'] for r in m[cnw]]):.2f} | {rss(d1)} | {rss(c1)} | "
          f"{rss(c1 + '_cache')} |")
    w('')

    w('## Table 5. Cascade schedule\n')
    w('Levels: number of configurations / SCF cycle of the promotion / mean number '
      'of Fock builds.  Model error: leave-one-out error of the cross-configuration '
      'model at the promotions (norm of the commutator [F, D]).\n')
    w('| ensemble | descriptors | levels (size / cycle / Fock builds) | model error at the promotions |')
    w('|---|---:|---|---|')
    for case in cases:
        m = data[case]
        if c1 not in m:
            continue
        r = m[c1][0]
        levels = np.array(r['levels'])
        nf = np.array(r['nfock'])
        prom = np.array(r['promoted_at'])
        lv = ', '.join('%d / %d / %.2f' % ((levels == i).sum(), prom[levels == i][0],
                                         nf[levels == i].mean())
                       for i in range(levels.max() + 1))
        errs = ', '.join('%.1e (%d src)' % (p['model_error'], p['nsrc'])
                         for p in (r['promotions'][0] if r['promotions'] else []))
        ndesc = r['ndescriptor'][0] if r['ndescriptor'] else '-'
        w(f"| {CASE_LABELS[case]} | {ndesc} | {lv} | {errs} |")
    w('')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    return n


# Chart colours: the first three slots of a palette validated for colour
# vision deficiencies; text uses ink colours only.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
GRAY = '#a8a7a1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_axisbelow(True)

def make_figures(data, n, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2,
                         'font.family': 'DejaVu Sans', 'figure.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE})
    cases = [c for c in CASE_ORDER if c in data and 'cascade_1w' in data[c]
             and 'direct_1t' in data[c]]
    labels = [CASE_LABELS[c] for c in cases]
    y = np.arange(len(cases))[::-1]

    # Figure 1: Fock builds per configuration
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(cases) + 1.3))
    _style(ax)
    nd = [nfock(data[c]['direct_1t']) for c in cases]
    nc = [nfock(data[c]['cascade_1w']) for c in cases]
    h = 0.30
    ax.barh(y + h / 2 + 0.02, nd, height=h, color=GRAY, label='independent mf.kernel()')
    ax.barh(y - h / 2 - 0.02, nc, height=h, color=BLUE, label='JointSCF, cascade')
    for yi, a, b in zip(y, nd, nc):
        ax.text(b + 0.12, yi - h / 2 - 0.02, '%.2f  (%.2f×)' % (b, a / b), va='center',
                ha='left', color=INK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, 12.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel('Fock builds per configuration (same convergence criteria)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_fock_builds.png'), dpi=200)
    plt.close(fig)

    # Figure 2: speed-up over the matching baseline, two panels with one scale
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.62 * len(cases) + 1.6), sharey=True)
    panels = [
        ('1 core: speed-up over independent calculations', 'direct_1t',
         [('direct_1t_cache', 'independent + AO cache', AQUA),
          ('cascade_1w', 'JointSCF cascade', BLUE),
          ('cascade_1w_cache', 'JointSCF cascade + AO cache', ORANGE)]),
        (f'{n} cores: speed-up over {n} independent processes', f'direct_{n}p',
         [(f'direct_{n}p_cache', 'independent + AO cache', AQUA),
          (f'cascade_{n}w', 'JointSCF cascade', BLUE),
          (f'cascade_{n}w_cache', 'JointSCF cascade + AO cache', ORANGE)]),
    ]
    h = 0.22
    xmax = 1.0
    for ax, (title, base, series) in zip(axes, panels):
        _style(ax)
        for k, (name, label, color) in enumerate(series):
            vals = []
            for c in cases:
                m = data[c]
                if base in m and name in m:
                    vals.append(wall(m[base])[0] / wall(m[name])[0])
                else:
                    vals.append(np.nan)
            pos = y + (1 - k) * (h + 0.03)
            ax.barh(pos, vals, height=h, color=color, label=label)
            xmax = max(xmax, np.nanmax(vals))
            if k == 2:
                for yi, v in zip(pos, vals):
                    if np.isfinite(v):
                        ax.text(v + 0.03, yi, '%.2f×' % v, va='center', ha='left',
                                color=INK, fontsize=8.5)
        ax.axvline(1.0, color=INK2, linewidth=0.8)
        ax.set_title(title, fontsize=9.5, color=INK, loc='left')
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_xlabel('speed-up (wall time of the baseline / wall time)')
    for ax in axes:
        ax.set_xlim(0, xmax * 1.16)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, color=INK)
    handles, leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(outdir, 'fig_speedup.png'), dpi=200)
    plt.close(fig)

    # Figure 3: batch size, water
    sizes = [m for m in (16, 32, 64, 128) if f'h2o-{m}' in data
             and 'cascade_1w' in data[f'h2o-{m}']]
    if len(sizes) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
        series = [('direct_1t', 'independent mf.kernel()', GRAY),
                  ('cascade_1w', 'JointSCF, cascade', BLUE)]
        for ax, quantity in zip(axes, ('nfock', 'time')):
            _style(ax)
            for name, label, color in series:
                if quantity == 'nfock':
                    vals = [nfock(data[f'h2o-{m}'][name]) for m in sizes]
                else:
                    vals = [wall(data[f'h2o-{m}'][name])[0] / m for m in sizes]
                ax.plot(sizes, vals, color=color, linewidth=2, marker='o', markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=label)
                fmt = '%.2f' if quantity == 'nfock' else '%.3f'
                ax.text(sizes[-1] * 1.08, vals[-1], fmt % vals[-1], va='center',
                        ha='left', color=INK, fontsize=8.5)
            ax.set_xscale('log', base=2)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(m) for m in sizes])
            ax.set_xlim(sizes[0] / 1.2, sizes[-1] * 1.5)
            ax.set_ylim(0, None)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_xlabel('configurations in the batch (H2O, PBE/def2-SVP)')
            ax.set_title('Fock builds per configuration' if quantity == 'nfock'
                         else 'wall time per configuration on one core / s',
                         fontsize=9.5, color=INK, loc='left')
        handles, leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, leg, loc='upper center', ncol=2, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(outdir, 'fig_batch_size.png'), dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results.json')
    env, data = load(path)
    outdir = os.path.dirname(os.path.abspath(path))
    n = make_tables(env, data, os.path.join(outdir, 'tables.md'))
    make_figures(data, n, outdir)
    print(open(os.path.join(outdir, 'tables.md')).read())
