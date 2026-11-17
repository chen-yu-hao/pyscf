#!/usr/bin/env python
'''
Tables and figures of the JointSCF benchmark.

Usage:
    python analyze.py [results.json] [ablation_results.json]

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


def nfock_shared(runs):
    return float(np.mean(runs[0].get('nfock_shared', [0])))


def nfock_polish(runs):
    return float(np.mean(runs[0].get('nfock_polish', [0])))


def make_tables(env, data, ablation, out):
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
    if n is None:
        n = '4'
    d1, dnt, dnp = 'direct_1t', f'direct_{n}t', f'direct_{n}p'
    s1, snw = 'sync_1w', f'sync_{n}w'
    c1, cnw = 'cascade_1w', f'cascade_{n}w'
    tp1, tpn = 'transferable_polish_1w', f'transferable_polish_{n}w'
    tn1, tnn = 'transferable_nopolish_1w', f'transferable_nopolish_{n}w'

    # Table 1: Exact SCF Fock builds & accuracy
    w('## Table 1. Fock builds per configuration and accuracy (exact SCF tier, conv_tol = 1e-9)\n')
    w('All methods below converge to PySCF default criteria (conv_tol = 1e-9 Ha, gradient 3.2e-5).  '
      'E_ref: independent calculations converged to 1e-12 Ha.\n')
    w('| ensemble | level of theory | N_AO | direct | sync | cascade | '
      'transferable polish (sh + po) | conv (casc / trans) | '
      'max\\|E_direct − E_ref\\| | max\\|E_cascade − E_ref\\| | '
      'mean / max\\|E_trans_polish − E_ref\\| |')
    w('|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|')
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
        sync = '%.2f' % nfock(m[s1]) if s1 in m else '-'
        casc_str = f"{nfock(m[c1]):.2f} ({nfock(m[d1]) / nfock(m[c1]):.2f}×)"
        if tp1 in m:
            e_tp = np.array(m[tp1][0]['e_tot'])
            tp_tot = nfock(m[tp1])
            tp_sh = nfock_shared(m[tp1])
            tp_po = nfock_polish(m[tp1])
            tp_str = f"{tp_tot:.2f} ({tp_sh:.2f} + {tp_po:.2f})"
            conv_tp = sum(m[tp1][0]['converged'])
            err_tp = (f"{abs(e_tp - ref).mean():.1e} / {abs(e_tp - ref).max():.1e}"
                      if ref is not None else '-')
        else:
            tp_str = '-'
            conv_tp = '-'
            err_tp = '-'
        conv_c = sum(m[c1][0]['converged'])
        w(f"| {CASE_LABELS[case]} | {CASE_LEVEL[case]} | {m[d1][0]['nao']} | "
          f"{nfock(m[d1]):.2f} | {sync} | {casc_str} | {tp_str} | "
          f"{conv_c}/{nconf} / {conv_tp}/{nconf} | {err_d} | {err_c} | {err_tp} |")
    w('')

    # Table 1b: Approximate mHa tier (polish=False)
    if any(tn1 in data[c] for c in cases):
        w('## Table 1b. Approximate tier: transferable (polish=False, converged=False, mHa variational bound)\n')
        w('Not comparable in accuracy to conv_tol=1e-9 SCF; listed separately with error distribution vs E_ref(1e-12).\n')
        w('| ensemble | nfock (shared only) | mean (E − E_ref) / mHa | median / mHa | '
          'P90 / mHa | max / mHa | min (signed) / mHa | mean est / mHa |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|')
        for case in cases:
            m = data[case]
            if tn1 not in m:
                continue
            ref = np.array(m['reference'][0]['e_tot']) if 'reference' in m else None
            e_tn = np.array(m[tn1][0]['e_tot'])
            est = float(np.mean(m[tn1][0].get('e_err_est', [0.0]))) * 1e3
            if ref is not None:
                de = (e_tn - ref) * 1e3
                ade = np.abs(de)
                w(f"| {CASE_LABELS[case]} | {nfock(m[tn1]):.2f} | {ade.mean():.3f} | "
                  f"{np.median(ade):.3f} | {np.percentile(ade, 90):.3f} | "
                  f"{ade.max():.3f} | {de.min():+.3e} | {est:.3f} |")
            else:
                w(f"| {CASE_LABELS[case]} | {nfock(m[tn1]):.2f} | - | - | - | - | - | {est:.3f} |")
        w('')

    # Table 2: Wall time on 1 core
    w('## Table 2. Wall time on one core (seconds; median and range of the repeats)\n')
    w('| ensemble | direct | sync | cascade | cascade speed-up | '
      'transferable polish | polish speed-up | transferable nopolish [approx mHa] | '
      'direct + AO cache | cascade + AO cache |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if d1 not in m or c1 not in m:
            continue
        td, tc = wall(m[d1])[0], wall(m[c1])[0]
        row = [
            CASE_LABELS[case],
            fmt_time(m[d1]),
            fmt_time(m[s1]) if s1 in m else '-',
            fmt_time(m[c1]),
            '%.2f×' % (td / tc),
            fmt_time(m[tp1]) if tp1 in m else '-',
            '%.2f×' % (td / wall(m[tp1])[0]) if tp1 in m else '-',
            fmt_time(m[tn1]) if tn1 in m else '-',
            fmt_time(m[d1 + '_cache']) if (d1 + '_cache') in m else '-',
            fmt_time(m[c1 + '_cache']) if (c1 + '_cache') in m else '-',
        ]
        w('| ' + ' | '.join(row) + ' |')
    w('')

    # Table 3: Wall time on n cores
    w(f'## Table 3. Wall time on {n} cores (seconds; median and range of the repeats)\n')
    w(f'direct ({n} threads): one calculation at a time with {n} OpenMP threads.  '
      f'direct ({n} processes): {n} independent single-thread calculations at a time.  '
      'Worker process start-up is reported separately in Table 4.\n')
    w(f'| ensemble | direct ({n} threads) | direct ({n} proc.) | sync ({n}w) | '
      f'cascade ({n}w) | cascade vs {n}p | transferable polish ({n}w) | polish vs {n}p | '
      f'transferable nopolish ({n}w) [approx] |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if dnp not in m or cnw not in m:
            continue
        tp, tc = wall(m[dnp])[0], wall(m[cnw])[0]
        row = [
            CASE_LABELS[case],
            fmt_time(m[dnt]) if dnt in m else '-',
            fmt_time(m[dnp]),
            fmt_time(m[snw]) if snw in m else '-',
            fmt_time(m[cnw]),
            '%.2f×' % (tp / tc),
            fmt_time(m[tpn]) if tpn in m else '-',
            '%.2f×' % (tp / wall(m[tpn])[0]) if tpn in m else '-',
            fmt_time(m[tnn]) if tnn in m else '-',
        ]
        w('| ' + ' | '.join(row) + ' |')
    w('')

    # Table 4: Start-up & Peak RSS
    w('## Table 4. Start-up of worker processes and peak memory (1 worker RSS / MB)\n')
    w(f'| ensemble | start-up direct ({n}p) / s | start-up cascade ({n}w) / s | '
      f'start-up transferable ({n}w) / s | RSS direct 1c | RSS cascade 1w | '
      f'RSS cascade + cache | RSS trans. polish 1w | RSS trans. nopolish 1w |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for case in cases:
        m = data[case]
        if dnp not in m or cnw not in m:
            continue
        def rss(name):
            if name in m and m[name][0].get('maxrss_mb'):
                return '%.0f' % m[name][0]['maxrss_mb']
            return '-'
        st_tp = ('%.2f' % np.median([r['startup'] for r in m[tpn]])) if tpn in m else '-'
        w(f"| {CASE_LABELS[case]} | {np.median([r['startup'] for r in m[dnp]]):.2f} | "
          f"{np.median([r['startup'] for r in m[cnw]]):.2f} | {st_tp} | "
          f"{rss(d1)} | {rss(c1)} | {rss(c1 + '_cache')} | {rss(tp1)} | {rss(tn1)} |")
    w('')

    # Table 5: Cascade schedule
    w('## Table 5. Cascade schedule\n')
    w('| ensemble | descriptors | levels (size / cycle / Fock builds) | model error at promotions |')
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

    # Table 6: Batch-size sublinearity scan (H2O x16/32/64/128)
    sizes = [sz for sz in (16, 32, 64, 128) if f'h2o-{sz}' in data]
    if sizes:
        w('## Table 6. Batch-size scaling (H2O ×16/32/64/128, PBE/def2-SVP)\n')
        w('| M | direct nfock/M | cascade nfock/M | trans_polish total (sh + po)/M | '
          'trans_polish total shared Focks | trans_nopolish nfock/M [approx] | '
          'nopolish mean \\|dE\\| / mHa | 1-core time/M (direct / cascade / trans_polish) / s |')
        w('|---:|---:|---:|---|---:|---:|---:|---|')
        for sz in sizes:
            m = data[f'h2o-{sz}']
            ref = np.array(m['reference'][0]['e_tot']) if 'reference' in m else None
            d_nf = nfock(m[d1]) if d1 in m else np.nan
            c_nf = nfock(m[c1]) if c1 in m else np.nan
            tp_str = '-'
            tp_sh_tot = '-'
            if tp1 in m:
                tp_str = f"{nfock(m[tp1]):.2f} ({nfock_shared(m[tp1]):.2f} + {nfock_polish(m[tp1]):.2f})"
                tp_sh_tot = '%d' % int(round(sum(m[tp1][0]['nfock_shared'])))
            tn_nf = f"{nfock(m[tn1]):.2f}" if tn1 in m else '-'
            tn_err = ('%.3f' % (abs(np.array(m[tn1][0]['e_tot']) - ref).mean() * 1e3)
                      if (tn1 in m and ref is not None) else '-')
            t_d = wall(m[d1])[0] / sz if d1 in m else np.nan
            t_c = wall(m[c1])[0] / sz if c1 in m else np.nan
            t_tp = wall(m[tp1])[0] / sz if tp1 in m else np.nan
            w(f"| {sz} | {d_nf:.2f} | {c_nf:.2f} | {tp_str} | {tp_sh_tot} | {tn_nf} | "
              f"{tn_err} | {t_d:.3f} / {t_c:.3f} / {t_tp:.3f} |")
        w('')

    # Table 7: Ablations
    if ablation:
        if ablation.get('batch_size'):
            w('## Table 7a. Ablation: molecule_batch (1 core)\n')
            w('| case | polish | molecule_batch | wall / s | nfock (sh + po) | mean / max \\|E − E_ref\\| (Ha) |')
            w('|---|---|---|---:|---|---:|')
            for r in ablation['batch_size']:
                w(f"| {CASE_LABELS.get(r['case'], r['case'])} | {r['polish']} | {r['molecule_batch']} | "
                  f"{r['wall']:.2f} | {r['nfock_mean']:.2f} ({r['nfock_shared_mean']:.2f} + {r['nfock_polish_mean']:.2f}) | "
                  f"{r['mean_abs_de']:.2e} / {r['max_abs_de']:.2e} |")
            w('')
        if ablation.get('correction_rank'):
            w('## Table 7b. Ablation: correction_rank r_Delta (1 core)\n')
            w('| case | polish | r_Delta | wall / s | nfock (sh + po) | mean / P90 / max \\|E − E_ref\\| (mHa) |')
            w('|---|---|---|---:|---|---:|')
            for r in ablation['correction_rank']:
                w(f"| {CASE_LABELS.get(r['case'], r['case'])} | {r['polish']} | {r['correction_rank']} | "
                  f"{r['wall']:.2f} | {r['nfock_mean']:.2f} ({r['nfock_shared_mean']:.2f} + {r['nfock_polish_mean']:.2f}) | "
                  f"{r['mean_abs_de']*1e3:.3f} / {r['p90_abs_de']*1e3:.3f} / {r['max_abs_de']*1e3:.3f} |")
            w('')
        if ablation.get('optimizer'):
            w('## Table 7c. Ablation: shared optimizer (lbfgs vs adam, 1 core)\n')
            w('| case | polish | optimizer | wall / s | nfock (sh + po) | mean / max \\|E − E_ref\\| (Ha) |')
            w('|---|---|---|---:|---|---:|')
            for r in ablation['optimizer']:
                w(f"| {CASE_LABELS.get(r['case'], r['case'])} | {r['polish']} | {r['optimizer']} | "
                  f"{r['wall']:.2f} | {r['nfock_mean']:.2f} ({r['nfock_shared_mean']:.2f} + {r['nfock_polish_mean']:.2f}) | "
                  f"{r['mean_abs_de']:.2e} / {r['max_abs_de']:.2e} |")
            w('')

    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    return n


SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
GRAY = '#a8a7a1'
BLUE, ORANGE, AQUA, PURPLE = '#2a78d6', '#eb6834', '#1baf7a', '#7c4dff'


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_axisbelow(True)


def make_figures(data, ablation, n, outdir):
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

    # Figure 1: Fock builds per configuration (exact tier)
    has_tp = all('transferable_polish_1w' in data[c] for c in cases)
    fig, ax = plt.subplots(figsize=(7.6, 0.58 * len(cases) + 1.4))
    _style(ax)
    nd = [nfock(data[c]['direct_1t']) for c in cases]
    nc = [nfock(data[c]['cascade_1w']) for c in cases]
    h = 0.25 if has_tp else 0.30
    if has_tp:
        nt_sh = [nfock_shared(data[c]['transferable_polish_1w']) for c in cases]
        nt_po = [nfock_polish(data[c]['transferable_polish_1w']) for c in cases]
        ax.barh(y + h + 0.03, nd, height=h, color=GRAY, label='independent mf.kernel()')
        ax.barh(y, nc, height=h, color=BLUE, label='JointSCF, cascade')
        ax.barh(y - h - 0.03, nt_sh, height=h, color=ORANGE,
                label='JointSCF, transferable (shared)')
        ax.barh(y - h - 0.03, nt_po, left=nt_sh, height=h, color='#f4a688',
                label='JointSCF, transferable (polish)')
        for yi, a, b, s, p in zip(y, nd, nc, nt_sh, nt_po):
            ax.text(b + 0.12, yi, '%.2f (%.2f×)' % (b, a / b), va='center',
                    ha='left', color=INK, fontsize=8)
            ax.text(s + p + 0.12, yi - h - 0.03, '%.2f (%.2f×)' % (s + p, a / (s + p)),
                    va='center', ha='left', color=INK, fontsize=8)
    else:
        ax.barh(y + h / 2 + 0.02, nd, height=h, color=GRAY, label='independent mf.kernel()')
        ax.barh(y - h / 2 - 0.02, nc, height=h, color=BLUE, label='JointSCF, cascade')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, max(nd) * 1.25)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel('Fock builds per configuration (exact SCF, conv_tol=1e-9)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_fock_builds.png'), dpi=200)
    plt.close(fig)

    # Figure 2: speed-up over matching baseline
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 0.65 * len(cases) + 1.7), sharey=True)
    panels = [
        ('1 core: speed-up over independent calculations', 'direct_1t',
         [('cascade_1w', 'JointSCF cascade', BLUE),
          ('cascade_1w_cache', 'JointSCF cascade + AO cache', AQUA),
          ('transferable_polish_1w', 'JointSCF transferable (polish=True)', ORANGE)]),
        (f'{n} cores: speed-up over {n} independent processes', f'direct_{n}p',
         [(f'cascade_{n}w', 'JointSCF cascade', BLUE),
          (f'cascade_{n}w_cache', 'JointSCF cascade + AO cache', AQUA),
          (f'transferable_polish_{n}w', 'JointSCF transferable (polish=True)', ORANGE)]),
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
            if np.any(np.isfinite(vals)):
                xmax = max(xmax, float(np.nanmax(vals)))
            for yi, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(v + 0.03, yi, '%.2f×' % v, va='center', ha='left',
                            color=INK, fontsize=7.5)
        ax.axvline(1.0, color=INK2, linewidth=0.8)
        ax.set_title(title, fontsize=9.5, color=INK, loc='left')
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_xlabel('speed-up (wall time of baseline / wall time)')
    for ax in axes:
        ax.set_xlim(0, xmax * 1.18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, color=INK)
    handles, leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(outdir, 'fig_speedup.png'), dpi=200)
    plt.close(fig)

    # Figure 3: batch size scaling (H2O x16..128)
    sizes = [m for m in (16, 32, 64, 128) if f'h2o-{m}' in data and 'cascade_1w' in data[f'h2o-{m}']]
    if len(sizes) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
        series = [
            ('direct_1t', 'independent mf.kernel()', GRAY, False),
            ('cascade_1w', 'JointSCF cascade', BLUE, False),
            ('transferable_polish_1w', 'transferable polish (total)', ORANGE, False),
            ('transferable_polish_1w', 'transferable shared phase only', ORANGE, True),
        ]
        for ax, quantity in zip(axes, ('nfock', 'time')):
            _style(ax)
            for name, label, color, shared_only in series:
                if any(name not in data[f'h2o-{m}'] for m in sizes):
                    continue
                if quantity == 'nfock':
                    vals = [nfock_shared(data[f'h2o-{m}'][name]) if shared_only
                            else nfock(data[f'h2o-{m}'][name]) for m in sizes]
                    ls = '--' if shared_only else '-'
                else:
                    if shared_only:
                        continue
                    vals = [wall(data[f'h2o-{m}'][name])[0] / m for m in sizes]
                    ls = '-'
                ax.plot(sizes, vals, color=color, linewidth=2, linestyle=ls, marker='o',
                        markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
                fmt = '%.2f' if quantity == 'nfock' else '%.3f'
                ax.text(sizes[-1] * 1.07, vals[-1], fmt % vals[-1], va='center',
                        ha='left', color=INK, fontsize=8)
            ax.set_xscale('log', base=2)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(m) for m in sizes])
            ax.set_xlim(sizes[0] / 1.2, sizes[-1] * 1.5)
            ax.set_ylim(0, None)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8)
            ax.set_xlabel('configurations in batch (H2O, PBE/def2-SVP)')
            ax.set_title('Fock builds per configuration' if quantity == 'nfock'
                         else 'wall time per configuration on 1 core / s',
                         fontsize=9.5, color=INK, loc='left')
        handles, leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, leg, loc='upper center', ncol=2, frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(os.path.join(outdir, 'fig_batch_size.png'), dpi=200)
        plt.close(fig)

    # Figure 4: Ablation (if available)
    if ablation and ablation.get('correction_rank') and ablation.get('batch_size'):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
        ax0, ax1 = axes
        _style(ax0)
        _style(ax1)
        ranks = [0, 1, 2, None]
        rank_labels = ['0', '1', '2', 'None']
        for c, lbl, col in [('h2o-64', 'H2O ×64', BLUE),
                            ('h2o-stretched-64', 'H2O ×64 stretched', ORANGE),
                            ('c2h4-48', 'C2H4 ×48', AQUA),
                            ('h2co-40', 'H2CO ×40', PURPLE)]:
            ys = []
            for r_d in ranks:
                matches = [r for r in ablation['correction_rank']
                           if r['case'] == c and not r['polish'] and r['correction_rank'] == r_d]
                ys.append(matches[0]['mean_abs_de'] * 1e3 if matches else np.nan)
            ax0.plot(range(len(ranks)), ys, marker='o', linewidth=2, color=col, label=lbl)
        ax0.set_xticks(range(len(ranks)))
        ax0.set_xticklabels(rank_labels)
        ax0.set_yscale('log')
        ax0.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax0.set_xlabel('correction_rank r_Delta')
        ax0.set_ylabel('mean |E - E_ref| / mHa')
        ax0.set_title('polish=False energy error vs r_Delta', fontsize=9.5, color=INK, loc='left')
        ax0.legend(frameon=False, fontsize=8)

        mbs = [1, 4, 8, 'full']
        for c, lbl, col in [('h2o-64', 'H2O ×64', BLUE),
                            ('h2o-scan2d-64', 'H2O ×64 scan2d', AQUA),
                            ('c2h4-path-48', 'C2H4 ×48 path', ORANGE),
                            ('c2h4-48', 'C2H4 ×48 random', PURPLE)]:
            ys = []
            for mb in mbs:
                matches = [r for r in ablation['batch_size']
                           if r['case'] == c and r['polish'] and r['molecule_batch'] == mb]
                ys.append(matches[0]['nfock_mean'] if matches else np.nan)
            ax1.plot(range(len(mbs)), ys, marker='o', linewidth=2, color=col, label=lbl)
        ax1.set_xticks(range(len(mbs)))
        ax1.set_xticklabels([str(x) for x in mbs])
        ax1.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax1.set_xlabel('molecule_batch')
        ax1.set_ylabel('total Fock builds / config')
        ax1.set_title('polish=True Fock/config vs mini-batch size', fontsize=9.5, color=INK, loc='left')
        ax1.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'fig_transferable_ablation.png'), dpi=200)
        plt.close(fig)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results.json')
    abl_path = (sys.argv[2] if len(sys.argv) > 2
                else os.path.join(os.path.dirname(os.path.abspath(path)), 'ablation_results.json'))
    env, data = load(path)
    ablation = None
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            ablation = json.load(f)
    outdir = os.path.dirname(os.path.abspath(path))
    n = make_tables(env, data, ablation, os.path.join(outdir, 'tables.md'))
    make_figures(data, ablation, n, outdir)
    print(open(os.path.join(outdir, 'tables.md')).read())
