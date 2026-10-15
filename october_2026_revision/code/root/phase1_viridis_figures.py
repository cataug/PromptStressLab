#!/usr/bin/env python3
"""Redraw Phase I Fig 3-5 with viridis. No recalculation, no API calls."""
from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

LEVELS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
FIELDS = ['concat', 'research_objective', 'methodology',
          'data_or_material', 'key_results', 'application_domain']
NAMES = {'concat': 'Concatenated output',
         'research_objective': 'Research objective',
         'methodology': 'Methodology',
         'data_or_material': 'Data / material',
         'key_results': 'Key results',
         'application_domain': 'Application domain'}
FIGS = ['Figure3_PSI_ECDF_with_insets_corrected',
        'Figure4_Field_Volatility_gradient_corrected',
        'Figure5_Pareto_common_axes_local_insets_corrected']
VIR = plt.get_cmap('viridis')
PCOLORS = {p: VIR(x) for p, x in zip(LEVELS[1:], np.linspace(.10, .88, 5))}
GCOLORS = {'short': VIR(.17), 'medium': VIR(.51), 'long': VIR(.85)}
MARKERS = {'short': 'o', 'medium': '^', 'long': 's'}


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.titlesize': 14, 'axes.labelsize': 12,
        'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.facecolor': 'white', 'axes.facecolor': '#f8faf9',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.linestyle': '--', 'grid.linewidth': .65, 'grid.alpha': .22,
    })


def save(fig, out, name):
    pdf = out / (name + '.pdf')
    png = out / (name + '.png')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(png, bbox_inches='tight', dpi=260, facecolor='white')
    plt.close(fig)
    print('[FIG]', pdf, flush=True)


def draw3(trans, out):
    # Original notebook layout: 2x3 ECDF, translucent step-histogram insets.
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.4), layout='constrained')
    for ax, field in zip(axes.flat, FIELDS):
        ax.set_title(NAMES[field], pad=9, weight='medium')
        for p in LEVELS[1:]:
            v = trans.loc[trans['prompt_level'].eq(p), 'psi_' + field]
            vals = np.sort(pd.to_numeric(v, errors='coerce').dropna().to_numpy(float))
            if vals.size:
                ax.step(vals, np.arange(1, vals.size + 1) / vals.size,
                        where='post', lw=2.1, color=PCOLORS[p], label=p)
        ax.set(xlim=(0, 1), ylim=(0, 1.015), xlabel='State-aware TF-IDF PSI', ylabel='ECDF')
        ax.grid(True)
        ax.set_axisbelow(True)
        ins = inset_axes(ax, width='38%', height='36%', loc='lower right', borderpad=.9)
        ins.set_facecolor('#fafbf9')
        bins = np.linspace(0, 1, 24)
        for p in LEVELS[1:]:
            vals = pd.to_numeric(trans.loc[trans['prompt_level'].eq(p), 'psi_' + field],
                                 errors='coerce').dropna().to_numpy(float)
            if len(vals):
                ins.hist(vals, bins=bins, density=True, histtype='stepfilled',
                         facecolor=PCOLORS[p], alpha=.075)
                ins.hist(vals, bins=bins, density=True, histtype='step',
                         color=PCOLORS[p], lw=1.05)
        ins.set_xlim(0, 1)
        ins.set_xticks([])
        ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_visible(True)
            spine.set_color('#89958e')
            spine.set_linewidth(.75)
    fig.legend(handles=[Line2D([0], [0], lw=2.6, color=PCOLORS[p], label=p)
                        for p in LEVELS[1:]],
               loc='outside lower center', frameon=False, ncol=5)
    save(fig, out, FIGS[0])


def draw4(vol, out):
    # Same bins and same global count color scale across six panels.
    bins = np.linspace(0, 1, 21)
    count_max = max(int(np.histogram(pd.to_numeric(vol['vol_' + f], errors='coerce').dropna(),
                                     bins=bins)[0].max()) for f in FIELDS)
    norm = Normalize(vmin=0, vmax=max(1, count_max))
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.4), layout='constrained')
    for ax, field in zip(axes.flat, FIELDS):
        vals = pd.to_numeric(vol['vol_' + field], errors='coerce').dropna().to_numpy(float)
        heights, _, bars = ax.hist(vals, bins=bins, color=VIR(.45),
                                  edgecolor='#293b33', linewidth=.55)
        for h, bar in zip(heights, bars):
            bar.set_facecolor(VIR(.06 + .9 * norm(h)))
        avg = float(vals.mean())
        p95 = float(np.quantile(vals, .95))
        ax.axvline(avg, color='#151e1a', ls='--', lw=1.6, zorder=5)
        ax.axvline(p95, color='#454e48', ls=':', lw=1.6, zorder=5)
        ax.text(.97, .94, f'Mean = {avg:.3f}\nP95 = {p95:.3f}',
                ha='right', va='top', transform=ax.transAxes, fontsize=10.5,
                bbox={'boxstyle': 'round,pad=.35', 'facecolor': 'white',
                      'edgecolor': '#bdcfc3', 'alpha': .94})
        ax.set(title=NAMES[field], xlabel='Volatility', ylabel='Documents',
               xlim=(0, 1), ylim=(0, count_max * 1.10))
        ax.grid(axis='y')
        ax.set_axisbelow(True)
    sm = ScalarMappable(norm=norm, cmap=VIR)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=.75, pad=.014, aspect=26)
    cbar.set_label('Documents per bin (shared count scale)')
    fig.legend(handles=[Line2D([0], [0], color='#151e1a', lw=1.6, ls='--', label='Mean'),
                        Line2D([0], [0], color='#454e48', lw=1.6, ls=':', label='P95')],
               ncol=2, loc='outside lower center', frameon=False)
    save(fig, out, FIGS[1])


def front(df):
    # Two-objective Pareto minimization: cost and latency.
    df = df.sort_values(['usage_total_tokens', 'latency_seconds'])
    current_best = float('inf')
    selected = []
    for idx, row in df.iterrows():
        if float(row['latency_seconds']) < current_best - 1e-12:
            selected.append(idx)
            current_best = float(row['latency_seconds'])
    return df.loc[selected]


def draw5(run, out):
    cols = ['estimated_abstract_tokens', 'usage_total_tokens', 'latency_seconds']
    r = run.copy()
    for c in cols:
        r[c] = pd.to_numeric(r[c], errors='coerce')
    r = r.dropna(subset=cols)
    assert all((r['prompt_level'] == p).sum() == 530 for p in LEVELS), 'Expected 530 calls per condition'
    q33, q66 = r['estimated_abstract_tokens'].quantile([1/3, 2/3]).tolist()
    r['stratum'] = np.select([r['estimated_abstract_tokens'] <= q33,
                              r['estimated_abstract_tokens'] <= q66],
                             ['short', 'medium'], default='long')
    x_max = max(1, float(r['usage_total_tokens'].max()) * 1.03)
    y_max = max(1, float(r['latency_seconds'].max()) * 1.04)
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 10), sharex=True, sharey=True,
                             layout='constrained')
    for ax, p in zip(axes.flat, LEVELS):
        d = r.loc[r['prompt_level'].eq(p)]
        ax.scatter(d['usage_total_tokens'], d['latency_seconds'],
                   s=8, alpha=.10, color='#74847c', rasterized=True)
        ins = inset_axes(ax, width='48%', height='45%', loc='upper right', borderpad=.85)
        ins.set_facecolor('#fcfdfb')
        counts = {}
        for g in ('short', 'medium', 'long'):
            small = d.loc[d['stratum'].eq(g)]
            pf = front(small)
            counts[g] = len(pf)
            ax.scatter(pf['usage_total_tokens'], pf['latency_seconds'],
                       s=35, marker=MARKERS[g], facecolors=GCOLORS[g],
                       edgecolors='#1c2b23', linewidths=.5, zorder=6)
            ins.scatter(small['usage_total_tokens'], small['latency_seconds'],
                        s=8, marker=MARKERS[g], color=GCOLORS[g], alpha=.14,
                        rasterized=True)
            ins.scatter(pf['usage_total_tokens'], pf['latency_seconds'],
                        s=25, marker=MARKERS[g], color=GCOLORS[g],
                        edgecolors='#1c2b23', linewidths=.4)
        left, right = np.quantile(d['usage_total_tokens'], [.01, .99])
        low, high = np.quantile(d['latency_seconds'], [.01, .99])
        dx = max(6, (right - left) * .15)
        dy = max(.15, (high - low) * .15)
        ins.set_xlim(left - dx, right + dx)
        ins.set_ylim(max(0, low - dy), high + dy)
        ins.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ins.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ins.tick_params(labelsize=7, pad=1)
        ins.set_title('Local view', fontsize=9, pad=2)
        for spine in ins.spines.values():
            spine.set_visible(True)
            spine.set_color('#82938b')
            spine.set_linewidth(.8)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_title(f'{p}  |  Pareto: s={counts["short"]}, '
                     f'm={counts["medium"]}, l={counts["long"]}',
                     weight='medium', pad=8)
        ax.grid(True)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel('Total tokens (input + completion)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Latency (seconds)')
    labels = [('short', f'Short (≤ Q33={q33:.1f})'),
              ('medium', f'Medium (Q33–Q66={q66:.1f})'),
              ('long', 'Long (> Q66)')]
    fig.legend(handles=[Line2D([0], [0], linestyle='None', marker=MARKERS[g],
                               markerfacecolor=GCOLORS[g], markeredgecolor='#1c2b23',
                               markersize=9, label=label) for g, label in labels],
               loc='outside lower center', ncol=3, frameon=False)
    save(fig, out, FIGS[2])
    print(f'[TERCILES] Q33={q33:.3f}, Q66={q66:.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/tahiti/PromptStressLab')
    args = parser.parse_args()
    root = Path(args.root)
    out = root / 'outputs' / 'phase1_reanalysis_2026'
    fig_dir = root / 'figs'
    legacy = root / 'legacy_PromptEng26' / 'run_summary.xlsx'
    expected = [out / 'phase1_corrected_transitions.csv',
                out / 'phase1_corrected_volatility.csv', legacy]
    for p in expected:
        if not p.is_file():
            raise FileNotFoundError(f'Missing input: {p}')
    trans = pd.read_csv(expected[0])
    vol = pd.read_csv(expected[1])
    run = pd.read_excel(legacy, sheet_name='per_call')
    print('[DATA]', len(trans), 'transitions;', len(vol), 'documents;', len(run), 'API records', flush=True)
    assert len(trans) == 2650 and len(vol) == 530 and len(run) == 3180
    configure()

    # Backup existing graphics before replacement.
    for folder in (out, fig_dir):
        backup = folder / 'before_viridis'
        for stem in FIGS:
            for ext in ('.pdf', '.png'):
                current = folder / (stem + ext)
                if current.exists():
                    backup.mkdir(parents=True, exist_ok=True)
                    old = backup / current.name
                    if not old.exists():
                        shutil.copy2(current, old)

    draw3(trans, out)
    draw4(vol, out)
    draw5(run, out)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for stem in FIGS:
        for ext in ('.pdf', '.png'):
            shutil.copy2(out / (stem + ext), fig_dir / (stem + ext))

    print('[OK] Figures 3–5 in viridis; originals backed up; LaTeX filenames unchanged.', flush=True)
    print('[DIR]', fig_dir, flush=True)


if __name__ == '__main__':
    main()
