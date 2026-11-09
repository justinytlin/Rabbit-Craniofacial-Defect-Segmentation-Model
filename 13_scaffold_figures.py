#!/usr/bin/env python3
"""Key-finding figures for Dr. Ren's nanogold scaffold scans.

Reads the web app's `scaffold` tab (radiomics_database/scaffold_database.csv —
one row per scan, k = 5 primary threshold) plus the per-scan depth profiles
written by 12_scaffold_gold.py, and draws one figure per finding of the
2026-10-26 study report:

  fig1_label_detected.png      labelled scaffolds vs unlabelled controls (gold % of scaffold)
  fig2_surface_crust.png       gold density vs depth + cumulative share; D50 ≈ 0.07 mm
  fig3_no_deep_migration.png   gold density below 1 mm: labelled ≈ control
  fig4_radial_position.png     where on the surface the cells sit, by scaffold
  fig5_by_run.png              gold amount by run / dose (confounded) per scaffold
  fig6_sections.png            representative coronal sections + top-view gold maps
  key_findings_summary.csv     group medians and the tests behind the annotations

Statistics follow the report: Mann-Whitney (labelled vs control, per scaffold),
Kruskal-Wallis + Holm-adjusted pairwise Mann-Whitney (scaffold effect), on
runs R2–R4 (the 8 mm scaffolds; R1 is the 6 mm pilot and is shown but not
tested). The duplicate rescan (6070) is excluded.

    python 13_scaffold_figures.py [--out DIR] [--no-sections]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

REPO = Path(__file__).resolve().parent
DB_CSV = REPO / 'radiomics_database' / 'scaffold_database.csv'
SCAFFOLD_ROOT = REPO.parent.parent.parent / 'Dr. Ren'
DEFAULT_OUT = SCAFFOLD_ROOT / 'results' / 'key_findings'

# ── palette (dataviz reference instance: categorical slots 1-3, chrome & ink) ──
COLORS = {'CG': '#2a78d6', 'MC': '#eb6834', 'PEG-MC': '#1baf7a'}
SCAFFOLDS = ['CG', 'MC', 'PEG-MC']
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, DEEMPH = '#e1e0d9', '#c3c2b7', '#c3c2b7'
CEILING = 0.02          # % of scaffold — control ceiling at k = 5 (threshold calibration)
RUN_MARKER = {'R1': 's', 'R2': 'o', 'R3': '^', 'R4': 'D'}
DPI = 200


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID,
        'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'axes.axisbelow': True,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'text.color': INK,
        'legend.frameon': False,
    })


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return 'p = n/a'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'


def title(fig, head, sub):
    fig.text(0.02, 0.985, head, ha='left', va='top', fontsize=12.5, fontweight='semibold', color=INK)
    fig.text(0.02, 0.945, sub, ha='left', va='top', fontsize=9, color=INK2, wrap=True)


def source(fig, text):
    fig.text(0.02, 0.012, text, ha='left', va='bottom', fontsize=7.5, color=MUTED)


def jitter(n, width=0.11, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def strip(ax, x, values, color, runs=None, floor=None, z=3, size=28, hollow=False):
    """Dots (≥8 px, surface ring) + a 2 px median bar."""
    v = np.asarray(values, float)
    shown = v.copy()
    if floor is not None:
        shown = np.where(shown <= 0, floor, shown)
    xs = x + jitter(len(v), seed=int(abs(hash((x, color))) % 1000))
    markers = [RUN_MARKER.get(r, 'o') for r in runs] if runs is not None else ['o'] * len(v)
    for xi, yi, m in zip(xs, shown, markers):
        ax.scatter(xi, yi, s=size, marker=m, facecolor=SURFACE if hollow else color,
                   edgecolor=color if hollow else SURFACE, linewidth=1.4 if hollow else 1.2,
                   zorder=z)
    if len(v):
        med = np.nanmedian(v)
        ax.plot([x - 0.22, x + 0.22], [med if med > 0 or floor is None else floor] * 2,
                color=color, lw=2, solid_capstyle='round', zorder=z + 1)
        return med
    return np.nan


def floor_tick(ax, floor, ticks, labels, values):
    """Log axis with zeros: add a '0' tick at the floor only when a zero is drawn there."""
    if np.any(np.asarray(values, float) <= 0):
        ax.set_yticks([floor] + ticks)
        ax.set_yticklabels(['0'] + labels)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def run_legend(ax, runs=('R2', 'R3', 'R4'), loc='lower right', extra=()):
    handles = [Line2D([], [], marker=RUN_MARKER[r], color=INK2, linestyle='', markersize=6,
                      markerfacecolor=INK2, label=f'run {r}') for r in runs]
    handles += list(extra)
    ax.legend(handles=handles, loc=loc, handletextpad=0.4, borderaxespad=0.2)


# ───────────────────────────────────────────────────────────────── data

def load():
    df = pd.read_csv(DB_CSV, dtype={'scan': str, 'run': str, 'batch': str})
    df = df[~df['note'].fillna('').str.contains('duplicate')].copy()
    df['control'] = df['labeled'].isin(['none', 'unlabeled'])
    df['labelled'] = df['labeled'].eq('PLL-AuNP')
    df['main'] = df['run'].isin(['R2', 'R3', 'R4'])
    return df


def depth_profile(scan: str):
    f = SCAFFOLD_ROOT / 'outputs' / f'{scan}_scaffold' / f'{scan}_scaffold_depth_profile.csv'
    if not f.is_file():
        return None
    p = pd.read_csv(f)
    return p if len(p) else None


# ───────────────────────────────────────────────────────────────── figures

def fig1_label(df, out, rows):
    main = df[df['main']]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.11, right=0.97)
    ax.set_yscale('log')
    floor = 2e-4
    for i, sc in enumerate(SCAFFOLDS):
        c = main[(main.scaffold == sc) & main.control]
        l = main[(main.scaffold == sc) & main.labelled]
        mc = strip(ax, i - 0.2, c.gold_frac_pct, DEEMPH, runs=c.run, floor=floor)
        ml = strip(ax, i + 0.2, l.gold_frac_pct, COLORS[sc], runs=l.run, floor=floor)
        p = stats.mannwhitneyu(l.gold_frac_pct, c.gold_frac_pct, alternative='greater').pvalue \
            if len(c) and len(l) else np.nan
        ratio = ml / mc if mc and mc > 0 else np.nan
        ax.text(i, 7.5, f'{ratio:,.0f}× the control median\n{fmt_p(p)}',
                ha='center', va='bottom', fontsize=8.5, color=INK2)
        rows.append(dict(figure='fig1', scaffold=sc, metric='gold_frac_pct',
                         labelled_n=len(l), labelled_median=ml, control_n=len(c),
                         control_median=mc, test='Mann-Whitney one-sided labelled > control', p=p))
    ax.axhline(CEILING, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(2.46, CEILING * 1.25, 'control ceiling 0.02 %', ha='right', va='bottom',
            fontsize=8, color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f'{s}\ncontrol   labelled' for s in SCAFFOLDS], color=INK2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(floor * 0.7, 60)
    floor_tick(ax, floor, [1e-3, 1e-2, 1e-1, 1, 10], ['0.001', '0.01', '0.1', '1', '10'],
               main.gold_frac_pct)
    ax.set_ylabel('gold-like voxels, % of scaffold volume (log)')
    run_legend(ax, loc='lower right')
    title(fig, 'The gold label is detected cleanly',
          'Gold-like voxels (per-slice scaffold baseline + 5σ) in every scan of runs R2–R4. Grey = unlabelled '
          'controls, colour = PLL-AuNP-labelled hMSCs. Bars are medians. The threshold was calibrated so that '
          '95 % of controls fall below 0.02 % of scaffold volume.')
    source(fig, 'Source: scaffold tab of radiomics_database.xlsx (web-app runs of 2026-10-28, k = 5).')
    fig.savefig(out / 'fig1_label_detected.png', dpi=DPI)
    plt.close(fig)


def fig2_depth(df, out, rows):
    lab = df[df['main'] & df['labelled']]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.08, right=0.98, wspace=0.28)
    axA, axB = axes
    axA.set_yscale('log')
    medians = {}
    for sc in SCAFFOLDS:
        profs, cums = [], []
        for scan in lab[lab.scaffold == sc].scan:
            p = depth_profile(scan)
            if p is None:
                continue
            profs.append(p.local_gold_pct.to_numpy())
            cums.append(np.cumsum(p.frac_of_gold.to_numpy()))
            x = (p.depth_lo_mm + 0.05).to_numpy()
            axA.plot(x, np.maximum(p.local_gold_pct, 1e-4), color=COLORS[sc], lw=0.8, alpha=0.22)
        if not profs:
            continue
        P = np.nanmedian(np.vstack(profs), axis=0)
        C = np.nanmedian(np.vstack(cums), axis=0)
        medians[sc] = (x, P, C)
        axA.plot(x, np.maximum(P, 1e-4), color=COLORS[sc], lw=2.2, solid_capstyle='round',
                 solid_joinstyle='round', zorder=4)
        axB.plot(x, C, color=COLORS[sc], lw=2.2, solid_capstyle='round', solid_joinstyle='round', zorder=4)
        axB.scatter([x[-1]], [C[-1]], s=24, color=COLORS[sc], edgecolor=SURFACE, linewidth=1.2, zorder=5)
        d50 = lab[lab.scaffold == sc].D50_mm.median()
        fs = lab[lab.scaffold == sc].frac_surface
        rows.append(dict(figure='fig2', scaffold=sc, metric='D50_mm', labelled_n=len(fs),
                         labelled_median=d50, extra=f'frac_within_0.25mm mean={fs.mean():.3f} sd={fs.std():.3f}'))
    for ax in axes:
        ax.set_xlim(0, 3.0)
        ax.set_xlabel('depth below the seeded top surface (mm)')
        for xref, lab_ in ((0.25, '0.25 mm'), (1.0, '1 mm')):
            ax.axvline(xref, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axA.set_ylim(1e-4, 20)
    axA.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    axA.set_yticklabels(['≤0.0001', '0.001', '0.01', '0.1', '1', '10'])
    axA.set_ylabel('local gold density, % of scaffold voxels at that depth (log)')
    axA.set_title('Gold density falls ~1000× within the first millimetre', loc='left')
    axB.set_ylim(0, 1.02)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_yticklabels(['0', '25', '50', '75', '100'])
    axB.set_ylabel('cumulative share of all gold (%)')
    axB.set_title('Almost all gold lies within 0.25 mm of the surface', loc='left')
    d50_all = lab.D50_mm.median()
    fs_rng = lab.groupby('scaffold').frac_surface.mean()
    axB.text(1.08, 0.50, f'median gold depth (D50)\n{d50_all:.2f} mm ≈ 2 voxels\n\n'
             f'within 0.25 mm:\n{100 * fs_rng.min():.0f}–{100 * fs_rng.max():.0f} % (scaffold means)',
             fontsize=9, color=INK2, va='center')
    handles = [Line2D([], [], color=COLORS[s], lw=2.2, label=s) for s in SCAFFOLDS]
    handles.append(Line2D([], [], color=INK2, lw=0.8, alpha=0.4, label='individual scans (left)'))
    axA.legend(handles=handles, loc='upper right')
    title(fig, 'At day 7 the labelled cells form a surface crust — there is no measurable penetration',
          'Labelled scaffolds, runs R2–R4. Thick lines are the per-scaffold median across scans; depth is measured '
          'from the top surface of each voxel column, so the dome shape does not bias it.')
    source(fig, 'Source: <scan>_scaffold_depth_profile.csv (0.1 mm bins, k = 5) from Dr. Ren/outputs/, written by the web app.')
    fig.savefig(out / 'fig2_surface_crust.png', dpi=DPI)
    plt.close(fig)


def fig3_deep(df, out, rows):
    main = df[df['main']]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.11, right=0.97)
    ax.set_yscale('log')
    floor = 2e-5
    for i, sc in enumerate(SCAFFOLDS):
        c = main[(main.scaffold == sc) & main.control]
        l = main[(main.scaffold == sc) & main.labelled]
        mc = strip(ax, i - 0.2, c.deep_density_pct, DEEMPH, runs=c.run, floor=floor)
        ml = strip(ax, i + 0.2, l.deep_density_pct, COLORS[sc], runs=l.run, floor=floor)
        p = stats.mannwhitneyu(l.deep_density_pct, c.deep_density_pct, alternative='two-sided').pvalue \
            if len(c) > 0 and len(l) > 0 else np.nan
        ax.text(i, 0.25, f'labelled median {ml:.4f} %\ncontrol median {mc:.4f} %\n{fmt_p(p)}',
                ha='center', va='bottom', fontsize=8.5, color=INK2)
        rows.append(dict(figure='fig3', scaffold=sc, metric='deep_density_pct', labelled_n=len(l),
                         labelled_median=ml, control_n=len(c), control_median=mc,
                         test='Mann-Whitney two-sided labelled vs control', p=p))
    ax.set_xticks(range(3))
    ax.set_xticklabels([f'{s}\ncontrol   labelled' for s in SCAFFOLDS], color=INK2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(floor * 0.7, 3)
    floor_tick(ax, floor, [1e-4, 1e-3, 1e-2, 1e-1, 1], ['0.0001', '0.001', '0.01', '0.1', '1'],
               main.deep_density_pct)
    ax.set_ylabel('deep gold density: gold > 1 mm deep,\n% of scaffold voxels > 1 mm deep (log)')
    run_legend(ax, loc='lower right')
    title(fig, 'Deep signal is at control level — no evidence of migration into the scaffold',
          'Gold-like voxels more than 1 mm below the surface, runs R2–R4. Labelled scaffolds are indistinguishable '
          'from unlabelled controls (single-scan controls per run give wide, non-significant comparisons). MC has the '
          'highest intrinsic bright-particle background.')
    source(fig, 'Source: scaffold tab, deep_density_pct at k = 5. Scans with no deep gold sit on the "0" tick.')
    fig.savefig(out / 'fig3_no_deep_migration.png', dpi=DPI)
    plt.close(fig)


def fig4_radial(df, out, rows):
    lab = df[df['main'] & df['labelled']]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={'width_ratios': [1.25, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.98, wspace=0.3)
    axA, axB = axes
    groups = [lab[lab.scaffold == s].mean_rnorm.dropna() for s in SCAFFOLDS]
    kw = stats.kruskal(*groups).pvalue
    pairs = [(0, 1), (0, 2), (1, 2)]
    raw = [stats.mannwhitneyu(groups[a], groups[b], alternative='two-sided').pvalue for a, b in pairs]
    adj = holm(raw)
    for i, sc in enumerate(SCAFFOLDS):
        g = lab[lab.scaffold == sc]
        med = strip(axA, i, g.mean_rnorm, COLORS[sc], runs=g.run)
        axA.text(i + 0.27, med, f'{med:.2f}', ha='left', va='center', fontsize=8.5, color=INK2)
        medc = strip(axB, i, g.frac_core, COLORS[sc], runs=g.run)
        axB.text(i + 0.27, medc, f'{100 * medc:.0f} %', ha='left', va='center', fontsize=8.5, color=INK2)
        rows.append(dict(figure='fig4', scaffold=sc, metric='mean_rnorm', labelled_n=len(g),
                         labelled_median=med, extra=f'frac_core median={medc:.3f}'))
    # Holm brackets
    y0 = 1.10
    for k, ((a, b), p) in enumerate(zip(pairs, adj)):
        y = y0 + 0.07 * k
        axA.plot([a, a, b, b], [y - 0.012, y, y, y - 0.012], color=AXIS, lw=0.8)
        axA.text((a + b) / 2, y + 0.005, f'Holm {fmt_p(p)}', ha='center', va='bottom', fontsize=7.8, color=INK2)
        rows.append(dict(figure='fig4', scaffold=f'{SCAFFOLDS[a]} vs {SCAFFOLDS[b]}', metric='mean_rnorm',
                         test='pairwise Mann-Whitney, Holm-adjusted', p=p, extra=f'raw p={raw[k]:.4f}'))
    rows.append(dict(figure='fig4', scaffold='all', metric='mean_rnorm', test='Kruskal-Wallis', p=kw))
    axA.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axA.text(-0.45, 0.505, 'r = 0.5', ha='left', va='bottom', fontsize=8, color=MUTED)
    axA.set_xticks(range(3)); axA.set_xticklabels(SCAFFOLDS, color=INK2)
    axA.set_xlim(-0.5, 2.6); axA.set_ylim(0.3, 1.34)
    axA.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
    axA.set_ylabel('mean normalised radial position of gold\n(0 = scaffold centre, 1 = edge)')
    axA.set_title(f'Kruskal-Wallis {fmt_p(kw)}', loc='left', color=INK2, fontsize=9.5)
    axB.set_xticks(range(3)); axB.set_xticklabels(SCAFFOLDS, color=INK2)
    axB.set_xlim(-0.5, 2.6); axB.set_ylim(0, 1.0)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); axB.set_yticklabels(['0', '25', '50', '75', '100'])
    axB.set_ylabel('share of gold in the central half-radius (%)')
    axB.set_title('Core share (r < 0.5)', loc='left', color=INK2, fontsize=9.5)
    run_legend(axB, loc='upper right')
    title(fig, 'The scaffolds differ in where on the surface the cells sit, not in how deep',
          'Labelled scaffolds, runs R2–R4. On CG the gold is peripheral; on PEG-MC it is central; MC is in between. '
          'Bars are medians; the effect holds within the 500k runs and within the 1M run.')
    source(fig, 'Source: scaffold tab, mean_rnorm and frac_core at k = 5 (radius normalised per slice by the equivalent scaffold radius).')
    fig.savefig(out / 'fig4_radial_position.png', dpi=DPI)
    plt.close(fig)


def fig5_runs(df, out, rows):
    runs = ['R1', 'R2', 'R3', 'R4']
    run_label = {'R1': 'R1\npilot, 6 mm\n4e5 / 8e5', 'R2': 'R2\n500k', 'R3': 'R3\n500k', 'R4': 'R4\n1M'}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.4), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.08, right=0.98, wspace=0.12)
    floor = 2e-4
    for ax, sc in zip(axes, SCAFFOLDS):
        ax.set_yscale('log')
        d = df[df.scaffold == sc]
        for i, r in enumerate(runs):
            c = d[(d.run == r) & d.control]
            l = d[(d.run == r) & d.labelled]
            strip(ax, i - 0.17, c.gold_frac_pct, DEEMPH, floor=floor, size=22)
            ml = strip(ax, i + 0.17, l.gold_frac_pct, COLORS[sc], floor=floor, size=22, hollow=(r == 'R1'))
            rows.append(dict(figure='fig5', scaffold=sc, metric='gold_frac_pct', run=r,
                             labelled_n=len(l), labelled_median=ml, control_n=len(c),
                             control_median=c.gold_frac_pct.median() if len(c) else np.nan))
        ax.axhline(CEILING, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(4)); ax.set_xticklabels([run_label[r] for r in runs], color=INK2, fontsize=8)
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(sc, loc='left', color=INK)
        ax.grid(axis='y')
    axes[0].set_ylim(floor * 0.7, 20)
    floor_tick(axes[0], floor, [1e-3, 1e-2, 1e-1, 1, 10], ['0.001', '0.01', '0.1', '1', '10'],
               df.gold_frac_pct)
    axes[0].set_ylabel('gold-like voxels, % of scaffold volume (log)')
    axes[0].text(3.55, CEILING * 1.3, 'control ceiling', ha='right', va='bottom', fontsize=7.5, color=MUTED)
    handles = [Line2D([], [], marker='o', linestyle='', color=INK2, markerfacecolor=INK2, markersize=6, label='labelled (R2–R4)'),
               Line2D([], [], marker='o', linestyle='', color=INK2, markerfacecolor=SURFACE, markersize=6, label='labelled (R1 pilot)'),
               Line2D([], [], marker='o', linestyle='', color=DEEMPH, markerfacecolor=DEEMPH, markersize=6, label='unlabelled / no-cell control')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.90), ncol=3,
               frameon=False, handletextpad=0.4, columnspacing=1.4)
    anova_note = ''
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        lab = df[df['main'] & df['labelled']].copy()
        lab['lg'] = np.log10(lab.gold_frac_pct.clip(lower=1e-4))
        m = smf.ols('lg ~ C(scaffold) + C(run)', data=lab).fit()
        a = anova_lm(m, typ=2)
        anova_note = (f"Type-II ANOVA on log gold fraction (R2–R4): run {fmt_p(a.loc['C(run)', 'PR(>F)'])}, "
                      f"scaffold {fmt_p(a.loc['C(scaffold)', 'PR(>F)'])}.")
        rows.append(dict(figure='fig5', scaffold='all', metric='log10 gold_frac_pct', test='ANOVA type II run',
                         p=a.loc['C(run)', 'PR(>F)']))
        rows.append(dict(figure='fig5', scaffold='all', metric='log10 gold_frac_pct', test='ANOVA type II scaffold',
                         p=a.loc['C(scaffold)', 'PR(>F)']))
    except Exception:                                   # noqa: BLE001
        pass
    title(fig, 'Gold amount differs by scaffold and strongly by run — dose is confounded with run',
          'Every scan, by run. R4 (1M cells) was a separate culture and scanning session from R2/R3 (500k), so '
          '500k vs 1M cannot be separated from batch effects. ' + anova_note)
    source(fig, 'Source: scaffold tab, gold_frac_pct at k = 5. R1 is the 6 mm pilot with 1–2 scaffolds per cell (descriptive only).')
    fig.savefig(out / 'fig5_by_run.png', dpi=DPI)
    plt.close(fig)


def fig6_sections(df, out, rows):
    """Representative labelled scan per scaffold: coronal slab + top-view gold map."""
    spec = importlib.util.spec_from_file_location('scaffold_gold', REPO / '12_scaffold_gold.py')
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    mp = sg.load_pipeline(SCAFFOLD_ROOT / 'analysis')
    lab = df[df['main'] & df['labelled'] & df.qc_overall.eq('pass')]
    picks = []
    for sc in SCAFFOLDS:
        g = lab[lab.scaffold == sc]
        med = g.gold_frac_pct.median()
        picks.append(g.iloc[(g.gold_frac_pct - med).abs().argsort().iloc[0]])
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.9), gridspec_kw={'height_ratios': [0.52, 1.04]})
    fig.subplots_adjust(top=0.82, bottom=0.05, left=0.045, right=0.985, hspace=0.3, wspace=0.06)
    FADE = np.array([0.93, 0.93, 0.92])
    NZ_MAX = 135                                  # pad all coronal panels to one height

    def scalebar(ax, x1, y, length):
        ax.plot([x1 - length, x1], [y, y], color=SURFACE, lw=4.5, solid_capstyle='butt', zorder=5)
        ax.plot([x1 - length, x1], [y, y], color=INK, lw=2, solid_capstyle='butt', zorder=6)
        ax.text(x1 - length / 2, y - 4, '1 mm', ha='center', va='bottom', fontsize=7.5, color=INK, zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', facecolor=SURFACE, edgecolor='none', alpha=0.9))

    for j, row in enumerate(picks):
        sc, scan = row.scaffold, row.scan
        scan_dir = SCAFFOLD_ROOT / str(row.batch) / scan
        vol, dz, px, n = sg.load_volume(scan_dir)
        voi, env, depth, rnorm, geo = mp.build_voi(vol, dz)
        base, sigma = mp.slice_baseline(vol, voi)
        gold = ((vol - base[:, None, None]) > 5.0 * sigma) & voi
        nz, ny, nx = vol.shape
        win = lambda a: np.clip((a - 600) / 2400.0, 0, 1)
        # coronal: mid-y slice of the CT, gold max-projected over a ±6-voxel slab
        y = ny // 2
        sec = win(vol[:, y, :])
        img = np.stack([sec] * 3, -1) * 0.92 + 0.08
        img[~env[:, y, :]] = img[~env[:, y, :]] * 0.2 + 0.8 * FADE
        slab = gold[:, max(0, y - 6):y + 7, :].any(axis=1)
        rgb = matplotlib.colors.to_rgb(COLORS[sc])
        img[slab] = rgb
        if nz < NZ_MAX:                            # same height for all three panels
            img = np.concatenate([img, np.tile(FADE, (NZ_MAX - nz, nx, 1))], axis=0)
        axA = axes[0, j]
        axA.imshow(img, interpolation='nearest', aspect=dz / px)
        axA.axhline(geo['z_end'] - 0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        axA.axis('off')
        axA.set_title(f'{sc} — scan {scan} ({row.run}, {int(row.cells) // 1000}k cells)\n'
                      f'gold {row.gold_frac_pct:.2f} % · D50 {row.D50_mm:.2f} mm · r̄ {row.mean_rnorm:.2f}',
                      loc='left', fontsize=9, color=INK)
        bar = 1.0 / px
        scalebar(axA, nx - 8, NZ_MAX - 8, bar)
        # top view: gold projected through the VOI over the mid-depth CT slice
        axB = axes[1, j]
        mid = win(vol[min(nz - 1, geo['z_end'] // 2)])
        top = np.stack([mid] * 3, -1) * 0.92 + 0.08
        outline = env[:geo['z_end']].any(axis=0)
        top[~outline] = top[~outline] * 0.2 + 0.8 * FADE
        top[gold.any(axis=0)] = rgb
        axB.imshow(top, interpolation='nearest')
        axB.axis('off')
        scalebar(axB, nx - 8, ny - 10, bar)
        rows.append(dict(figure='fig6', scaffold=sc, metric='representative scan', extra=scan))
    axes[0, 0].text(-0.01, 0.5, 'coronal section\n(top = seeded surface)', transform=axes[0, 0].transAxes,
                    rotation=90, ha='right', va='center', fontsize=8.5, color=INK2)
    axes[1, 0].text(-0.01, 0.5, 'top view\n(gold projected through the depth)', transform=axes[1, 0].transAxes,
                    rotation=90, ha='right', va='center', fontsize=8.5, color=INK2)
    title(fig, 'What the findings look like: a thin surface crust, placed differently on each scaffold',
          'One representative labelled scaffold per material (the scan nearest its material\'s median gold fraction, R2–R4). '
          'Coloured voxels exceed baseline + 5σ; dashed line = floor cut; faded = outside the scaffold envelope.')
    source(fig, 'Source: raw SCANCO stacks (37 µm) in Dr. Ren/<batch>/<scan>, segmented with analysis/microct_pipeline.py exactly as the web app does.')
    fig.savefig(out / 'fig6_sections.png', dpi=DPI)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--no-sections', action='store_true', help='skip fig6 (loads three CT stacks)')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    df = load()
    print(f'{len(df)} scans ({int(df.labelled.sum())} labelled, {int(df.control.sum())} controls; '
          f'{int(df.main.sum())} in R2–R4)')
    rows = []
    for fn in (fig1_label, fig2_depth, fig3_deep, fig4_radial, fig5_runs):
        fn(df, out, rows)
        print('  wrote', fn.__name__)
    if not a.no_sections:
        fig6_sections(df, out, rows)
        print('  wrote fig6_sections')
    pd.DataFrame(rows).to_csv(out / 'key_findings_summary.csv', index=False)
    print(f'Figures in {out}')


if __name__ == '__main__':
    main()
