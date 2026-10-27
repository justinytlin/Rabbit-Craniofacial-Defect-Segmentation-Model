#!/usr/bin/env python3
"""Ex vivo figure set using EVERY OPG-study specimen, all treatment groups.

Same style as 9_opg_figures.py (ex_vivo/), but with all groups of the OPG
study at both ex vivo timepoints, each treatment kept separate:

  3 months (2021 cohort, lab sheets #182/#184, intact trephined discs,
  full-cylinder ROI): Defect, CG, MC, Positive Control, C-OPG + CG,
  C-OPG + MC, Soaked-OPG + CG, Soaked-OPG + MC — 4 rabbits each.
  6 months (UCLA cohort, #262, halved specimens, half-cylinder ROI):
  Defect, MC, C-OPG, Soaked-OPG.

One value per RABBIT: rabbits scanned more than once (#182 first run, #184
second run, #262 repeat 5783) are averaged. Excluded: the low-res duplicate
scans 4131-4136, and 4202 (its sheet entry "u57-96 MC3" names the
C-OPG MC3 rabbit with the MC3 label — unresolved). 4194 (S-OPG MC4 second
run) is on disk but was never segmented; that rabbit enters via 4152.

Groups are compared WITHIN each timepoint only (different cohorts, ROI
geometry and specimen preparation): one-way ANOVA, Tukey HSD; brackets
show significant differences versus Defect, the full Tukey table is in
Table1. Heatmap z-scores are computed within each timepoint.

    python 9c_exvivo_all_figures.py [--images]
Outputs: ../figures/opg_comparison/ex_vivo_all/
"""

import argparse
import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


FIG = _load('_fig', '9_opg_figures.py')
SUB = 'ex_vivo_all'

# Rabbit -> scans, from lab sheets #182 / #184 (3 months) and #262 (6 months).
RABBITS = {
    3: {
        'Defect': {'u56-11': ['4117', '4163'], 'u56-17': ['4116', '4164'],
                   'u57-99': ['4157', '4165'], 'u58-00': ['4158', '4166']},
        'CG': {'u56-14': ['4118', '4171'], 'u56-10': ['4121'], 'u56-13': ['4127'], 'u57-88': ['4156']},
        'MC': {'u56-23': ['4115', '4130', '4175'], 'u56-24': ['4119', '4176'],
               'u57-95': ['4146', '4177'], 'u57-87': ['4155', '4178']},
        'Positive Control': {'u56-22': ['4122', '4167'], 'u56-19': ['4123', '4168'],
                             'u56-20': ['4124', '4169'], 'u57-91': ['4153', '4170']},
        'C-OPG + CG': {'u56-18': ['4126'], 'u56-16': ['4139'], 'u57-97': ['4147', '4181'],
                       'u57-90': ['4142', '4182']},
        'C-OPG + MC': {'u56-12': ['4125'], 'u56-15': ['4140'], 'u57-96': ['4145', '4185'],
                       'u57-98': ['4151', '4186']},
        'Soaked-OPG + CG': {'u56-21': ['4138', '4187'], 'u57-86': ['4141', '4188'],
                            'u57-85': ['4143', '4189'], 'u57-93': ['4148', '4190']},
        'Soaked-OPG + MC': {'u56-09': ['4128', '4191'], 'u57-89': ['4129', '4192'],
                            'u57-94': ['4149', '4193'], 'u57-92': ['4152']},
    },
    6: {
        'Defect': {'MR52523': ['5782', '5783'], 'MR52524': ['5784']},
        'MC': {'MR52519': ['5778'], 'MR52520': ['5779']},
        'C-OPG': {'MR52521': ['5780'], 'MR52522': ['5781'], 'MR52525': ['5785'],
                           'MR52526': ['5786']},
        'Soaked-OPG': {'MR52527': ['5787'], 'MR52528': ['5788'], 'MR52529': ['5789'],
                       'MR52530': ['5790']},
    },
}
EXCLUDED = {
    '4131-4136': 'low-res duplicate scans (re-scanned as 4138-4143)',
    '4202': 'sheet #184 lists it as "u57-96 MC3": rabbit ID of C-OPG MC3, group label of MC3 — identity unresolved',
    '4194': 'S-OPG MC4 second run; on disk but never segmented (rabbit included via 4152)',
}

# Representative scans chosen by hand (user, 2026-10-12); others = closest to group median.
REP_SCANS = {(3, 'Defect'): '4157', (3, 'CG'): '4121'}

# Groups left out of the analysis (user, 2026-10-14): every CG-containing group.
DROP_GROUPS = ('CG', 'C-OPG + CG', 'Soaked-OPG + CG')

# Positive Control last among the 3-month groups (user, 2026-10-14); 3-month and
# 6-month groups never share a panel row, so it sits right of / below Soaked-OPG + MC.
ORDER = ['Defect', 'CG', 'MC', 'C-OPG + CG', 'C-OPG + MC', 'C-OPG',
         'Soaked-OPG + CG', 'Soaked-OPG + MC', 'Positive Control', 'Soaked-OPG']
# Colour families: Defect blue (as in the main figures), controls grey,
# C-OPG family orange, Soaked-OPG family green. Every bar is also labelled.
COLOR = {'Defect': '#2a78d6', 'CG': '#a3a29c', 'MC': '#86857f', 'Positive Control': '#52514e',
         'C-OPG + CG': '#f2a07e', 'C-OPG + MC': '#b8481b', 'C-OPG': '#eb6834',
         'Soaked-OPG + CG': '#7fd3b2', 'Soaked-OPG + MC': '#0f7a52', 'Soaked-OPG': '#1baf7a'}
SHORT = {'Defect': 'Defect', 'CG': 'CG', 'MC': 'MC', 'Positive Control': 'Pos. ctrl',
         'C-OPG + CG': 'C-OPG\n+ CG', 'C-OPG + MC': 'C-OPG\n+ MC', 'C-OPG': 'C-OPG',
         'Soaked-OPG + CG': 'S-OPG\n+ CG', 'Soaked-OPG + MC': 'S-OPG\n+ MC', 'Soaked-OPG': 'S-OPG'}


def load():
    df = pd.read_csv(FIG.DB_CSV, dtype={'subject': str})
    ex = df[df.scan_type == 'ex vivo'].set_index('subject')
    num = ex.select_dtypes('number').columns
    rows = []
    for t, groups in RABBITS.items():
        for g, rabbits in groups.items():
            for rab, scans in rabbits.items():
                missing = [s for s in scans if s not in ex.index]
                if missing:
                    raise SystemExit(f'scans not in the spreadsheet: {missing}')
                r = ex.loc[scans, num].mean().to_dict()
                r.update(timepoint=t, group=g, animal=rab, scans='+'.join(scans), n_scans=len(scans),
                         roi='half cylinder' if t == 6 else 'full cylinder')
                rows.append(r)
    a = pd.DataFrame(rows)
    a = a[~a.group.isin(DROP_GROUPS)]
    a['gi'] = a.group.map({g: i for i, g in enumerate(ORDER)})
    return a.sort_values(['timepoint', 'gi', 'animal']).reset_index(drop=True)


def active(a):
    return [g for g in ORDER if (a.group == g).any()]


def groups_at(a, t):
    return [g for g in ORDER if (a[a.timepoint == t].group == g).any()]


def bars(a):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(FIG.BAR_EX), figsize=(7.09, 2.7))
    rng = np.random.default_rng(7)
    for ax, (col, lab, sc), letter in zip(axes.flat, FIG.BAR_EX, 'abcdefgh'):
        x0, ticks, tlabs, seps = 0.0, [], [], []
        for t in (3, 6):
            sub = a[a.timepoint == t]
            gs = groups_at(a, t)
            FIG.GROUPS[:] = gs
            res = FIG.compare(sub, col, sc)
            pos = {}
            for i, g in enumerate(gs):
                v = sub.loc[sub.group == g, col].dropna().to_numpy() * sc
                x = x0 + i
                pos[g] = x
                m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
                ax.bar(x, m, 0.72, color=COLOR[g], alpha=0.35, edgecolor=COLOR[g], lw=0.8, zorder=2)
                ax.errorbar(x, m, yerr=sd, color=FIG.INK, lw=0.7, capsize=1.8, capthick=0.7, zorder=3)
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=COLOR[g],
                           edgecolor='white', lw=0.4, zorder=4)
                ticks.append(x); tlabs.append(SHORT[g].replace('\n', ' '))
            # Tukey brackets vs Defect only
            lo, hi = ax.get_ylim()
            top = np.nanmax(sub[col].to_numpy() * sc)
            y = top
            for (g1, g2), p in res['tukey'].items():
                if 'Defect' not in (g1, g2) or not FIG.stars(p):
                    continue
                other = g2 if g1 == 'Defect' else g1
                y += (hi - lo) * 0.07
                ax.plot([pos['Defect'], pos['Defect'], pos[other], pos[other]],
                        [y, y + (hi - lo) * 0.02, y + (hi - lo) * 0.02, y], color=FIG.INK, lw=0.5)
                ax.text((pos['Defect'] + pos[other]) / 2, y + (hi - lo) * 0.02, FIG.stars(p),
                        ha='center', va='bottom', fontsize=6)
            ax.text(x0 + (len(gs) - 1) / 2, 1.0, f'{t} months', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=6.5, color=FIG.INK2)
            seps.append(x0 + len(gs) - 0.5 + 0.5)
            x0 += len(gs) + 1
        ax.axvline(seps[0], color=FIG.AXIS, lw=0.6)
        ax.set_xticks(ticks, tlabs, fontsize=5.2, rotation=50, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlim(-0.7, x0 - 1.3)
        ax.set_ylabel(lab, fontsize=6.5)
        ax.yaxis.grid(True, color=FIG.GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight='bold')
    fig.tight_layout(h_pad=1.6, w_pad=1.2)
    FIG.save(fig, 'Fig2_bars', SUB)
    plt.close(fig)


# Heatmap features, grouped into families (rows, top to bottom).
FAMILIES = [
    ('Mineral amount', [('spec_bvtv_226mgha', 'BV/TV (226 mg HA/cm³)'),
                        ('to_ring_spec_bvtv_226mgha', 'Defect : reference BV/TV'),
                        ('spec_bmd_mgha', 'BMD'), ('bvtv_fixed', 'BV/TV (226 HU)'),
                        ('mean_hu', 'Mean HU'), ('p90_hu', '90th percentile HU')]),
    ('Mineral density', [('tmd_mgha', 'TMD'), ('bone_mean_hu', 'Bone mean HU'),
                         ('particle_mean_mgha', 'Particle density')]),
    ('Mineral structure', [('particle_count', 'Particles (n)'),
                           ('particle_volume_mean_mm3', 'Particle volume'),
                           ('particle_eqdiam_mean_mm', 'Particle diameter'),
                           ('particle_largest_fraction', 'Largest-particle fraction'),
                           ('particle_nn_dist_mean_mm', 'Particle spacing'),
                           ('bone_components', 'Bone components'),
                           ('bone_volume_to_surface_mm', 'BV/BS (thickness)'),
                           ('bone_sphericity', 'Sphericity'), ('bone_solidity', 'Solidity')]),
    ('Intensity distribution', [('std_hu', 'SD HU'), ('skewness', 'Skewness'),
                                ('kurtosis', 'Kurtosis'), ('entropy_bits', 'Entropy')]),
    ('Texture', [('glcm_contrast', 'GLCM contrast'), ('glcm_homogeneity', 'GLCM homogeneity'),
                 ('glcm_correlation', 'GLCM correlation'), ('glcm_entropy_bits', 'GLCM entropy'),
                 ('glrlm_run_percentage', 'GLRLM run %'), ('glszm_zone_percentage', 'GLSZM zone %'),
                 ('glszm_large_area_emphasis', 'GLSZM large area'),
                 ('ngtdm_coarseness', 'NGTDM coarseness'), ('ngtdm_busyness', 'NGTDM busyness')]),
]
SOAKED = {3: 'Soaked-OPG + MC', 6: 'Soaked-OPG'}
C_OPG = {3: 'C-OPG + MC', 6: 'C-OPG'}
KEY_MIN_G = 0.5      # Hedges' g, Soaked-OPG vs Defect and vs C-OPG, at BOTH timepoints


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if not sp:
        return np.nan
    return (x.mean() - y.mean()) / sp * (1 - 3 / (4 * (nx + ny) - 9))


def key_features(a):
    """Features where Soaked-OPG exceeds Defect and its C-OPG counterpart at
    both timepoints (Hedges' g >= KEY_MIN_G in all four comparisons)."""
    rows = []
    for fam, feats in FAMILIES:
        for k, lab in feats:
            col = f'core_{k}'
            r = {'family': fam, 'feature': lab}
            for t in (3, 6):
                sub = a[a.timepoint == t]
                v = lambda grp: sub.loc[sub.group == grp, col]
                r[f'g_vs_Defect_{t}mo'] = hedges_g(v(SOAKED[t]), v('Defect'))
                r[f'g_vs_C-OPG_{t}mo'] = hedges_g(v(SOAKED[t]), v(C_OPG[t]))
                r[f'g_vs_MC_{t}mo'] = hedges_g(v(SOAKED[t]), v('MC'))
            gs = [r[f'g_vs_{c}_{t}mo'] for c in ('Defect', 'C-OPG') for t in (3, 6)]
            r['min_g'] = np.nanmin(gs)
            r['key'] = bool(np.nanmin(gs) >= KEY_MIN_G)
            rows.append(r)
    out = pd.DataFrame(rows).sort_values('min_g', ascending=False)
    out.round(3).to_csv(FIG.OUT / SUB / 'Soaked-OPG_feature_ranking.csv', index=False)
    return set(out.loc[out.key, 'feature'])


def heatmap(a, t):
    """Fig 3a/3b: shared layout from 9_opg_figures.grouped_heatmap, z-scored within the
    timepoint against the treatment groups (Positive Control shown but not in the scale)."""
    sub = a[a.timepoint == t]
    FIG.grouped_heatmap(sub, groups_at(a, t), FAMILIES, COLOR,
                        {g: SHORT[g].replace('/\n', '/').replace('\n', ' ') for g in COLOR},
                        f'Fig3{"a" if t == 3 else "b"}_heatmap_{t}mo', SUB,
                        sub[sub.group != 'Positive Control'],
                        'Scaled per\nfeature across\nthe treatment\ngroups')


def table(a):
    import matplotlib.pyplot as plt
    rows = []
    for t in (3, 6):
        sub = a[a.timepoint == t]
        gs = groups_at(a, t)
        FIG.GROUPS[:] = gs
        for col, lab, sc in FIG.EX_METRICS:
            res = FIG.compare(sub, col, sc)
            r = {'timepoint': f'{t} months', 'metric': lab}
            for g in active(a):
                r[g] = FIG.fmt_group(res['groups'].get(g)) if g in gs else ''
            r['ANOVA p'] = FIG.fmt_p(res['anova_p']); r['Kruskal–Wallis p'] = FIG.fmt_p(res['kw_p'])
            r['Tukey p < 0.05'] = '; '.join(f'{x} vs {y}: {FIG.fmt_p(p)}' for (x, y), p in res['tukey'].items()
                                             if p < 0.05) or '–'
            rows.append(r)
    t1 = pd.DataFrame(rows)
    d = FIG.OUT / SUB
    d.mkdir(parents=True, exist_ok=True)
    notes = [
        'Values are mean ± SD (n = rabbits; rabbits scanned more than once are averaged). Groups compared within each '
        'timepoint by one-way ANOVA with Tukey HSD; Kruskal–Wallis as non-parametric check. n = 2–4 per group: p-values are exploratory.',
        '3 months: 2021 cohort (lab sheets #182/#184), intact trephined discs, full-cylinder ROI (10 mm core, 14–18 mm ring, 8 mm tall). '
        '6 months: UCLA cohort (#262, = in vivo MR52519–MR52530), halved specimens cut through the defect, half-cylinder ROI. '
        'Timepoints are not directly comparable (different cohorts, ROI geometry and preparation).',
        'SCANCO µCT 15 µm, analysed at 60 µm; mineral densities from the scanner HA calibration; threshold 226 mg HA/cm³. '
        'BV/TV and BMD measured inside the specimen. Particles: Gaussian σ = 0.8 voxel, ≥ 0.01 mm³, 26-connected.',
        'Excluded: ' + '; '.join(f'{k} ({v})' for k, v in EXCLUDED.items()) + '.'
        + (f' Groups left out of the analysis: {", ".join(DROP_GROUPS)}.' if DROP_GROUPS else ''),
    ]
    t1.to_csv(d / 'Table1_group_summary.csv', index=False)
    with pd.ExcelWriter(d / 'Table1_group_summary.xlsx') as xw:
        t1.to_excel(xw, sheet_name='Table 1', index=False)
        pd.DataFrame({'Notes': notes}).to_excel(xw, sheet_name='Notes', index=False)
    (d / 'Table1_notes.txt').write_text('\n'.join(notes) + '\n')
    # rendered (landscape: 10 group columns)
    grp = active(a)
    show = t1[['timepoint', 'metric'] + grp + ['ANOVA p']].copy()
    show.columns = ['Timepoint', 'Metric'] + [SHORT[g].replace('\n', ' ') for g in grp] + ['ANOVA p']
    lines = sum(len(textwrap.wrap(n, 210)) for n in notes)
    th, nh = 0.2 * (len(show) + 1.4), 0.1 * lines + 0.1
    fig = plt.figure(figsize=(10.5, th + nh))
    gsp = fig.add_gridspec(2, 1, height_ratios=[th, nh], hspace=0.02)
    ax = fig.add_subplot(gsp[0]); ax.axis('off')
    nax = fig.add_subplot(gsp[1]); nax.axis('off')
    tb = ax.table(cellText=show.to_numpy(), colLabels=show.columns, bbox=[0, 0, 1, 1], cellLoc='center',
                  colWidths=[0.07, 0.17] + [0.71 / len(grp)] * len(grp) + [0.05])
    tb.auto_set_font_size(False); tb.set_fontsize(5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(FIG.GRID); cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(fontweight='bold'); cell.set_facecolor('#f0efec'); continue
        cell.set_facecolor('white' if show.iloc[r - 1, 0].startswith('3') else '#fafaf8')
        if c == len(show.columns) - 1:
            pv = show.iloc[r - 1, -1]
            if pv != '–' and (pv == '<0.001' or float(pv) < 0.05):
                cell.set_text_props(fontweight='bold')
    nax.text(0, 1, '\n'.join(sum((textwrap.wrap(n, 210) for n in notes), [])), transform=nax.transAxes,
             fontsize=5.2, color=FIG.INK2, va='top')
    FIG.save(fig, 'Table1_group_summary', SUB)
    plt.close(fig)
    return t1


def scans(a):
    import matplotlib.pyplot as plt
    B = _load('_b', '9b_opg_scan_figure.py')
    B.FIG.COLOR.update(COLOR)
    df = pd.read_csv(FIG.DB_CSV, dtype={'subject': str})
    rows_t = [3, 6]
    ncol = max(len(groups_at(a, t)) for t in rows_t)
    cells = [(ri, ci, t, g) for ri, t in enumerate(rows_t) for ci, g in enumerate(groups_at(a, t))]
    fig = plt.figure(figsize=(7.09, 7.09 / ncol * len(rows_t) * 1.06 + 0.3))
    gsp = fig.add_gridspec(len(rows_t), ncol + 1, width_ratios=[1] * ncol + [0.06], wspace=0.05, hspace=0.14,
                           top=0.95, bottom=0.02, left=0.04, right=0.94)
    im = None
    for ri, ci, t, g in cells:
        sub = a[(a.timepoint == t) & (a.group == g)]
        cols = ['core_spec_bvtv_226mgha', 'core_spec_bmd_mgha', 'core_tmd_mgha']
        sd = a[a.timepoint == t][cols].std(ddof=1)
        if (t, g) in REP_SCANS:
            scan = REP_SCANS[(t, g)]
            r = sub[sub.scans.str.split('+').apply(lambda x: scan in x)].iloc[0]
        else:
            r = sub.loc[((sub[cols] - sub[cols].median()) / sd).abs().sum(axis=1).idxmin()]
            scan = r.scans.split('+')[0]
        row = df[(df.scan_type == 'ex vivo') & (df.subject == scan)].iloc[0]
        print(f'  {t} mo {g}: {r.animal} (scan {scan})')
        top, uv, side, u, tt, cut_deg = B.views(FIG.BASE / row.scan_dir, FIG.BASE / row.roi_series_dir, True)
        ax = fig.add_subplot(gsp[ri, ci])
        im = B._top_panel(ax, top, uv, B.EX_WIN, g, f'{r.animal} ({scan})', ri == 0 and ci == 0, cut_deg)
        ax.set_title(f'{g}', fontsize=6.5, fontweight='bold', color=COLOR[g], pad=2)
        if ci == 0:
            ax.set_ylabel(f'{t} months', fontsize=7)
    cax = fig.add_subplot(gsp[:, ncol])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
    cb.set_label('mg HA/cm³', fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2)
    FIG.save(fig, 'Fig1_representative_scans', SUB)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', action='store_true')
    args = ap.parse_args()
    FIG.setup_style()
    FIG.COLOR.update(COLOR)
    a = load()
    print(a.groupby(['timepoint', 'group'], sort=False).size().to_string())
    (FIG.OUT / SUB).mkdir(parents=True, exist_ok=True)
    a[['timepoint', 'group', 'animal', 'scans', 'n_scans', 'roi'] + [c for c, _, _ in FIG.EX_METRICS]] \
        .to_csv(FIG.OUT / SUB / 'analysis_set.csv', index=False)
    table(a)
    bars(a)
    key = key_features(a)
    print('exploratory effect-size ranking written; features with g >= 0.5 in all four comparisons:', sorted(key))
    for t in (3, 6):
        heatmap(a, t)
    if args.images:
        scans(a)
    print(f'outputs in {FIG.OUT / SUB}')


if __name__ == '__main__':
    main()
