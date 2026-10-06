#!/usr/bin/env python3
"""Defect vs C-OPG/SPDP-OPG vs Soaked-OPG — publication tables and figures.

Reads ../radiomics_database/radiomics_database.csv (never writes it) and
builds an animal-level analysis set from EVERY scan of the three groups,
with the lab-sheet corrections applied (2026-10-05):

  * C-OPG and SPDP-OPG are the same treatment ("C-OPG/SPDP-OPG").
  * Ex vivo 2021 cohort (#182/#184 sheets) is NOT used (removed 2026-10-05 at
    the user's request): it had no plain C-OPG or S-OPG group, only Defect
    controls. Its low-res "C-OPG"/"Soaked-OPG" uploads 4131-4136 are
    duplicate scans of +CG/+MC combination specimens.
  * Ex vivo 6-month UCLA cohort (#262 sheet): specimens 5780-5790 are the in
    vivo MR52521-MR52530 animals; 5783 is a crossed-out repeat of 5782
    (rabbit x52523) and is dropped. The specimens are halved calvaria cut
    through the defect, measured with half-cylinder ROIs (11_halfcut_exvivo_roi.py).
  * In vivo: 3 and 6 months (the OPG groups have no 9-month scans, so the
    earlier cohort's 9-month Defect scans are left out, 2026-10-05). 6-month MR525xx ROIs were placed
    by raw network inference (not the validated registration path) and are
    drawn hollow / hatched and footnoted.

Ex vivo values use the SCANCO HA calibration (mg HA/cm^3, 226 mg HA/cm^3
threshold); in vivo values are HU at the 226 HU study threshold. The two
modalities are never pooled.

Statistics: per timepoint/cohort, one-way ANOVA with Tukey HSD post hoc
(Kruskal-Wallis reported alongside as a non-parametric check). Group sizes
are small (n = 2-5); treat p-values as exploratory.

    python 9_opg_figures.py            # tables + data figures (seconds)
    python 9_opg_figures.py --images   # also the side-by-side scan figure (minutes)

Outputs: ../figures/opg_comparison/ (PDF + 600 dpi PNG + TIFF per figure,
tables as CSV/XLSX, the animal-level analysis set as CSV).
"""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

REPO_DIR = Path(__file__).resolve().parent
DATA_ROOT = REPO_DIR.parent
BASE = DATA_ROOT.parent                       # roi_series_dir / scan_dir are relative to this
DB_CSV = DATA_ROOT / 'radiomics_database' / 'radiomics_database.csv'
OUT = DATA_ROOT / 'figures' / 'opg_comparison'

GROUPS = ['Defect', 'C-OPG/SPDP-OPG', 'Soaked-OPG']
GROUP_OF = {'Defect': 'Defect', 'C-OPG': 'C-OPG/SPDP-OPG', 'SPDP': 'C-OPG/SPDP-OPG',
            'SPDP-OPG': 'C-OPG/SPDP-OPG', 'Soaked-OPG': 'Soaked-OPG'}
# Validated categorical slots 1-3 (dataviz reference palette, all-pairs CVD-safe).
COLOR = {'Defect': '#2a78d6', 'C-OPG/SPDP-OPG': '#eb6834', 'Soaked-OPG': '#1baf7a'}
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
DIVERGING = ['#184f95', '#3987e5', '#9ec5f4', '#f0efec', '#f2a7a6', '#e34948', '#a3292a']

# Ex vivo animal map (lab sheets #182/#184/#262).
EXVIVO_ANIMALS = {
    # 6-month UCLA cohort = in vivo MR525xx animals
    'x52523': ('6 months', 'Defect', ['5782']),          # 5783 = repeat scan, dropped
    'x52524': ('6 months', 'Defect', ['5784']),
    'x52521': ('6 months', 'C-OPG/SPDP-OPG', ['5780']),
    'x52522': ('6 months', 'C-OPG/SPDP-OPG', ['5781']),
    'x52525': ('6 months', 'C-OPG/SPDP-OPG', ['5785']),
    'x52526': ('6 months', 'C-OPG/SPDP-OPG', ['5786']),
    'x52527': ('6 months', 'Soaked-OPG', ['5787']),
    'x52528': ('6 months', 'Soaked-OPG', ['5788']),
    'x52529': ('6 months', 'Soaked-OPG', ['5789']),
    'x52530': ('6 months', 'Soaked-OPG', ['5790']),
}
EXCLUDED = {
    '5783': 'repeat scan of x52523 (= 5782), crossed out on the #262 sheet',
    '4131': 'low-res duplicate of S-OPG CG1 (re-scanned as 4138); combination group',
    '4134': 'low-res duplicate of S-OPG CG2 (re-scanned as 4141); combination group',
    '4136': 'low-res duplicate of S-OPG CG3 (re-scanned as 4143); combination group',
    '4132': 'low-res duplicate of C-OPG CG2 (re-scanned as 4139); combination group',
    '4133': 'low-res duplicate of C-OPG MC2 (re-scanned as 4140); combination group',
    '4135': 'low-res duplicate of C-OPG CG4 (re-scanned as 4142); combination group',
}

# (column, label, unit-scale)
EX_METRICS = [
    ('core_spec_bvtv_226mgha', 'Defect BV/TV (%)', 100),
    ('core_to_ring_spec_bvtv_226mgha', 'Defect : reference BV/TV', 1),
    ('core_spec_bmd_mgha', 'Defect BMD (mg HA/cm³)', 1),
    ('core_tmd_mgha', 'Defect TMD (mg HA/cm³)', 1),
    ('core_particle_count', 'Mineralised particles (n)', 1),
    ('core_particle_volume_mean_mm3', 'Mean particle volume (mm³)', 1),
    ('core_particle_mean_mgha', 'Particle density (mg HA/cm³)', 1),
    ('core_particle_nn_dist_mean_mm', 'Particle spacing (mm)', 1),
]
IV_METRICS = [
    ('core_bvtv_fixed', 'Defect BV/TV (%)', 100),
    ('core_to_ring_bvtv_fixed', 'Defect : reference BV/TV', 1),
    ('core_mean_hu', 'Defect mean density (HU)', 1),
    ('core_bone_mean_hu', 'Defect bone density (HU)', 1),
]
HEAT_COMMON = [
    ('mean_hu', 'Mean HU'), ('std_hu', 'SD HU'), ('p90_hu', 'P90 HU'),
    ('skewness', 'Skewness'), ('kurtosis', 'Kurtosis'), ('entropy_bits', 'Entropy'),
    ('bvtv_fixed', 'BV/TV (226 HU)'), ('bone_mean_hu', 'Bone mean HU'),
    ('bone_components', 'Bone components'), ('bone_volume_to_surface_mm', 'BV/BS'),
    ('bone_sphericity', 'Sphericity'), ('bone_solidity', 'Solidity'),
    ('glcm_contrast', 'GLCM contrast'), ('glcm_homogeneity', 'GLCM homogeneity'),
    ('glcm_correlation', 'GLCM correlation'), ('glcm_entropy_bits', 'GLCM entropy'),
    ('glrlm_run_percentage', 'GLRLM run %'), ('glszm_zone_percentage', 'GLSZM zone %'),
    ('glszm_large_area_emphasis', 'GLSZM large area'),
    ('ngtdm_coarseness', 'NGTDM coarseness'), ('ngtdm_busyness', 'NGTDM busyness'),
]
HEAT_EXVIVO = [
    ('spec_bvtv_226mgha', 'BV/TV (226 mg HA)'), ('spec_bmd_mgha', 'BMD'), ('tmd_mgha', 'TMD'),
    ('particle_count', 'Particles (n)'), ('particle_volume_mean_mm3', 'Particle vol.'),
    ('particle_eqdiam_mean_mm', 'Particle diam.'), ('particle_largest_fraction', 'Largest particle frac.'),
    ('particle_mean_mgha', 'Particle density'), ('particle_nn_dist_mean_mm', 'Particle spacing'),
] + HEAT_COMMON

CAPTION_NOTES = {}
MIN_CORE_IN_SPECIMEN = 0.5   # ex vivo: exclude cut specimens missing most of the defect
IV_TIMEPOINTS = (3, 6)   # no 9-month scans exist for the OPG groups
VALIDATED = ('network (3-month, direct)', 'registration from 3-month ROI',
             'manual annotation (ground truth)')


# ───────────────────────────────────────────────────────────── data

def load_sets():
    df = pd.read_csv(DB_CSV, dtype={'subject': str})
    num = df.select_dtypes('number').columns

    ex = df[df.scan_type == 'ex vivo'].set_index('subject')
    rows = []
    for animal, (cohort, group, scans) in EXVIVO_ANIMALS.items():
        sub = ex.loc[scans]
        r = sub[num].mean().to_dict()
        r.update(animal=animal, cohort=cohort, group=group, scans='+'.join(scans),
                 n_scans=len(scans))
        if len(scans) == 2:     # scan-rescan repeatability on the headline metric
            a, b = sub['core_bvtv_226mgha']
            r['rescan_abs_diff_bvtv'] = abs(a - b)
        rows.append(r)
    exs = pd.DataFrame(rows)
    low = exs.core_specimen_fraction < MIN_CORE_IN_SPECIMEN
    for _, r in exs[low].iterrows():
        EXCLUDED[r.scans] = (f'only {100 * r.core_specimen_fraction:.0f}% of the defect core lies inside '
                             'the specimen (cut through/beside the defect, #262 batch)')
    exs = exs[~low].reset_index(drop=True)

    iv = df[(df.scan_type == 'in vivo') & df.treatment.isin(GROUP_OF)].copy()
    iv['group'] = iv.treatment.map(GROUP_OF)
    iv['animal'] = iv.subject
    iv['placement'] = np.where(iv.placement_method.isin(VALIDATED), 'validated',
                       np.where(iv.manually_adjusted.astype(str) == 'True',
                                'manual adjustment', 'unvalidated (raw network)'))
    iv['timepoint_months'] = iv.timepoint_months.astype(int)
    iv = iv[iv.timepoint_months.isin(IV_TIMEPOINTS)]
    return exs, iv


# ───────────────────────────────────────────────────────────── stats

def compare(frame, col, scale=1):
    """One-way ANOVA + Tukey HSD + Kruskal-Wallis across the groups present."""
    data = {g: frame.loc[frame.group == g, col].dropna().to_numpy() * scale for g in GROUPS}
    data = {g: v for g, v in data.items() if len(v)}
    out = {'groups': data, 'anova_p': np.nan, 'kw_p': np.nan, 'tukey': {}}
    usable = [v for v in data.values() if len(v) >= 2]
    if len(usable) >= 2 and len(usable) == len(data):
        vals = list(data.values())
        out['anova_p'] = float(stats.f_oneway(*vals).pvalue)
        try:
            out['kw_p'] = float(stats.kruskal(*vals).pvalue)
        except ValueError:
            pass
        tk = stats.tukey_hsd(*vals)
        names = list(data)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                out['tukey'][(names[i], names[j])] = float(tk.pvalue[i, j])
    return out


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return '–'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def fmt_group(v):
    if v is None:
        return '–'
    m = v.mean()
    a = max(abs(m), v.std(ddof=1) if len(v) > 1 else 0)
    d = 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3
    if len(v) < 2:
        return f'{m:.{d}f} (n=1)'
    return f'{m:.{d}f} ± {v.std(ddof=1):.{d}f} (n={len(v)})'


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else None


# ───────────────────────────────────────────────────────────── style

def setup_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 7.5,
        'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03, 'hatch.linewidth': 0.6,
    })


def save(fig, name, sub):
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(d / f'{name}.{ext}', facecolor='white')
    fig.savefig(d / f'{name}.tiff', facecolor='white', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    print(f'  wrote {sub}/{name}.pdf/.png/.tiff')


MARK = {'2021 cohort': 's', '6 months': 'o'}
JIT = np.random.default_rng(7)


def bar_panel(ax, frame, col, scale, slots, ylabel, hollow_mask=None, hatch_slots=()):
    """Bars = mean ± SD per (slot, group); individual animals overlaid.

    slots: list of (slot_key, slot_label, frame_mask). Groups are dodged
    within each slot; only groups with data in the slot are drawn.
    """
    width = 0.24
    xt, xl = [], []
    centres = {}
    for si, (key, label, mask) in enumerate(slots):
        present = [g for g in GROUPS if (mask & (frame.group == g) & frame[col].notna()).any()]
        offs = (np.arange(len(present)) - (len(present) - 1) / 2) * (width + 0.04)
        for g, o in zip(present, offs):
            sel = mask & (frame.group == g)
            v = frame.loc[sel, col].dropna().to_numpy() * scale
            x = si + o
            centres[(key, g)] = x
            m, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
            hatched = key in hatch_slots
            ax.bar(x, m, width, color=COLOR[g] if not hatched else 'white',
                   edgecolor=COLOR[g], linewidth=0.8, alpha=0.35 if not hatched else 1,
                   hatch='/////' if hatched else None, zorder=2)
            ax.errorbar(x, m, yerr=sd, color=INK, lw=0.7, capsize=2, capthick=0.7, zorder=3)
            xs = x + JIT.uniform(-width * 0.28, width * 0.28, len(v))
            hm = (hollow_mask[sel].to_numpy()[frame.loc[sel, col].notna().to_numpy()]
                  if hollow_mask is not None else np.zeros(len(v), bool))
            for xi, yi, h in zip(xs, v, hm):
                ax.scatter(xi, yi, s=11, marker=MARK.get(key, 'o'), zorder=4,
                           facecolor='white' if h else COLOR[g], edgecolor=COLOR[g]
                           if h else 'white', linewidth=0.8 if h else 0.5)
        xt.append(si); xl.append(label)
    ax.set_xticks(xt, xl)
    ax.set_xlim(-0.6, len(slots) - 0.4)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    return centres


def brackets(ax, centres, slot_key, res):
    """Draw Tukey brackets for p < 0.05 within one slot."""
    sig = [(a, b, p) for (a, b), p in res['tukey'].items() if stars(p)]
    if not sig:
        return
    lo, hi = ax.get_ylim()
    top = max(max(v) for v in res['groups'].values())
    step = (hi - lo) * 0.08
    y = top + step * 0.6
    for a, b, p in sorted(sig, key=lambda t: abs(centres[(slot_key, t[0])] - centres[(slot_key, t[1])])):
        x1, x2 = centres[(slot_key, a)], centres[(slot_key, b)]
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y], color=INK, lw=0.6)
        ax.text((x1 + x2) / 2, y + step * 0.3, stars(p), ha='center', va='bottom', fontsize=7)
        y += step * 1.1
    ax.set_ylim(lo, max(hi, y + step * 0.3))


def legend_handles(include_cohort=True, hollow=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(facecolor=COLOR[g], alpha=0.35, edgecolor=COLOR[g], label=g) for g in GROUPS]
    if include_cohort:
        h += [Line2D([], [], ls='', marker='s', color=MUTED, ms=4, label='2021 cohort (rabbit; mean of 2 scans)'),
              Line2D([], [], ls='', marker='o', color=MUTED, ms=4, label='6-month UCLA cohort')]
    if hollow:
        h += [Line2D([], [], ls='', marker='o', mfc='white', mec=MUTED, ms=4,
                     label='ROI placed by raw network inference (unvalidated)'),
              Patch(facecolor='white', edgecolor=MUTED, hatch='/////',
                    label='Hatched: timepoint includes unvalidated ROIs')]
    return h


# ───────────────────────────────────────────────────────────── figures

def fig_exvivo_bars(exs):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(7.09, 3.9))
    exs = exs[exs.cohort == '6 months']
    slots = [('6 months', '6 months', exs.cohort == '6 months')]
    for ax, (col, lab, sc), letter in zip(axes.flat, EX_METRICS, 'abcdefgh'):
        c = bar_panel(ax, exs, col, sc, slots, lab)
        brackets(ax, c, '6 months', compare(exs[exs.cohort == '6 months'], col, sc))
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.text(-0.32, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight='bold')
    fig.legend(handles=legend_handles(include_cohort=False), loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.2, h_pad=1.4)
    save(fig, 'Fig2_bars', 'ex_vivo')
    plt.close(fig)


def fig_invivo_bars(iv):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(7.09, 2.3))
    slots = [(t, str(t), iv.timepoint_months == t) for t in IV_TIMEPOINTS]
    hollow = iv.placement.str.startswith('unvalidated')
    unval_t = sorted(iv.loc[hollow, 'timepoint_months'].unique())
    for ax, (col, lab, sc), letter in zip(axes.flat, IV_METRICS, 'abcd'):
        c = bar_panel(ax, iv, col, sc, slots, lab, hollow_mask=hollow, hatch_slots=tuple(unval_t))
        for t in (3, 6):
            if t not in unval_t:
                brackets(ax, c, t, compare(iv[iv.timepoint_months == t], col, sc))
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_xlabel('Months after surgery')
        ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight='bold')
    fig.legend(handles=legend_handles(include_cohort=False, hollow=bool(unval_t)), loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12), columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.4)
    save(fig, 'Fig2_bars', 'in_vivo')
    plt.close(fig)


def fig_longitudinal(iv, exs):
    """Same animals over time: in vivo 3 -> 6 months (HU, 226 HU) per rabbit."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.2), sharey=True)
    for ax, g, letter in zip(axes, GROUPS, 'abc'):
        sub = iv[iv.group == g]
        ends = []
        for a, s in sub.groupby('animal'):
            s = s.sort_values('timepoint_months')
            ax.plot(s.timepoint_months, s.core_to_ring_bvtv_fixed, color=COLOR[g], lw=1, alpha=0.8)
            for _, r in s.iterrows():
                h = r.placement.startswith('unvalidated')
                ax.scatter(r.timepoint_months, r.core_to_ring_bvtv_fixed, s=12, zorder=3,
                           facecolor='white' if h else COLOR[g], edgecolor=COLOR[g], lw=0.8)
            last = s.iloc[-1]
            ends.append([last.timepoint_months, last.core_to_ring_bvtv_fixed, a])
        gap = 0.04                       # minimum label separation (data units)
        for t in {e[0] for e in ends}:
            col = sorted([e for e in ends if e[0] == t], key=lambda e: e[1])
            ys = [e[1] for e in col]
            for i in range(1, len(ys)):
                ys[i] = max(ys[i], ys[i - 1] + gap)
            for e, y in zip(col, ys):
                ax.text(t + 0.15, y, e[2], fontsize=5.5, color=INK2, va='center')
        ax.set_title(g, color=INK)
        ax.set_xticks(list(IV_TIMEPOINTS)); ax.set_xlim(2.3, 7.4)
        ax.set_xlabel('Months after surgery')
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.text(-0.27 if letter == 'a' else -0.06, 1.06, letter, transform=ax.transAxes,
                fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Defect : reference BV/TV (in vivo)')
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls='', marker='o', color=MUTED, ms=4,
                 label='ROI: network at 3 months; registration or manual adjustment later')]
    if iv.placement.str.startswith('unvalidated').any():
        hs.append(Line2D([], [], ls='', marker='o', mfc='white', mec=MUTED, ms=4,
                         label='ROI placed by raw network inference (unvalidated)'))
    fig.legend(handles=hs, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 'Fig3_longitudinal', 'in_vivo')
    plt.close(fig)


def heatmap(frame, feats, row_label, strips, name, title_note, sub):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cols = [f'core_{k}' for k, _ in feats if f'core_{k}' in frame]
    labels = [l for k, l in feats if f'core_{k}' in frame]
    X = frame[cols].astype(float)
    keep = X.std(ddof=1) > 0
    X, labels = X.loc[:, keep], [l for l, k in zip(labels, keep) if k]
    Z = ((X - X.mean()) / X.std(ddof=1)).clip(-2.5, 2.5)
    cmap = LinearSegmentedColormap.from_list('div', DIVERGING)
    cmap.set_bad('#ffffff')
    n_r, n_c = Z.shape
    fig_h = 0.9 + 0.13 * n_r
    fig = plt.figure(figsize=(7.09, fig_h))
    ns = len(strips)
    gs = fig.add_gridspec(1, ns + 2, width_ratios=[0.25] * ns + [n_c * 0.33, 0.15], wspace=0.04)
    ax = fig.add_subplot(gs[0, ns])
    im = ax.imshow(Z.to_numpy(), cmap=cmap, vmin=-2.5, vmax=2.5, aspect='auto',
                   interpolation='nearest')
    ax.set_xticks(range(n_c), labels, rotation=55, ha='right', rotation_mode='anchor', fontsize=6)
    ax.set_yticks([]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(n_c + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_r + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='minor', length=0)
    # group boundaries
    g = frame.group.to_numpy()
    for i in range(1, n_r):
        if g[i] != g[i - 1]:
            ax.axhline(i - 0.5, color=INK, lw=0.8)
    for k, (skey, colors, stitle) in enumerate(strips):
        sax = fig.add_subplot(gs[0, k])
        sax.set_ylim(n_r - 0.5, -0.5)
        vals = frame[skey].to_numpy()
        for i, v in enumerate(vals):
            sax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1, color=colors.get(v, '#ffffff'), lw=0))
        sax.set_xlim(-0.5, 0.5); sax.set_xticks([0], [stitle], rotation=55, ha='right',
                                                  rotation_mode='anchor', fontsize=6)
        sax.tick_params(length=0)
        for s in sax.spines.values():
            s.set_visible(False)
        if k == 0:
            sax.set_yticks(range(n_r), frame[row_label].tolist(), fontsize=5.8)
        else:
            sax.set_yticks([])
    cax = fig.add_subplot(gs[0, ns + 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('z-score (per feature)', fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)
    CAPTION_NOTES[f'{sub}/{name}'] = title_note
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLOR[g], label=g) for g in GROUPS], loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=6.5)
    save(fig, name, sub)
    plt.close(fig)


def _summary_rows(frame, metrics, timepoint, dataset):
    rows = []
    for col, lab, sc in metrics:
        res = compare(frame, col, sc)
        r = {'dataset': dataset, 'timepoint': timepoint, 'metric': lab}
        for gname in GROUPS:
            r[gname] = fmt_group(res['groups'].get(gname))
        r['ANOVA p'] = fmt_p(res['anova_p']); r['Kruskal–Wallis p'] = fmt_p(res['kw_p'])
        r['Tukey p < 0.05'] = '; '.join(f'{a} vs {b}: {fmt_p(p)}' for (a, b), p in res['tukey'].items()
                                         if p < 0.05) or '–'
        rows.append(r)
    return rows


def tables(exs, iv):
    """One summary table per figure set, each with its own footnotes."""
    stats_note = ('Values are mean ± SD (n = animals). Groups compared within each timepoint: one-way ANOVA, '
                  'Tukey HSD post hoc; Kruskal–Wallis as non-parametric check. n = 4–5 per group: p-values are exploratory.')
    iv_rows = []
    for t in IV_TIMEPOINTS:
        sub = iv[iv.timepoint_months == t]
        unv = sub.placement.str.startswith('unvalidated').any()
        iv_rows += _summary_rows(sub, IV_METRICS, f'{t} months' + (' †' if unv else ''), 'in vivo (HU)')
    ex_rows = _summary_rows(exs, EX_METRICS, '6 months', 'ex vivo (mg HA/cm³)')
    notes = {
        'in_vivo': [
            stats_note,
            'In vivo µCT, 100 µm; 226 HU threshold; densities in HU (no mineral calibration). '
            'The OPG groups were scanned at 3 and 6 months only (harvested at 6 months).',
            ('† Timepoint includes ROIs placed by raw network inference (biased at non-3-month timepoints); interpret with caution.'
             if iv.placement.str.startswith('unvalidated').any() else
             'ROI placement: network at 3 months, or manually adjusted in the web app; later timepoints by rigid registration '
             'from the animal\'s 3-month ROI or manual adjustment. See placement_method in the database.'),
        ],
        'ex_vivo': [
            ('Values are mean ± SD (n = specimens; one per rabbit). Groups compared by one-way ANOVA, Tukey HSD post hoc; '
             'Kruskal–Wallis as non-parametric check. n = 2–4 per group: p-values are exploratory.'
             if exs.groupby('group').size().min() >= 2 else
             'Values are mean ± SD (n = specimens; one per rabbit). No statistical test: a group has n = 1.'),
            'Ex vivo SCANCO µCT, 15 µm, analysed at 60 µm; densities from the scanner HA calibration; bone/particle threshold '
            '226 mg HA/cm³. Particles: Gaussian σ = 0.8 voxel, ≥ 0.01 mm³, 26-connected.',
            '6-month UCLA cohort (= the in vivo MR52521–MR52530 animals). Specimens are halved calvaria cut through the middle '
            'of the defect: the ROI is the half of the template (half-disk core, half-annulus reference ring) on the specimen '
            'side of the cut, centred on the defect along the cut line; BV/TV and BMD are measured inside the specimen.',
            'Excluded: ' + '; '.join(f'{k} ({v})' for k, v in EXCLUDED.items()
                                     if k.startswith('57')) + '. The 2021 ex vivo cohort is not used (no plain C-OPG / S-OPG group).',
        ],
    }
    out = {}
    for sub, rows in (('in_vivo', iv_rows), ('ex_vivo', ex_rows)):
        t = pd.DataFrame(rows)
        if sub == 'ex_vivo' and exs.groupby('group').size().min() < 2:
            t = t.drop(columns=['ANOVA p', 'Kruskal–Wallis p', 'Tukey p < 0.05'])
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        t.drop(columns=['dataset']).to_csv(d / 'Table1_group_summary.csv', index=False)
        with pd.ExcelWriter(d / 'Table1_group_summary.xlsx') as xw:
            t.drop(columns=['dataset']).to_excel(xw, sheet_name='Table 1', index=False)
            pd.DataFrame({'Notes': notes[sub]}).to_excel(xw, sheet_name='Notes', index=False)
        (d / 'Table1_notes.txt').write_text('\n'.join(notes[sub]) + '\n')
        out[sub] = t
    return out, notes


def fig_table(tabs, notes):
    """Rendered table (vector) for each figure set."""
    import matplotlib.pyplot as plt
    import textwrap
    for sub, t in tabs.items():
        has_p = 'ANOVA p' in t
        cols = ['timepoint', 'metric'] + GROUPS + (['ANOVA p'] if has_p else [])
        s = t[cols].copy()
        s.columns = ['Timepoint', 'Metric'] + GROUPS + (['ANOVA p'] if has_p else [])
        lines = sum(len(textwrap.wrap(n, 150)) for n in notes[sub])
        th, nh = 0.19 * (len(s) + 1), 0.11 * lines + 0.1
        fig = plt.figure(figsize=(7.09, th + nh))
        gs = fig.add_gridspec(2, 1, height_ratios=[th, nh], hspace=0.02)
        ax = fig.add_subplot(gs[0]); ax.axis('off')
        nax = fig.add_subplot(gs[1]); nax.axis('off')
        widths = [0.1, 0.24, 0.17, 0.17, 0.17] + ([0.08] if has_p else [])
        tb = ax.table(cellText=s.to_numpy(), colLabels=s.columns, bbox=[0, 0, 1, 1],
                      cellLoc='center', colLoc='center', colWidths=widths)
        tb.auto_set_font_size(False); tb.set_fontsize(6.2)
        tps = list(dict.fromkeys(s.Timepoint))
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(GRID); cell.set_linewidth(0.4)
            if r == 0:
                cell.set_text_props(fontweight='bold', color=INK)
                cell.set_facecolor('#f0efec')
                continue
            if c == 1:
                cell._loc = 'left'; cell.PAD = 0.03
            cell.set_facecolor('white' if tps.index(s.iloc[r - 1, 0]) % 2 == 0 else '#fafaf8')
            if has_p and c == len(cols) - 1:
                pv = s.iloc[r - 1, -1]
                if pv != '–' and (pv == '<0.001' or float(pv) < 0.05):
                    cell.set_text_props(fontweight='bold')
        nax.text(0, 1, '\n'.join(sum((textwrap.wrap(n, 150) for n in notes[sub]), [])),
                 transform=nax.transAxes, fontsize=5.3, color=INK2, va='top')
        save(fig, 'Table1_group_summary', sub)
        plt.close(fig)


# ───────────────────────────────────────────────────────────── main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', action='store_true', help='also render the representative-scan figures')
    args = p.parse_args()
    setup_style()
    exs, iv = load_sets()
    (OUT / 'ex_vivo').mkdir(parents=True, exist_ok=True)
    (OUT / 'in_vivo').mkdir(parents=True, exist_ok=True)
    keep = ['animal', 'cohort', 'group', 'scans', 'core_specimen_fraction'] + [c for c, _, _ in EX_METRICS]
    exs[keep].to_csv(OUT / 'ex_vivo' / 'analysis_set.csv', index=False)
    iv[['animal', 'group', 'timepoint_months', 'placement', 'placement_method', 'feature_voxel_mm']
       + [c for c, _, _ in IV_METRICS]].sort_values(['group', 'animal', 'timepoint_months']) \
        .to_csv(OUT / 'in_vivo' / 'analysis_set.csv', index=False)
    print(f'ex vivo: {len(exs)} specimens; in vivo: {len(iv)} scans of {iv.animal.nunique()} animals')

    tabs, notes = tables(exs, iv)
    fig_table(tabs, notes)
    fig_exvivo_bars(exs)
    fig_invivo_bars(iv)
    fig_longitudinal(iv, exs)

    gcol = {g: COLOR[g] for g in GROUPS}
    exh = exs.assign(gi=exs.group.map({g: i for i, g in enumerate(GROUPS)})).sort_values(['gi', 'animal']) \
        .reset_index(drop=True)
    exh['row'] = exh.animal.str.replace('x', 'MR') + ' (' + exh.scans + ')'
    heatmap(exh, HEAT_EXVIVO, 'row', [('group', gcol, 'Group')], 'Fig3_heatmap',
            'Ex vivo defect core at 6 months, one row per specimen; values z-scored per feature across specimens. '
            'White cell = not defined (fewer than two particles).', 'ex_vivo')
    ivh = iv[iv.feature_voxel_mm == 0.1].copy()          # same voxel size only (texture comparability)
    ivh['gi'] = ivh.group.map({g: i for i, g in enumerate(GROUPS)})
    ivh = ivh.sort_values(['gi', 'timepoint_months', 'animal']).reset_index(drop=True)
    ivh['row'] = ivh.animal + ' · ' + ivh.timepoint_months.astype(str) + ' mo' + \
        np.where(ivh.placement.str.startswith('unvalidated'), ' †', '')
    heatmap(ivh, HEAT_COMMON, 'row',
            [('group', gcol, 'Group'),
             ('timepoint_months', {3: '#9ec5f4', 6: '#184f95'}, 'Timepoint')],
            'Fig4_heatmap',
            'In vivo defect core, one row per scan; values z-scored per feature across scans (timepoint strip: light = 3 months, dark = 6 months).'
            + (' † ROI placed by raw network inference (unvalidated).' if iv.placement.str.startswith('unvalidated').any() else '')
            , 'in_vivo')

    if args.images:
        _img = importlib.util.spec_from_file_location('_img', REPO_DIR / '9b_opg_scan_figure.py')
        m = importlib.util.module_from_spec(_img); _img.loader.exec_module(m)
        m.make_invivo(iv)
        m.make_exvivo(exs)
        CAPTION_NOTES.update(m.FIG.CAPTION_NOTES)
    for sub in ('in_vivo', 'ex_vivo'):
        (OUT / sub / 'figure_notes.txt').write_text(
            '\n'.join(f'{k.split("/")[1]}: {v}' for k, v in CAPTION_NOTES.items() if k.startswith(sub)) + '\n')
    print(f'\nOutputs in {OUT}/in_vivo and {OUT}/ex_vivo')


if __name__ == '__main__':
    main()
