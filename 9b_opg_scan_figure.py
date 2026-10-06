#!/usr/bin/env python3
"""Fig. 1 of each figure set — representative scans, Defect vs C-OPG/SPDP-OPG vs Soaked-OPG.

  make_invivo : one MR525xx rabbit per group (closest to its group median of
                in vivo defect:reference BV/TV and mean HU at 6 months), shown
                at 3 and 6 months top-down plus a 6-month section. HU window.
  make_exvivo : one 6-month specimen per group (closest to its group median of
                specimen-masked BV/TV, BMD and TMD), top-down plus section.
                mg HA/cm^3 window (SCANCO calibration).

Every view is resliced perpendicular to the fitted defect axis
(axial_view.AxialView), centred on the ROI, with identical display windows
across groups.

Called from 9_opg_figures.py --images (or run directly).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import map_coordinates

REPO_DIR = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_DIR / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FIG = _load('_fig', '9_opg_figures.py')
DP = _load('_dp', '8_density_particles.py')
from axial_view import AxialView  # noqa: E402

FOV_MM = 10.5           # half-width of every panel (ring OD radius = 9 mm)
IV_WIN = (0, 2400)      # HU — shared across panels; ≈ the web app's per-image 1–99 % stretch
EX_WIN = (0, 1100)      # mg HA/cm^3
EX_STRIDE = 4           # 15 µm -> 60 µm sampling (same grid as the features)
SLAB_MM = 4.0           # ex vivo MIP slab thickness (thin plate)
# Manual choice overrides the closest-to-median pick (user request 2026-10-05:
# MR52529 replaced; MR52527 = reviewed at both timepoints and = ex vivo specimen 5787).
IV_REPRESENTATIVE = {'Soaked-OPG': 'MR52527'}
IV_SLAB_MM = 8.0        # in vivo MIP slab = ROI height, as in the web app's view


def closest_to_median(frame, cols):
    """Per group, the row closest to its group median (robust z across groups)."""
    sd = frame[cols].std(ddof=1)
    out = {}
    for g in FIG.GROUPS:
        sub = frame[frame.group == g]
        if sub.empty:
            continue
        out[g] = sub.loc[((sub[cols] - sub[cols].median()) / sd).abs().sum(axis=1).idxmin()]
    return out


def to_units(av, raw, ex):
    ds = pydicom.dcmread(str(next(iter(av._in_by_inst.values()))), stop_before_pixels=True)
    hu = raw * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    if not ex:
        return hu
    cal = DP.read_ha_calibration(ds)
    return DP.hu_to_mgha(hu, cal)


def views(scan_dir, series_dir, ex):
    av = AxialView(scan_dir, series_dir, fov_mm=FOV_MM, stride=EX_STRIDE if ex else None)
    slab = SLAB_MM if ex else IV_SLAB_MM
    top, uv = av.slab(slab_mm=slab, fov_mm=FOV_MM, n=int(slab / av.spacing.min()) + 1)
    # section: plane containing the defect axis, through the centre — along the
    # cut normal for half ROIs (so it runs from the cut edge into the specimen)
    cut = getattr(av, 'cut_normal', None)
    sdir = av.e1 if cut is None else cut
    step = float(av.spacing.min())
    u = np.arange(-FOV_MM, FOV_MM + step, step)
    t = np.arange(-4.5, 4.5 + step, step)
    U, T = np.meshgrid(u, t, indexing='ij')
    p = av.center_mm + U[..., None] * sdir + T[..., None] * av.axis
    idx = p / av.spacing - av.origin
    side = map_coordinates(av.ct, [idx[..., 0], idx[..., 1], idx[..., 2]], order=1, cval=0.0)
    cut_deg = None if cut is None else float(np.degrees(np.arctan2(cut @ av.e2, cut @ av.e1)))
    return to_units(av, top, ex), uv, to_units(av, side, ex), u, t, cut_deg


def _strip(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _top_panel(ax, img, uv, win, g, tag, scale_bar, cut_deg=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle
    im = ax.imshow(img.T, cmap='gray', vmin=win[0], vmax=win[1], origin='lower',
                   extent=[uv[0], uv[-1], uv[0], uv[-1]], interpolation='nearest')
    for rad, ls in ((5.0, '-'), (7.0, '--'), (9.0, '--')):
        kw = dict(lw=0.7 if rad == 5 else 0.5, ls=ls,
                  color=FIG.COLOR[g] if rad == 5 else '#ffffff', alpha=0.95)
        if cut_deg is None:
            ax.add_patch(Circle((0, 0), rad, fill=False, **kw))
        else:                                   # half ROI: semicircles + the cut
            ax.add_patch(Arc((0, 0), 2 * rad, 2 * rad, theta1=cut_deg - 90, theta2=cut_deg + 90, **kw))
    if cut_deg is not None:
        ang = np.radians(cut_deg + 90)
        ax.plot([-9 * np.cos(ang), 9 * np.cos(ang)], [-9 * np.sin(ang), 9 * np.sin(ang)],
                color=FIG.COLOR[g], lw=0.7)
    _strip(ax)
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, color='white', fontsize=5.5, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.25', facecolor='black', alpha=0.6, lw=0))
    if scale_bar:
        ax.add_patch(plt.Rectangle((-10.2, -10.3), 3.4, 1.3, color='black', alpha=0.6, lw=0))
        ax.plot([-9.5, -7.5], [-9.9, -9.9], color='white', lw=1.5, solid_capstyle='butt')
        ax.text(-8.5, -9.75, '2 mm', color='white', fontsize=5, ha='center', va='bottom')
    return im


def _side_panel(ax, side, u, t, win, g, half=False):
    ax.imshow(side.T, cmap='gray', vmin=win[0], vmax=win[1], origin='lower',
              extent=[u[0], u[-1], t[0], t[-1]], aspect='equal', interpolation='nearest')
    for x in ((0, 5) if half else (-5, 5)):
        ax.axvline(x, color=FIG.COLOR[g], lw=0.7)
    for x in ((7, 9) if half else (-9, -7, 7, 9)):
        ax.axvline(x, color='white', lw=0.5, ls='--')
    _strip(ax)


def _layout(n_top):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.09, 2.45 * n_top + 1.25))
    gs = fig.add_gridspec(n_top + 1, 4, width_ratios=[1, 1, 1, 0.06],
                          height_ratios=[1] * n_top + [0.48], wspace=0.06, hspace=0.16)
    return fig, gs


def _colorbar(fig, gs, ri, im, label):
    cax = fig.add_subplot(gs[ri, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)


def make_invivo(iv):
    import matplotlib.pyplot as plt
    FIG.setup_style()
    six = iv[(iv.timepoint_months == 6) & iv.animal.str.startswith('MR525')]
    reps = closest_to_median(six, ['core_to_ring_bvtv_fixed', 'core_mean_hu'])
    for g, a in IV_REPRESENTATIVE.items():
        reps[g] = six[six.animal == a].iloc[0]
    df = pd.read_csv(FIG.DB_CSV, dtype={'subject': str})
    fig, gs = _layout(2)
    labels = ['3 months', '6 months', '6 months\nsection']
    for ci, g in enumerate(FIG.GROUPS):
        animal = reps[g].animal
        print(f'  in vivo {g}: {animal}')
        for ri, mo in enumerate((3, 6)):
            row = df[(df.scan_type == 'in vivo') & (df.subject == animal) & (df.timepoint_months == mo)].iloc[0]
            top, uv, side, u, t, _ = views(FIG.BASE / row.scan_dir, FIG.BASE / row.roi_series_dir, False)
            ax = fig.add_subplot(gs[ri, ci])
            im = _top_panel(ax, top, uv, IV_WIN, g, animal, ci == 0)
            if ri == 0:
                ax.set_title(g, fontsize=8, fontweight='bold', color=FIG.COLOR[g], pad=4)
            if ci == 0:
                ax.set_ylabel(labels[ri], fontsize=7)
            if ci == 0:
                _colorbar(fig, gs, ri, im, 'HU')
        axs = fig.add_subplot(gs[2, ci])
        _side_panel(axs, side, u, t, IV_WIN, g)
        if ci == 0:
            axs.set_ylabel(labels[2], fontsize=7)
    FIG.CAPTION_NOTES['in_vivo/Fig1_representative_scans'] = (
        f'In vivo µCT (100 µm) of one rabbit per group at 3 and 6 months. Top-down views are {IV_SLAB_MM:.0f} mm '
        'maximum-intensity slabs perpendicular to the fitted defect axis; the bottom row is a 6-month section containing '
        'the defect axis. Coloured solid circle/lines = 10 mm defect core ROI; white dashed = 14–18 mm reference ring. '
        'Window 0–2400 HU, identical for all panels. Defect and C-OPG/SPDP-OPG: the MR525xx rabbit closest to its group median 6-month '
        'defect:reference BV/TV and defect mean HU; Soaked-OPG: MR52527 (the same rabbit as ex vivo specimen 5787).')
    FIG.save(fig, 'Fig1_representative_scans', 'in_vivo')
    plt.close(fig)


def make_exvivo(exs):
    import matplotlib.pyplot as plt
    FIG.setup_style()
    reps = closest_to_median(exs, ['core_spec_bvtv_226mgha', 'core_spec_bmd_mgha', 'core_tmd_mgha'])
    df = pd.read_csv(FIG.DB_CSV, dtype={'subject': str})
    fig, gs = _layout(1)
    for ci, g in enumerate(FIG.GROUPS):
        r = reps[g]
        row = df[(df.scan_type == 'ex vivo') & (df.subject == r.scans)].iloc[0]
        print(f'  ex vivo {g}: specimen {r.scans} ({r.animal})')
        top, uv, side, u, t, cut_deg = views(FIG.BASE / row.scan_dir, FIG.BASE / row.roi_series_dir, True)
        ax = fig.add_subplot(gs[0, ci])
        im = _top_panel(ax, top, uv, EX_WIN, g, f'specimen {r.scans} ({r.animal.replace("x", "MR")})', ci == 0, cut_deg)
        ax.set_title(g, fontsize=8, fontweight='bold', color=FIG.COLOR[g], pad=4)
        if ci == 0:
            ax.set_ylabel('Top-down', fontsize=7)
            _colorbar(fig, gs, 0, im, 'mg HA/cm³')
        axs = fig.add_subplot(gs[1, ci])
        _side_panel(axs, side, u, t, EX_WIN, g, half=cut_deg is not None)
        if ci == 0:
            axs.set_ylabel('Section', fontsize=7)
    FIG.CAPTION_NOTES['ex_vivo/Fig1_representative_scans'] = (
        f'Ex vivo SCANCO µCT (15 µm, shown at 60 µm) of one 6-month specimen per group. Top: {SLAB_MM:.0f} mm '
        'maximum-intensity slab perpendicular to the defect axis; bottom: section containing the defect axis. '
        'Coloured solid circle/lines = 10 mm defect core ROI; white dashed = 14–18 mm reference ring. '
        'Window 0–1100 mg HA/cm³. Specimens are halved calvaria cut through the defect, so the ROI is the half of the '
        'template on the specimen side of the cut (coloured straight line); the section runs from the cut into the specimen. '
        'Representative = specimen '
        'closest to its group median specimen-masked BV/TV, BMD and TMD.')
    FIG.save(fig, 'Fig1_representative_scans', 'ex_vivo')
    plt.close(fig)


if __name__ == '__main__':
    exs, iv = FIG.load_sets()
    make_invivo(iv)
    make_exvivo(exs)
