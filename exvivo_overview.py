#!/usr/bin/env python3
"""Whole-specimen overview of ex vivo scans, to find where the defect actually is.

The web app's adjustment view only shows a window around the current ROI. This
renders the ENTIRE scan top-down (ex vivo plates lie roughly in the slice
plane): a bone-thickness map (mm of tissue above 226 mg HA/cm^3 along Z) with
the current ROI core / ring drawn on it. A defect reads as a round hole fully
surrounded by bone; a specimen cut through the defect shows the hole open to
the specimen edge.

    python exvivo_overview.py --subjects 5788 5789      # selected scans
    python exvivo_overview.py --all                     # every ex vivo row in the database
Writes ../figures/exvivo_overview/<subject>.png and a contact sheet.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

REPO = Path(__file__).resolve().parent


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


INF = _load('_inf', '3_inference.py')
DP = _load('_dp', '8_density_particles.py')
OUT = REPO.parent / 'figures' / 'exvivo_overview'
BASE = REPO.parent.parent
STEP_MM = 0.12


def overview(scan_dir: Path):
    sl = INF.dcm_files_sorted(scan_dir)
    ds0 = pydicom.dcmread(str(sl[0][1]), stop_before_pixels=True)
    ps = float(ds0.PixelSpacing[0])
    k = max(1, int(round(STEP_MM / ps)))
    cal = DP.read_ha_calibration(ds0)
    thr_hu = DP.mgha_to_hu(226.0, cal) if cal else 1366.0
    slope, icpt = float(ds0.RescaleSlope), float(ds0.RescaleIntercept)
    thr_raw = (thr_hu - icpt) / slope
    h, w = ds0.Rows // k, ds0.Columns // k
    thick = np.zeros((h, w), np.float32)
    zs = range(0, len(sl), k)
    for z in zs:
        a = pydicom.dcmread(str(sl[z][1])).pixel_array[:h * k, :w * k]
        a = a.reshape(h, k, w, k).mean(axis=(1, 3))
        thick += a > thr_raw
    thick *= ps * k          # mm of mineralised tissue along Z
    return thick, ps * k


def roi_center(series_dir: Path):
    f = series_dir.parent / (series_dir.name + '_features.json')
    m = json.loads(f.read_text())['meta']
    return m['center_mm'], m['axis']


def render(sub, row, thick, step, ax):
    from matplotlib.patches import Circle
    ext = [0, thick.shape[1] * step, thick.shape[0] * step, 0]
    ax.imshow(thick, cmap='gray', vmin=0, vmax=np.percentile(thick[thick > 0], 99) if (thick > 0).any() else 1,
              extent=ext)
    try:
        c, _ = roi_center(BASE / row.roi_series_dir)
        for r, col in ((5, '#3bc8e8'), (7, '#f2c94c'), (9, '#6fcf97')):
            ax.add_patch(Circle((c[2], c[1]), r, fill=False, color=col, lw=1, ls='--'))
    except Exception:                       # noqa: BLE001
        pass
    ax.set_title(f'{sub}  {row.treatment}', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='*')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    df = pd.read_csv(REPO.parent / 'radiomics_database' / 'radiomics_database.csv', dtype={'subject': str})
    ex = df[(df.scan_type == 'ex vivo') & df.roi_series_dir.notna()]
    if not a.all:
        ex = ex[ex.subject.isin(a.subjects)]
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    for _, r in ex.iterrows():
        png = OUT / f'{r.subject}.png'
        if not png.exists():
            try:
                thick, step = overview(BASE / r.scan_dir)
            except Exception as e:          # noqa: BLE001
                print(r.subject, 'FAILED', e); continue
            np.save(OUT / f'{r.subject}.npy', thick)
            fig, ax = plt.subplots(figsize=(4, 4))
            render(r.subject, r, thick, step, ax)
            fig.savefig(png, dpi=110, bbox_inches='tight'); plt.close(fig)
            print(r.subject, 'done', flush=True)
        done.append(r)
    # contact sheet
    n = len(done)
    if n:
        cols = 6
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.7))
        for ax in np.atleast_1d(axes).flat:
            ax.axis('off')
        for ax, r in zip(np.atleast_1d(axes).flat, done):
            thick = np.load(OUT / f'{r.subject}.npy')
            ds0 = pydicom.dcmread(str(INF.dcm_files_sorted(BASE / r.scan_dir)[0][1]), stop_before_pixels=True)
            ps = float(ds0.PixelSpacing[0]); k = max(1, int(round(STEP_MM / ps)))
            render(r.subject, r, thick, ps * k, ax); ax.axis('on')
        fig.tight_layout()
        name = 'contact_sheet_all.png' if a.all else 'contact_sheet_' + '_'.join(a.subjects[:4]) + '.png'
        fig.savefig(OUT / name, dpi=90); plt.close(fig)
        print('sheet:', OUT / name)


if __name__ == '__main__':
    main()
