#!/usr/bin/env python3
"""Half-cylinder ROIs for ex vivo specimens cut through the defect (#262 batch).

The 6-month UCLA specimens (5778-5790) are halved calvaria: the cut runs
through the middle of the defect, so the full 10 mm template ends up half in
air. For these the ROI is the standard template (8 mm tall, core r <= 5 mm,
ring 7-9 mm) restricted to the specimen side of a cut plane through the
defect centre.

Placement, per specimen (top-down, on the bone-thickness map that
exvivo_overview.py builds; ex vivo plates lie within ~4 deg of the slice
plane):
  1. specimen footprint = largest mineralised component; the cut is the
     longest edge of its convex hull (the straight saw cut, which also spans
     the open half-defect)
  2. the defect centre lies ON that line: slide a half-disk (r 5 mm) along
     it and take the position where the half-annulus 5.5-7 mm is most
     mineralised relative to the half-disk (bone around, defect inside)
  3. axis and plate height (centre z) come from the existing full-template
     placement (5_exvivo_roi.py), which fits the plate orientation well

Writes, next to the old series (kept), <old>_half_output_dicom (union series
only — the per-region copies are ~1.3 GB each at 15 µm), its <series>_pose.json (centre, axis,
cut_normal — read by 6_extract_features.py, 8_density_particles.py and
axial_view.py) and a top-down preview PNG.

    python 11_halfcut_exvivo_roi.py                   # place + write + extract all 13
    python 11_halfcut_exvivo_roi.py --only 5788       # one specimen
    python 11_halfcut_exvivo_roi.py --update-db       # then swap the database rows
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parent
PY = sys.executable
SPECIMENS = ['5778', '5779', '5780', '5781', '5782', '5783', '5784', '5785',
             '5786', '5787', '5788', '5789', '5790']


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, REPO / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


INF = _load('_inf', '3_inference.py')
OV = _load('_ov', 'exvivo_overview.py')
BASE = REPO.parent.parent


class HalfCylinder(INF.OrientedCylinder):
    """The standard template restricted to (p - c) . n >= 0."""

    def __init__(self, center_mm, axis, spacing, cut_normal):
        super().__init__(center_mm, axis, spacing)
        n = np.asarray(cut_normal, float)
        n = n - np.dot(n, self.a) * self.a
        self.n = n / np.linalg.norm(n)

    def slice_mask(self, z, kind, row0, nrow, col0, ncol):
        m = super().slice_mask(z, kind, row0, nrow, col0, ncol)
        dz = z * self.sp[0] - self.c[0]
        dr = np.arange(row0, row0 + nrow, dtype=np.float64) * self.sp[1] - self.c[1]
        dc = np.arange(col0, col0 + ncol, dtype=np.float64) * self.sp[2] - self.c[2]
        R, C = np.meshgrid(dr, dc, indexing='ij')
        return m & ((dz * self.n[0] + R * self.n[1] + C * self.n[2]) >= 0)


def thickness_map(subject, scan_dir):
    f = OV.OUT / f'{subject}.npy'
    if f.exists():
        thick = np.load(f)
    else:
        thick, _ = OV.overview(scan_dir)
        np.save(f, thick)
    ds0 = pydicom.dcmread(str(INF.dcm_files_sorted(scan_dir)[0][1]), stop_before_pixels=True)
    ps = float(ds0.PixelSpacing[0])
    k = max(1, int(round(OV.STEP_MM / ps)))
    return thick, ps, k


def find_cut_and_centre(thick, ps, k, prior_rc=None, window_mm=3.0):
    """Cut line (point, unit direction, unit normal into the specimen) and the
    defect centre on it, all in full-frame (row, col) mm."""
    step = ps * k
    off = (k - 1) / 2 * ps                         # block centre offset
    foot = ndimage.binary_opening(thick > 0.25, iterations=2)
    lab, n = ndimage.label(foot)
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    pts = np.argwhere(big) * step + off            # (row, col) mm
    hull = ConvexHull(pts)
    v = pts[hull.vertices]
    edges = [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]
    a, b = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    if np.dot(pts.mean(0) - a, nrm) < 0:
        nrm = -nrm
    # matched half-disk filter along the line
    rr, cc = np.mgrid[0:thick.shape[0], 0:thick.shape[1]]
    grid = np.stack([rr * step + off, cc * step + off], -1)
    L = np.linalg.norm(b - a)
    s_range = np.arange(0, L + 1e-9, 0.25)
    prior_used = False
    if prior_rc is not None and np.dot(np.asarray(prior_rc) - a, nrm) >= 2.0:
        # The full-template placement (moat filter) found the defect inside the
        # specimen: trust its position along the cut. Dense mineral islands in
        # a defect otherwise pull the low-density matched filter off it (5790).
        s0 = float(np.dot(np.asarray(prior_rc) - a, d))
        s_range = s_range[np.abs(s_range - s0) <= window_mm]
        prior_used = True
    best = None
    for s in s_range:
        c = a + s * d
        rel = grid - c
        dist = np.hypot(rel[..., 0], rel[..., 1])
        side = rel @ nrm
        core = (dist <= 5.0) & (side >= 0)
        ann = (dist >= 5.5) & (dist <= 7.0) & (side >= 0.5)
        if core.sum() < 50 or ann.sum() < 50:
            continue
        score = thick[ann].mean() - thick[core].mean()
        if best is None or score > best[0]:
            best = (score, c, thick[core].mean(), thick[ann].mean())
    straight = np.mean(np.abs((v - a) @ nrm) < 0.3)   # fraction of hull vertices on the line
    return dict(point=a, direction=d, normal=nrm, centre_rc=best[1], score=float(best[0]),
                core_thick=float(best[2]), annulus_thick=float(best[3]),
                cut_length_mm=float(L), hull_on_line=float(straight), prior_window_used=prior_used)


def write_half_series(scan_dir, out, center, axis, cut_normal, log):
    slices = INF.dcm_files_sorted(scan_dir)
    n = len(slices)
    ds = pydicom.dcmread(str(slices[0][1]), stop_before_pixels=True)
    orig_h, orig_w = ds.Rows, ds.Columns
    iop = [float(v) for v in ds.ImageOrientationPatient]
    row_cos, col_cos = np.array(iop[0:3]), np.array(iop[3:6])
    ps = [float(v) for v in ds.PixelSpacing]
    p0 = np.array([float(v) for v in ds.ImagePositionPatient])
    p1 = np.array([float(v) for v in pydicom.dcmread(str(slices[1][1]), stop_before_pixels=True).ImagePositionPatient])
    spacing = np.array([float(np.linalg.norm(p1 - p0)), ps[0], ps[1]])
    geom = HalfCylinder(center, axis, spacing, cut_normal)
    half = geom.bbox_half_mm() + INF.BBOX_MARGIN_MM
    lo, hi = (geom.c - half) / spacing, (geom.c + half) / spacing
    z_lo, z_hi = max(0, int(np.floor(lo[0]))), min(n - 1, int(np.ceil(hi[0])))
    row_min, row_max = max(0, int(np.round(lo[1]))), min(orig_h, int(np.round(hi[1])) + 1)
    col_min, col_max = max(0, int(np.round(lo[2]))), min(orig_w, int(np.round(hi[2])) + 1)
    crop_h, crop_w = row_max - row_min, col_max - col_min
    out.mkdir(parents=True, exist_ok=True)
    # Only the union series: at 15 µm each masked series is ~1.3 GB, and the
    # analysis reads the raw scan + pose file, not the per-region copies
    # (writing all five filled the data drive on 2026-10-06).
    outputs = [('union', out, None)]
    INF.write_all_series(slices, geom, outputs, row_min=row_min, crop_h=crop_h,
                         col_min=col_min, crop_w=crop_w, row_cos=row_cos, col_cos=col_cos,
                         row_sp=ps[0], col_sp=ps[1], z_lo=z_lo, z_hi=z_hi)
    return geom


def preview(subject, thick, ps, k, det, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    step = ps * k
    off = (k - 1) / 2 * ps
    fig, ax = plt.subplots(figsize=(4, 4))
    ext = [off - step / 2, thick.shape[1] * step + off - step / 2, thick.shape[0] * step + off - step / 2, off - step / 2]
    ax.imshow(thick, cmap='gray', extent=ext, vmin=0, vmax=np.percentile(thick[thick > 0], 99))
    c, nrm = det['centre_rc'], det['normal']
    ang = np.degrees(np.arctan2(nrm[0], nrm[1]))           # (x=col, y=row) plotting frame
    for r0, r1, col in ((0, 5, '#3bc8e8'), (7, 9, '#6fcf97')):
        ax.add_patch(Wedge((c[1], c[0]), r1, ang - 90, ang + 90, width=r1 - r0, fill=False,
                           edgecolor=col, lw=1))
    p, d = det['point'], det['direction']
    seg = np.array([p - 30 * d, p + 30 * d])
    ax.plot(seg[:, 1], seg[:, 0], color='#eb6834', lw=0.6, ls=':')
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f'{subject}  half ROI  (contrast {det["score"]:.2f} mm)', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out_png, dpi=110, bbox_inches='tight'); plt.close(fig)


def place(subject, df, log):
    r = df[(df.scan_type == 'ex vivo') & (df.subject == subject)].iloc[0]
    scan_dir = BASE / r.scan_dir
    old = BASE / r.roi_series_dir
    if old.name.endswith('_half_output_dicom'):          # already re-placed: start from the original
        old = old.parent / old.name.replace('_half_output_dicom', '_output_dicom')
    meta = json.loads((old.parent / (old.name + '_features.json')).read_text())['meta']
    center0, axis = np.asarray(meta['center_mm'], float), np.asarray(meta['axis'], float)
    thick, ps, k = thickness_map(subject, scan_dir)
    det = find_cut_and_centre(thick, ps, k, prior_rc=center0[1:])
    center = np.array([center0[0], det['centre_rc'][0], det['centre_rc'][1]])
    nrm3 = np.array([0.0, det['normal'][0], det['normal'][1]])
    a = axis / np.linalg.norm(axis)
    nrm3 = nrm3 - np.dot(nrm3, a) * a
    nrm3 /= np.linalg.norm(nrm3)
    out = old.parent / old.name.replace('_output_dicom', '_half_output_dicom')
    print(f'{subject}: cut {det["cut_length_mm"]:.1f} mm long; centre moved '
          f'{np.linalg.norm(center[1:] - center0[1:]):.2f} mm in-plane; '
          f'annulus-core contrast {det["score"]:.2f} mm', flush=True)
    write_half_series(scan_dir, out, center, a, nrm3, log)
    pose = {'center_mm': [round(float(x), 4) for x in center], 'axis': [round(float(x), 5) for x in a],
            'cut_normal': [round(float(x), 5) for x in nrm3], 'template': 'half cylinder (cut specimen)',
            'placed_by': '11_halfcut_exvivo_roi.py', 'utc': datetime.utcnow().isoformat(timespec='seconds'),
            'detection': {k2: (v.tolist() if isinstance(v, np.ndarray) else v) for k2, v in det.items()},
            'previous_series': old.name}
    (out.parent / (out.name + '_pose.json')).write_text(json.dumps(pose, indent=1))
    preview(subject, thick, ps, k, det, out.parent / (out.name + '_top_view.png'))
    for script, extra in (('6_extract_features.py', ['--bone-threshold', '226']), ('8_density_particles.py', [])):
        res = subprocess.run([PY, script, '--input', str(scan_dir), '--roi', str(out)] + extra,
                             cwd=REPO, capture_output=True, text=True)
        log.write(f'\n### {script} {out}\n{res.stdout}\n{res.stderr}\n')
        if res.returncode:
            return {'subject': subject, 'status': f'{script} failed', 'output': out}
    return {'subject': subject, 'status': 'placed', 'output': out, 'score': det['score'],
            'moved_mm': float(np.linalg.norm(center[1:] - center0[1:]))}


def shift_along_cut(subject, shift_mm, df, log):
    """Manual correction: slide an existing half ROI along its cut line.
    Positive = toward the top of the top-down views (decreasing image row)."""
    r = df[(df.scan_type == 'ex vivo') & (df.subject == subject)].iloc[0]
    scan_dir = BASE / r.scan_dir
    out = BASE / r.roi_series_dir
    assert out.name.endswith('_half_output_dicom'), f'{subject} has no half ROI yet'
    pf = out.parent / (out.name + '_pose.json')
    pose = json.loads(pf.read_text())
    det = pose['detection']
    d = np.asarray(det['direction'], float)
    up = d if d[0] < 0 else -d
    c_rc = np.asarray(det['centre_rc'], float) + shift_mm * up
    center = np.array([pose['center_mm'][0], c_rc[0], c_rc[1]])
    a = np.asarray(pose['axis'], float)
    n3 = np.asarray(pose['cut_normal'], float)
    # replace the series: move the previous one to the Trash (reversible), write fresh
    subprocess.run(['osascript', '-e', f'tell application "Finder" to delete (POSIX file "{out}" as alias)'],
                   capture_output=True, text=True, timeout=600)
    if out.exists():
        raise RuntimeError(f'could not move the previous series to the Trash: {out}')
    write_half_series(scan_dir, out, center, a, n3, log)
    hist = pose.get('manual_shifts', []) + [{'shift_mm_along_cut_up': shift_mm,
                                             'utc': datetime.utcnow().isoformat(timespec='seconds')}]
    pose.update(center_mm=[round(float(x), 4) for x in center], manual_shifts=hist)
    pose['detection']['centre_rc'] = c_rc.tolist()
    pf.write_text(json.dumps(pose, indent=1))
    thick, ps, k = thickness_map(subject, scan_dir)
    det2 = dict(det, centre_rc=c_rc, point=np.asarray(det['point']), direction=d,
                normal=np.asarray(det['normal']))
    preview(subject, thick, ps, k, det2, out.parent / (out.name + '_top_view.png'))
    for script, extra in (('6_extract_features.py', ['--bone-threshold', '226']), ('8_density_particles.py', [])):
        res = subprocess.run([PY, script, '--input', str(scan_dir), '--roi', str(out)] + extra,
                             cwd=REPO, capture_output=True, text=True)
        log.write(f'\n### {script} {out}\n{res.stdout}\n{res.stderr}\n')
        if res.returncode:
            return {'subject': subject, 'status': f'{script} failed', 'output': out}
    print(f'{subject}: half ROI moved {shift_mm:+.1f} mm along the cut', flush=True)
    return {'subject': subject, 'status': 'existing', 'output': out, 'manual': True}


def contact_sheet(subjects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mi
    df = pd.read_csv(REPO.parent / 'radiomics_database' / 'radiomics_database.csv', dtype={'subject': str})
    pngs = []
    for s in subjects:
        r = df[(df.scan_type == 'ex vivo') & (df.subject == s)].iloc[0]
        d = BASE / r.roi_series_dir
        base = d.name if d.name.endswith('_half_output_dicom') else d.name.replace('_output_dicom', '_half_output_dicom')
        f = d.parent / (base + '_top_view.png')
        if f.exists():
            pngs.append((s, r.treatment, f))
    cols = 5
    rows = int(np.ceil(len(pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.1))
    for ax in np.atleast_1d(axes).flat:
        ax.axis('off')
    for ax, (s, t, f) in zip(np.atleast_1d(axes).flat, pngs):
        ax.imshow(mi.imread(f)); ax.set_title(t, fontsize=8)
    fig.tight_layout()
    out = OV.OUT / 'half_roi_contact_sheet.png'
    fig.savefig(out, dpi=90); plt.close(fig)
    print('sheet:', out)


def update_db(results):
    db = _load('_db', '7_build_database.py')
    out_dir = db.DATA_ROOT / 'radiomics_database'
    df = db.load_existing(out_dir)
    df = db.carry_over_xlsx_edits(df, out_dir, db.STR_COLS)
    n = 0
    for res in results:
        if res['status'] not in ('placed', 'existing'):
            continue
        sdir = Path(res['output'])
        m = (df.scan_type == 'ex vivo') & (df.subject.astype(str) == res['subject'])
        assert m.sum() == 1, res['subject']
        i = df.index[m][0]
        old = df.loc[i, 'roi_series']
        scan = {k: df.loc[i, k] for k in ('treatment', 'subject', 'timepoint', 'timepoint_months',
                                         'scan_type', 'study_id', 'patient_id', 'study_date',
                                         'n_slices', 'pixel_spacing_mm', 'slice_spacing_mm', 'note')}
        scan.update(manufacturer=df.loc[i, 'scanner'], scan_rel=df.loc[i, 'scan_dir'],
                    scan_dir=str(BASE / df.loc[i, 'scan_dir']))
        ser = {'series_dir': str(sdir), 'series_name': sdir.name, 'series_files': len(db.dcm_names(sdir))}
        row = db.series_row(scan, ser, {}, {}, 'extracted now')
        for k, v in row.items():
            if k.startswith(('core_', 'ring_')) or k in (
                    'features_extracted_utc', 'feature_voxel_mm', 'bone_threshold_hu', 'roi_center_z_mm',
                    'roi_center_row_mm', 'roi_center_col_mm', 'roi_axis_z', 'roi_axis_row', 'roi_axis_col',
                    'roi_tilt_deg', 'roi_series', 'roi_series_dir', 'roi_series_date'):
                if k not in df.columns:
                    df[k] = None
                df[k] = df[k].astype(object)
                df.at[i, k] = v
        df.at[i, 'placement_method'] = 'ex vivo half-cylinder (specimen cut through the defect)'
        df.at[i, 'manually_adjusted'] = bool(res.get('manual')) or bool(
            json.loads((sdir.parent / (sdir.name + '_pose.json')).read_text()).get('manual_shifts'))
        df.at[i, 'webapp_job_id'] = None
        df.at[i, 'qc_overall'] = 'half ROI — review top view'
        df.at[i, 'qc_flags'] = ''
        df.at[i, 'added_via'] = (f'11_halfcut_exvivo_roi.py {datetime.now():%Y-%m-%d %H:%M} '
                                 f'(replaced full-template series {old})')
        oth = str(df.at[i, 'other_series_on_disk'] or '').replace('nan', '')
        df.at[i, 'other_series_on_disk'] = '; '.join(x for x in (oth, old) if x)
        n += 1
    print(f'updated {n} rows')
    db.write_outputs(df, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--update-db', action='store_true', help='swap the database rows to the half-ROI series')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--shift', nargs=2, metavar=('SUBJECT', 'MM'),
                    help='slide that specimen\'s half ROI along the cut (+ = toward the top of the views)')
    a = ap.parse_args()
    df = pd.read_csv(REPO.parent / 'radiomics_database' / 'radiomics_database.csv', dtype={'subject': str})
    if a.shift:
        logf = REPO / 'logs' / f'halfcut_shift_{datetime.now():%Y%m%d_%H%M}.log'
        with open(logf, 'w') as log:
            res = shift_along_cut(a.shift[0], float(a.shift[1]), df, log)
        contact_sheet(SPECIMENS)
        if a.update_db:
            update_db([res])
        return
    subjects = a.only or SPECIMENS
    logf = REPO / 'logs' / f'halfcut_{datetime.now():%Y%m%d_%H%M}.log'
    results = []
    with open(logf, 'w') as log:
        for s in subjects:
            r = df[(df.scan_type == 'ex vivo') & (df.subject == s)].iloc[0]
            d = BASE / r.roi_series_dir
            half = d if d.name.endswith('_half_output_dicom') else d.parent / d.name.replace('_output_dicom', '_half_output_dicom')
            if a.skip_existing and (half.parent / (half.name + '_density.json')).exists():
                results.append({'subject': s, 'status': 'existing', 'output': half}); continue
            try:
                results.append(place(s, df, log))
            except Exception as e:                      # noqa: BLE001
                results.append({'subject': s, 'status': f'error: {e}'})
            print('  ->', results[-1]['status'], flush=True)
    pd.DataFrame(results).to_csv(logf.with_suffix('.csv'), index=False)
    contact_sheet(subjects)
    if a.update_db:
        update_db(results)


if __name__ == '__main__':
    main()
