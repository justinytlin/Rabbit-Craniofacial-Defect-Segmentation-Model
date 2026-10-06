#!/usr/bin/env python3
"""Mineral density + particle analysis for placed ROIs (add-on to step 6).

EX VIVO (SCANCO) ONLY. Computed per region (core / ring) on exactly the same
working grid and template pose as 6_extract_features.py, but written to a
SEPARATE sidecar file so the existing <roi>_features.json files, the ROI
series and every existing database column stay untouched. Everything uses
the SCANCO HA calibration stored in each ex vivo DICOM (private tags
0029,1000-1006) and the Bouxsein 2010 mineral threshold in real units,
226 mg HA/cm^3 (~1365 HU on SCANCO; the study's 226 HU is only
~24 mg HA/cm^3 there, i.e. soft tissue).

  * density
      bmd_mgha      mean density over the whole region
      tmd_mgha      mean density of voxels above 226 mg HA/cm^3 (tissue mineral density)
      bvtv_226mgha  BV/TV at 226 mg HA/cm^3

  * particles — every 26-connected piece of mineralised material inside the
    region (residual scaffold, bone islands). SCANCO-style Gaussian smoothing
    (sigma 0.8 working voxel) before the 226 mg HA/cm^3 threshold; pieces
    smaller than --min-particle-mm3 (default 0.01 mm^3, ~0.27 mm diameter)
    are dropped. Pieces are cut at the region boundary, so a plate crossing
    it counts once per region. Per-particle density uses unsmoothed values.
      particle_count, particle_number_density (per mm^3 of region),
      particle volume mean / median / p90, mean equivalent-sphere diameter,
      largest-particle fraction, mean per-particle HU and mg HA/cm^3,
      mean nearest-neighbour centroid distance.

In vivo (SOFIE, 100 µm) scans get an all-empty sidecar: they carry no mineral
calibration, and at 100 µm a 226 HU particle count measured noise and
fragmentation (an empty defect scored more "particles" than a scaffold-filled
one — tested 2026-09-30 on 37951 vs 37950). Use the HU features already in
the database (mean_hu, bone_mean_hu, bone_components) for in vivo.

    python 8_density_particles.py --input DICOM_DIR --roi SERIES_DIR   # one series
    python 8_density_particles.py --all [--only SUBJECT ...]           # every database row
    python 8_density_particles.py --merge                              # add columns to the database

--merge backs up radiomics_database.csv/.xlsx first, then adds the new
core_ / ring_ / core_to_ring_ columns to the EXISTING rows (matched on
roi_series_dir); no other column is changed and nothing is re-extracted.
7_build_database.py also reads the sidecars, so full rebuilds keep them.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage
from scipy.spatial import cKDTree

REPO_DIR = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, REPO_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FX = _load('_extract', '6_extract_features.py')   # grid, pose, masks (imports 3_inference)
_INF = _FX._INF

DENSITY_VERSION = 2   # v2 (2026-10-05): specimen-masked keys for cut specimens
MIN_PARTICLE_MM3 = 0.01
SMOOTH_SIGMA_VOX = 0.8     # Gaussian sigma in working voxels, before thresholding
TRUE_BONE_MGHA = 226.0      # Bouxsein 2010 mineral threshold, in real units
DENSITY_KEYS = ('bmd_mgha', 'tmd_mgha', 'bvtv_226mgha')
SPECIMEN_KEYS = ('specimen_fraction', 'spec_bvtv_226mgha', 'spec_bmd_mgha',
                 'spec_particle_number_density')
PARTICLE_KEYS = (
    'particle_count', 'particle_number_density', 'particle_volume_mean_mm3',
    'particle_volume_median_mm3', 'particle_volume_p90_mm3',
    'particle_eqdiam_mean_mm', 'particle_largest_fraction',
    'particle_mean_hu', 'particle_mean_mgha', 'particle_nn_dist_mean_mm')


def density_path(series_dir: Path) -> Path:
    return series_dir.parent / (series_dir.name + '_density.json')


# ─────────────────────────────────────────────────────────── calibration

def read_ha_calibration(ds):
    """SCANCO HA calibration as hu -> mg HA/cm^3 coefficients (a, b), or None.

    SCANCO DICOMs store HU = 1000 (mu - mu_water) / mu_water and the density
    calibration as mgHA = slope * mu + intercept, so
    mgHA = slope * mu_water * (HU + 1000) / 1000 + intercept.
    """
    try:
        if 'mg HA' not in str(ds[0x0029, 0x1003].value):
            return None
        slope = float(ds[0x0029, 0x1004].value)
        intercept = float(ds[0x0029, 0x1005].value)
        mu_water = float(ds[0x0029, 0x1006].value)
    except (KeyError, ValueError, TypeError):
        return None
    # Cross-check the HU rescale against mu_scaling so a re-exported series
    # with a different convention cannot silently mis-calibrate.
    try:
        mu_scaling = float(ds[0x0029, 0x1000].value)
        expect = 1000.0 / (mu_scaling * mu_water)
        if (abs(float(ds.RescaleSlope) / expect - 1.0) > 0.01
                or abs(float(ds.RescaleIntercept) + 1000.0) > 1.0):
            print('  WARNING: HU rescale does not match the SCANCO calibration '
                  '— mg HA/cm^3 features left empty')
            return None
    except (KeyError, ValueError, TypeError, AttributeError):
        pass
    a = slope * mu_water / 1000.0
    return {'a': a, 'b': a * 1000.0 + intercept,
            'slope': slope, 'intercept': intercept, 'mu_water': mu_water}


def hu_to_mgha(hu, cal):
    return cal['a'] * hu + cal['b']


def mgha_to_hu(mgha, cal):
    return (mgha - cal['b']) / cal['a']


# ─────────────────────────────────────────────────────────── features

def density_features(vals, cal, thr_hu):
    out = {k: float('nan') for k in DENSITY_KEYS}
    if vals.size == 0:
        return out
    v = vals.astype(np.float64)
    out['bmd_mgha'] = float(hu_to_mgha(v.mean(), cal))
    bone = v[v > thr_hu]
    if bone.size:
        out['tmd_mgha'] = float(hu_to_mgha(bone.mean(), cal))
    out['bvtv_226mgha'] = float(bone.size / v.size)
    return out


def particle_features(vol, smooth, mask, thr_hu, voxel_mm, cal, min_mm3):
    """Particles segmented on the smoothed volume; densities from the raw one."""
    out = {k: float('nan') for k in PARTICLE_KEYS}
    voxv = float(np.prod(voxel_mm))
    region_mm3 = float(mask.sum()) * voxv
    out['particle_count'] = 0.0
    out['particle_number_density'] = 0.0
    bone = mask & (smooth > thr_hu)
    if not bone.any():
        return out
    lab, n = ndimage.label(bone, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(lab.ravel())[1:]
    min_vox = max(1, math.ceil(min_mm3 / voxv - 1e-9))
    keep = np.flatnonzero(sizes >= min_vox) + 1          # label ids
    if keep.size == 0:
        return out
    vols = sizes[keep - 1] * voxv
    out['particle_count'] = float(keep.size)
    out['particle_number_density'] = float(keep.size / region_mm3) if region_mm3 else float('nan')
    out['particle_volume_mean_mm3'] = float(vols.mean())
    out['particle_volume_median_mm3'] = float(np.median(vols))
    out['particle_volume_p90_mm3'] = float(np.percentile(vols, 90))
    out['particle_eqdiam_mean_mm'] = float(np.mean((6.0 * vols / np.pi) ** (1.0 / 3.0)))
    out['particle_largest_fraction'] = float(vols.max() / vols.sum())
    means = np.asarray(ndimage.mean(vol, lab, keep), dtype=np.float64)
    out['particle_mean_hu'] = float(means.mean())
    out['particle_mean_mgha'] = float(hu_to_mgha(means, cal).mean())
    if keep.size >= 2:
        cm = np.asarray(ndimage.center_of_mass(bone, lab, keep)) * np.asarray(voxel_mm)
        d, _ = cKDTree(cm).query(cm, k=2)
        out['particle_nn_dist_mean_mm'] = float(d[:, 1].mean())
    return out


def specimen_mask(smooth, thr_hu):
    """Where the specimen is, for specimens cut through the defect (#262 batch).

    The defect itself is not mineralised, so 'tissue above threshold' would
    exclude it; instead take the convex hull, in the slice plane, of the
    specimen's mineralised footprint. A hull spans the open half-defect but
    stops at the straight cut edge. Ex vivo plates lie near the slice plane
    (axis tilt < 4 deg), so the 2D hull is extruded along Z.
    """
    from skimage.morphology import convex_hull_image
    foot = ndimage.binary_opening((smooth > thr_hu).any(axis=0), iterations=2)
    lab, n = ndimage.label(foot)
    if n == 0:
        return np.zeros(smooth.shape, bool)
    big = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return np.broadcast_to(convex_hull_image(big), smooth.shape)


def specimen_features(vol, mask, spec, thr_hu, voxel_mm, cal, n_particles):
    inside = mask & spec
    n_in = int(inside.sum())
    out = {'specimen_fraction': float(n_in / mask.sum()) if mask.any() else float('nan'),
           'spec_bvtv_226mgha': float('nan'), 'spec_bmd_mgha': float('nan'),
           'spec_particle_number_density': float('nan')}
    if n_in:
        v = vol[inside].astype(np.float64)
        out['spec_bvtv_226mgha'] = float((v > thr_hu).mean())
        out['spec_bmd_mgha'] = float(hu_to_mgha(v.mean(), cal))
        out['spec_particle_number_density'] = float(n_particles / (n_in * np.prod(voxel_mm)))
    return out


def extract(input_dir: Path, roi_dir: Path, min_mm3: float):
    slices = _INF.dcm_files_sorted(input_dir)
    ds0 = pydicom.dcmread(str(slices[0][1]), stop_before_pixels=True)
    ps = [float(v) for v in ds0.PixelSpacing]
    if len(slices) > 1:      # same spacing rule as 6_extract_features.main
        p0 = np.array([float(v) for v in ds0.ImagePositionPatient])
        p1 = np.array([float(v) for v in pydicom.dcmread(
            str(slices[1][1]), stop_before_pixels=True).ImagePositionPatient])
        sz = float(np.linalg.norm(p1 - p0))
    else:
        sz = float(getattr(ds0, 'SliceThickness', ps[0]))
    spacing = np.array([sz, ps[0], ps[1]])
    cal = read_ha_calibration(ds0)
    meta = {
        'input': str(input_dir), 'roi': str(roi_dir),
        'extracted_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'density_version': DENSITY_VERSION,
        'ha_calibration': cal,
    }
    if cal is None:
        print('  no mineral calibration (in vivo) — empty sidecar, see docstring')
        empty = {k: None for k in DENSITY_KEYS + PARTICLE_KEYS + SPECIMEN_KEYS}
        meta['note'] = 'no mineral calibration; density/particle features are ex vivo only'
        return {'meta': meta, 'core': dict(empty), 'ring': dict(empty)}
    thr_hu = float(mgha_to_hu(TRUE_BONE_MGHA, cal))
    print(f'  SCANCO calibration: 226 mg HA/cm^3 = {thr_hu:.0f} HU')

    center, axis = _FX.fit_pose_from_series(input_dir, roi_dir, spacing, ds0.Rows)
    vol, z_mm, r_mm, c_mm, voxel_mm = _FX.load_working_volume(
        slices, spacing, ds0.Rows, ds0.Columns, center, axis)
    cut = _FX.load_cut(roi_dir)
    core, ring = _FX.region_masks(z_mm, r_mm, c_mm, center, axis, cut)
    smooth = ndimage.gaussian_filter(vol, SMOOTH_SIGMA_VOX)
    spec = specimen_mask(smooth, thr_hu)

    feats = {}
    for name, mask in (('core', core), ('ring', ring)):
        f = density_features(vol[mask], cal, thr_hu)
        f.update(particle_features(vol, smooth, mask, thr_hu, voxel_mm, cal, min_mm3))
        f.update(specimen_features(vol, mask, spec, thr_hu, voxel_mm, cal, f['particle_count']))
        feats[name] = f
        tmd = f['tmd_mgha']
        print(f'  {name}: {int(f["particle_count"])} particles'
              + (f', TMD {tmd:.0f} mg HA/cm^3' if np.isfinite(tmd) else ''))

    meta.update({
        'voxel_mm': [round(v, 5) for v in voxel_mm],
        'threshold_mgha': TRUE_BONE_MGHA,
        'threshold_hu': round(thr_hu, 1),
        'smooth_sigma_vox': SMOOTH_SIGMA_VOX,
        'min_particle_mm3': min_mm3,
        'particle_connectivity': 26,
        'template': 'half cylinder (cut specimen)' if cut is not None else 'full cylinder',
        'specimen_mask': 'convex hull (in the slice plane) of the largest mineralised '
                         'component, projected along Z and extruded — keeps the cut edge of '
                         'halved specimens; spec_* keys are computed inside it',
        'center_mm': [round(float(v), 3) for v in center],
        'axis': [round(float(v), 4) for v in axis],
    })
    return {'meta': meta, 'core': feats['core'], 'ring': feats['ring']}


# ─────────────────────────────────────────────────────────── batch / merge

def is_current(sdir: Path, db) -> bool:
    f = density_path(sdir)
    if not f.exists() or os.stat(f).st_mtime < db.newest_mtime(sdir):
        return False
    try:
        return json.loads(f.read_text())['meta'].get('density_version') == DENSITY_VERSION
    except (OSError, ValueError, KeyError):
        return False


def run_one(input_dir: Path, sdir: Path, args) -> bool:
    t0 = time.time()
    print(f'{sdir.name}')
    try:
        d = extract(input_dir, sdir, args.min_particle_mm3)
    except Exception as e:                  # noqa: BLE001  (keep the batch going)
        print(f'  FAILED: {e}')
        return False
    density_path(sdir).write_text(json.dumps(d, indent=1))
    print(f'  wrote {density_path(sdir).name} ({time.time() - t0:.0f} s)')
    return True


def run_all(args, db):
    df = db.load_existing(Path(args.db))
    if df is None:
        raise SystemExit('no radiomics_database.csv found')
    base = db.DATA_ROOT.parent
    todo = df[df['roi_series_dir'].notna()]
    if args.only:
        todo = todo[todo['subject'].astype(str).isin(args.only)]
    done = failed = skipped = 0
    for _, r in todo.iterrows():
        sdir, idir = base / r['roi_series_dir'], base / r['scan_dir']
        if not args.force and is_current(sdir, db):
            skipped += 1
            continue
        if not (sdir.is_dir() and idir.is_dir()):
            print(f'{sdir.name}: missing series or scan folder — skipped'); failed += 1
            continue
        if run_one(idir, sdir, args):
            done += 1
        else:
            failed += 1
    print(f'\n{done} extracted, {skipped} already current, {failed} failed')


def carry_over_xlsx_edits(df, out_dir: Path, str_cols=()):
    """Moved to 7_build_database.py (shared with the web-app add path)."""
    return _load('_build_db', '7_build_database.py').carry_over_xlsx_edits(df, out_dir, str_cols)


def merge(args, db):
    out_dir = Path(args.db)
    df = db.load_existing(out_dir)
    if df is None:
        raise SystemExit('no radiomics_database.csv found')
    df = carry_over_xlsx_edits(df, out_dir, db.STR_COLS)
    base = db.DATA_ROOT.parent
    n = 0
    new_cols = {}
    for i, r in df.iterrows():
        if not isinstance(r['roi_series_dir'], str):
            continue
        cols = db.density_columns(base / r['roi_series_dir'])
        if cols:
            n += 1
            for k, v in cols.items():
                new_cols.setdefault(k, {})[i] = v
    for k, vals in new_cols.items():         # overwrite only the density/particle columns
        df[k] = df.index.map(vals)
    print(f'merged density/particle features into {n} of {len(df)} rows')
    db.write_outputs(df, out_dir)


def main():
    db = _load('_build_db', '7_build_database.py')
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', help='raw DICOM scan directory (single-series mode)')
    p.add_argument('--roi', help='ROI union series directory (single-series mode)')
    p.add_argument('--all', action='store_true', help='every series in the database')
    p.add_argument('--only', action='append', metavar='SUBJECT', help='with --all: restrict to subject(s)')
    p.add_argument('--force', action='store_true', help='recompute even if the sidecar is current')
    p.add_argument('--merge', action='store_true', help='add sidecar columns to the database')
    p.add_argument('--db', default=os.environ.get(
        'DEFECT_DB_DIR', str(db.DATA_ROOT / 'radiomics_database')))
    p.add_argument('--min-particle-mm3', type=float, default=MIN_PARTICLE_MM3)
    args = p.parse_args()

    if args.input and args.roi:
        sys.exit(0 if run_one(Path(args.input), Path(args.roi), args) else 1)
    if args.all:
        run_all(args, db)
    if args.merge:
        merge(args, db)
    if not (args.all or args.merge):
        p.error('give --input and --roi, or --all and/or --merge')


if __name__ == '__main__':
    main()
