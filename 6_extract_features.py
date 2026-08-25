#!/usr/bin/env python3
"""Extract Otsu and radiomic features from a placed ROI.

Given a raw scan and a written ROI series (any placement mode — network,
registration, ex vivo, manual), this recomputes the exact core / ring
geometry from the series and extracts, per region:

  * first-order intensity statistics (mean, spread, percentiles, skewness,
    kurtosis, RMS, entropy, uniformity — 25 HU bins over −1000…3000 HU)
  * Otsu features (Otsu threshold inside the region, BV/TV at Otsu, 3-class
    multi-Otsu thresholds and class fractions, BV/TV at the fixed study
    threshold, mean HU of supra-threshold bone ≈ tissue-mineral-density proxy)
  * 3D grey-level co-occurrence (GLCM) texture over 13 directions at
    distance 1 (contrast, dissimilarity, homogeneity, ASM/energy,
    correlation, cluster shade/prominence, entropy, max probability)

plus core-to-ring ratios for the headline features. pyradiomics is NOT used
— it does not install on current Python/NumPy — the features follow the
same IBSI-style definitions implemented on numpy/scipy/scikit-image.

    python 6_extract_features.py --input DICOM_DIR --roi OUT_BASE

Writes <roi>_features.csv (feature, core, ring, core_to_ring) and
<roi>_features.json (values + extraction metadata). Values are computed on
a working grid of ~0.06 mm minimum voxel (native for in vivo scans; ex vivo
µCT is block-averaged in-plane and strided in Z), recorded in the metadata.
Features are comparable only between runs extracted at the same voxel size,
bin settings, and placement method.
"""

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from scipy import stats as sstats
from skimage.filters import threshold_multiotsu, threshold_otsu

_SPEC = importlib.util.spec_from_file_location(
    '_inference', Path(__file__).parent / '3_inference.py')
_INF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INF)
from axial_view import _crop_offset  # noqa: E402  (shares the repo's geometry)

HU_LO, HU_HI = -1000.0, 3000.0
FO_BIN_HU = 25.0        # first-order entropy/uniformity bin width
GLCM_BINS = 32          # grey levels for co-occurrence
TARGET_VOXEL_MM = 0.06  # working-grid floor; native spacing is kept if coarser
# 13 unique 3D neighbour directions (the other 13 are their negatives, which
# the symmetric accumulation already covers).
GLCM_OFFSETS = [(dz, dy, dx)
                for dz in (0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dz, dy, dx) > (0, 0, 0)]


def fit_pose_from_series(input_dir: Path, roi_dir: Path, spacing, orig_h):
    """Centre + axis of the stamped template, re-fitted from the union series
    (same math as axial_view.AxialView, without loading any CT)."""
    hdr_out = _INF.dcm_files_sorted(roi_dir)
    k = max(1, int(round(0.1 / float(min(spacing)))))
    ds_in0 = pydicom.dcmread(str(_INF.dcm_files_sorted(input_dir)[0][1]),
                             stop_before_pixels=True)
    r0 = c0 = None
    pts = []
    for zi in range(0, len(hdr_out), k):
        arr = pydicom.dcmread(str(hdr_out[zi][1])).pixel_array[::k, ::k]
        if not arr.any():
            continue
        if r0 is None:
            r0, c0 = _crop_offset(ds_in0, pydicom.dcmread(
                str(hdr_out[zi][1]), stop_before_pixels=True))
        rr, cc = np.where(arr != 0)
        pts.append(np.stack([np.full(rr.shape, zi), rr * k + r0, cc * k + c0],
                            axis=1))
    if not pts:
        raise RuntimeError(f'no non-empty slices in {roi_dir}')
    coords = np.concatenate(pts).astype(np.float64) * np.asarray(spacing)
    center, axis, _ = _INF.fit_axis(coords)
    return center, axis


def load_working_volume(slices, spacing, orig_h, orig_w, center, axis):
    """HU subvolume covering the template bbox on the working grid.

    In-plane voxels are BLOCK-AVERAGED (not strided) so 15 µm ex vivo noise
    is genuinely reduced rather than sampled; Z uses a stride. For in vivo
    scans the factor is 1 and the volume is native.
    """
    k = max(1, int(round(TARGET_VOXEL_MM / float(min(spacing)))))
    geom = _INF.OrientedCylinder(center, axis, spacing)
    half = geom.bbox_half_mm() + 0.5
    lo = np.floor((center - half) / spacing).astype(int)
    hi = np.ceil((center + half) / spacing).astype(int)
    z0 = max(0, lo[0]); z1 = min(len(slices) - 1, hi[0])
    r0 = max(0, lo[1]); r1 = min(orig_h, hi[1] + 1)
    c0 = max(0, lo[2]); c1 = min(orig_w, hi[2] + 1)
    nr = (r1 - r0) // k
    nc = (c1 - c0) // k
    zs = list(range(z0, z1 + 1, k))
    vol = np.empty((len(zs), nr, nc), dtype=np.float32)
    for i, z in enumerate(zs):
        ds = pydicom.dcmread(str(slices[z][1]))
        hu = _INF.pixel_to_hu(ds.pixel_array, ds)[r0:r0 + nr * k, c0:c0 + nc * k]
        vol[i] = hu.reshape(nr, k, nc, k).mean(axis=(1, 3))
        if (i + 1) % 100 == 0 or i + 1 == len(zs):
            print(f'  loaded {i + 1}/{len(zs)} slices')
    # mm coordinates of working voxel (i,j,l), full-frame
    z_mm = np.array(zs) * spacing[0]
    r_mm = (r0 + np.arange(nr) * k + (k - 1) / 2.0) * spacing[1]
    c_mm = (c0 + np.arange(nc) * k + (k - 1) / 2.0) * spacing[2]
    voxel_mm = (spacing[0] * k, spacing[1] * k, spacing[2] * k)
    return vol, z_mm, r_mm, c_mm, voxel_mm


def region_masks(z_mm, r_mm, c_mm, center, axis):
    dz = (z_mm - center[0]).astype(np.float32)[:, None, None]
    dr = (r_mm - center[1]).astype(np.float32)[None, :, None]
    dc = (c_mm - center[2]).astype(np.float32)[None, None, :]
    t = dz * axis[0] + dr * axis[1] + dc * axis[2]
    d2 = dz ** 2 + dr ** 2 + dc ** 2 - t ** 2
    d = np.sqrt(np.maximum(d2, 0.0))
    within = np.abs(t) <= _INF.CYL_HEIGHT_MM / 2.0
    core = within & (d <= _INF.CYLINDER_MM)
    ring = within & (d >= _INF.RING_INNER_MM) & (d <= _INF.RING_OUTER_MM)
    return core, ring


def first_order(vals, voxel_mm3):
    v = vals.astype(np.float64)
    hist, _ = np.histogram(np.clip(v, HU_LO, HU_HI),
                           bins=int((HU_HI - HU_LO) / FO_BIN_HU),
                           range=(HU_LO, HU_HI))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {
        'voxels': int(v.size),
        'volume_mm3': float(v.size * voxel_mm3),
        'mean_hu': float(v.mean()),
        'std_hu': float(v.std()),
        'min_hu': float(v.min()),
        'max_hu': float(v.max()),
        'p10_hu': float(q[0]), 'p25_hu': float(q[1]), 'median_hu': float(q[2]),
        'p75_hu': float(q[3]), 'p90_hu': float(q[4]),
        'iqr_hu': float(q[3] - q[1]),
        'range_hu': float(v.max() - v.min()),
        'skewness': float(sstats.skew(v)),
        'kurtosis': float(sstats.kurtosis(v)),
        'rms_hu': float(np.sqrt((v ** 2).mean())),
        'mean_abs_dev_hu': float(np.abs(v - v.mean()).mean()),
        'entropy_bits': float(-(p * np.log2(p)).sum()),
        'uniformity': float((p ** 2).sum()),
    }


def otsu_features(vals, bone_thr):
    v = vals.astype(np.float64)
    out = {'bvtv_fixed': float((v > bone_thr).mean())}
    bone = v[v > bone_thr]
    out['bone_mean_hu'] = float(bone.mean()) if bone.size else float('nan')
    try:
        t = float(threshold_otsu(v))
        out['otsu_threshold_hu'] = t
        out['bvtv_otsu'] = float((v > t).mean())
    except ValueError:
        out['otsu_threshold_hu'] = out['bvtv_otsu'] = float('nan')
    try:
        t1, t2 = (float(x) for x in threshold_multiotsu(v, classes=3))
        out['multiotsu_t1_hu'] = t1
        out['multiotsu_t2_hu'] = t2
        out['fraction_low'] = float((v <= t1).mean())
        out['fraction_mid'] = float(((v > t1) & (v <= t2)).mean())
        out['fraction_high'] = float((v > t2).mean())
    except ValueError:
        for key in ('multiotsu_t1_hu', 'multiotsu_t2_hu',
                    'fraction_low', 'fraction_mid', 'fraction_high'):
            out[key] = float('nan')
    return out


def glcm_features(vol, mask):
    """Symmetric, normalised 3D GLCM over the 13 directions at distance 1."""
    b = np.clip(((vol - HU_LO) / (HU_HI - HU_LO) * GLCM_BINS).astype(np.int32),
                0, GLCM_BINS - 1)
    P = np.zeros((GLCM_BINS, GLCM_BINS), dtype=np.float64)

    def shifted(a, off):
        sl_a, sl_b = [], []
        for o in off:
            n = a.shape[len(sl_a)]
            if o >= 0:
                sl_a.append(slice(0, n - o)); sl_b.append(slice(o, n))
            else:
                sl_a.append(slice(-o, n)); sl_b.append(slice(0, n + o))
        return a[tuple(sl_a)], a[tuple(sl_b)]

    for off in GLCM_OFFSETS:
        ma, mb = shifted(mask, off)
        both = ma & mb
        if not both.any():
            continue
        va, vb = shifted(b, off)
        pair = va[both] * GLCM_BINS + vb[both]
        h = np.bincount(pair, minlength=GLCM_BINS * GLCM_BINS) \
            .reshape(GLCM_BINS, GLCM_BINS)
        P += h + h.T                       # symmetric
    s = P.sum()
    if s == 0:
        return {k: float('nan') for k in
                ('glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
                 'glcm_asm', 'glcm_energy', 'glcm_correlation',
                 'glcm_cluster_shade', 'glcm_cluster_prominence',
                 'glcm_entropy_bits', 'glcm_max_prob')}
    P /= s
    i = np.arange(GLCM_BINS, dtype=np.float64)
    I, J = np.meshgrid(i, i, indexing='ij')
    mu = (P * I).sum()                     # symmetric → same for rows/cols
    sig2 = (P * (I - mu) ** 2).sum()
    nz = P[P > 0]
    corr = float(((P * (I - mu) * (J - mu)).sum() / sig2) if sig2 > 0 else 1.0)
    return {
        'glcm_contrast': float((P * (I - J) ** 2).sum()),
        'glcm_dissimilarity': float((P * np.abs(I - J)).sum()),
        'glcm_homogeneity': float((P / (1.0 + (I - J) ** 2)).sum()),
        'glcm_asm': float((P ** 2).sum()),
        'glcm_energy': float(np.sqrt((P ** 2).sum())),
        'glcm_correlation': corr,
        'glcm_cluster_shade': float((P * (I + J - 2 * mu) ** 3).sum()),
        'glcm_cluster_prominence': float((P * (I + J - 2 * mu) ** 4).sum()),
        'glcm_entropy_bits': float(-(nz * np.log2(nz)).sum()),
        'glcm_max_prob': float(P.max()),
    }


RATIO_FEATURES = ['mean_hu', 'bvtv_fixed', 'bvtv_otsu', 'bone_mean_hu',
                  'entropy_bits', 'glcm_contrast', 'glcm_homogeneity',
                  'glcm_entropy_bits']


def main():
    p = argparse.ArgumentParser(description='Otsu + radiomic feature extraction')
    p.add_argument('--input', required=True, help='raw DICOM scan directory')
    p.add_argument('--roi', required=True,
                   help='ROI output base (the union series directory)')
    p.add_argument('--bone-threshold', type=float, default=_INF.BONE_THRESHOLD_HU)
    p.add_argument('--out', help='output base (default: <roi>_features)')
    args = p.parse_args()

    input_dir, roi_dir = Path(args.input), Path(args.roi)
    for d in (input_dir, roi_dir):
        if not d.is_dir():
            print(f'ERROR: not a directory: {d}'); sys.exit(1)
    out_base = Path(args.out) if args.out else \
        roi_dir.parent / (roi_dir.name + '_features')

    print(f'Scan    : {input_dir}')
    print(f'ROI     : {roi_dir}')
    slices = _INF.dcm_files_sorted(input_dir)
    ds0 = pydicom.dcmread(str(slices[0][1]), stop_before_pixels=True)
    ps = [float(v) for v in ds0.PixelSpacing]
    if len(slices) > 1:
        p0 = np.array([float(v) for v in ds0.ImagePositionPatient])
        p1 = np.array([float(v) for v in pydicom.dcmread(
            str(slices[1][1]), stop_before_pixels=True).ImagePositionPatient])
        sz = float(np.linalg.norm(p1 - p0))
    else:
        sz = float(getattr(ds0, 'SliceThickness', ps[0]))
    spacing = np.array([sz, ps[0], ps[1]])

    print('Re-fitting template pose from the ROI series...')
    center, axis = fit_pose_from_series(input_dir, roi_dir, spacing, ds0.Rows)
    print(f'  centre {np.round(center, 2)} mm, axis {np.round(axis, 3)}')

    print('Loading working volume...')
    vol, z_mm, r_mm, c_mm, voxel_mm = load_working_volume(
        slices, spacing, ds0.Rows, ds0.Columns, center, axis)
    print(f'  {vol.shape} voxels at {tuple(round(v, 4) for v in voxel_mm)} mm')
    core, ring = region_masks(z_mm, r_mm, c_mm, center, axis)
    voxel_mm3 = float(np.prod(voxel_mm))

    feats = {}
    for name, mask in (('core', core), ('ring', ring)):
        print(f'Extracting {name} features ({int(mask.sum()):,} voxels)...')
        vals = vol[mask]
        f = first_order(vals, voxel_mm3)
        f.update(otsu_features(vals, args.bone_threshold))
        f.update(glcm_features(vol, mask))
        feats[name] = f

    order = list(feats['core'].keys())
    ratios = {}
    for k in RATIO_FEATURES:
        c, r = feats['core'].get(k), feats['ring'].get(k)
        ratios[k] = float(c / r) if (c is not None and r not in (None, 0)
                                     and np.isfinite(c) and np.isfinite(r)
                                     and r != 0) else None

    csv_path = out_base.with_suffix('.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['feature', 'core', 'ring', 'core_to_ring'])
        for k in order:
            w.writerow([k, feats['core'][k], feats['ring'][k],
                        ratios.get(k, '')])
    meta = {
        'input': str(input_dir), 'roi': str(roi_dir),
        'extracted_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'voxel_mm': [round(v, 5) for v in voxel_mm],
        'hu_window': [HU_LO, HU_HI], 'first_order_bin_hu': FO_BIN_HU,
        'glcm_bins': GLCM_BINS, 'glcm_distance': 1,
        'bone_threshold_hu': args.bone_threshold,
        'center_mm': [round(float(v), 3) for v in center],
        'axis': [round(float(v), 4) for v in axis],
        'engine': 'numpy/scipy/scikit-image (pyradiomics unavailable on this '
                  'Python; IBSI-style definitions)',
    }
    json_path = out_base.with_suffix('.json')
    json_path.write_text(json.dumps(
        {'meta': meta, 'core': feats['core'], 'ring': feats['ring'],
         'core_to_ring': ratios}, indent=1))
    print(f'\nWrote {csv_path.name} and {json_path.name} '
          f'({len(order)} features per region).')
    print('NOTE: compare features only between runs extracted at the same '
          'voxel size, bins, and placement method.')


if __name__ == '__main__':
    main()
