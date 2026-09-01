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
    distance 1 — 24 features (contrast, dissimilarity, homogeneity,
    ASM/energy, correlation, cluster shade/prominence/tendency, entropy,
    max probability, autocorrelation, joint average, sum of squares,
    sum/difference entropy, difference variance, inverse difference
    variants, IMC1/2, maximal correlation coefficient)
  * grey-level run-length (GLRLM, 16), size-zone (GLSZM, 16), dependence
    (GLDM, 14, alpha=0) and neighbourhood grey-tone difference (NGTDM, 5)
    texture — all on the same 32-bin discretisation, 26-neighbourhood,
    directions merged
  * bone-mask morphometry of the segmented bone (voxels above the fixed
    threshold) inside each region: volume, marching-cubes surface area,
    volume/surface, sphericity, connected components, largest-component
    fraction, elongation/flatness/anisotropy from the inertia eigenvalues,
    convex-hull solidity. (Shape of the region itself is NOT reported —
    the ROI is a rigid stamped cylinder, identical for every sample.)

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
from scipy import ndimage
from scipy import stats as sstats
from skimage import measure as skmeasure
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
N26_OFFSETS = GLCM_OFFSETS + [tuple(-o for o in off) for off in GLCM_OFFSETS]
MAX_RUN = 512           # GLRLM run-length cap (longer runs are clipped)


def shift_full(a, off, fill):
    """Array of a's value at x+off, placed at x (out-of-range -> fill)."""
    out = np.full_like(a, fill)
    src, dst = [], []
    for i, o in enumerate(off):
        n = a.shape[i]
        if o >= 0:
            dst.append(slice(0, n - o)); src.append(slice(o, n))
        else:
            dst.append(slice(-o, n)); src.append(slice(0, n + o))
    out[tuple(dst)] = a[tuple(src)]
    return out


def bin_volume(vol):
    """Discretise HU to the shared GLCM_BINS grey levels."""
    return np.clip(((vol - HU_LO) / (HU_HI - HU_LO) * GLCM_BINS)
                   .astype(np.int16), 0, GLCM_BINS - 1)


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


GLCM_KEYS = (
    'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
    'glcm_asm', 'glcm_energy', 'glcm_correlation',
    'glcm_cluster_shade', 'glcm_cluster_prominence',
    'glcm_entropy_bits', 'glcm_max_prob',
    'glcm_autocorrelation', 'glcm_cluster_tendency', 'glcm_joint_average',
    'glcm_sum_of_squares', 'glcm_sum_entropy_bits',
    'glcm_difference_entropy_bits', 'glcm_difference_variance',
    'glcm_inverse_difference', 'glcm_inverse_difference_norm',
    'glcm_idm_norm', 'glcm_inverse_variance', 'glcm_imc1', 'glcm_imc2',
    'glcm_mcc')


def glcm_features(b, mask):
    """Symmetric, normalised 3D GLCM over the 13 directions at distance 1.

    Dissimilarity equals the IBSI difference average, and sum average is
    2x the joint average — those duplicates are not reported twice.
    """
    P = np.zeros((GLCM_BINS, GLCM_BINS), dtype=np.float64)
    for off in GLCM_OFFSETS:
        both = mask & shift_full(mask, off, False)
        if not both.any():
            continue
        vb = shift_full(b, off, 0)
        pair = b[both].astype(np.int64) * GLCM_BINS + vb[both]
        h = np.bincount(pair, minlength=GLCM_BINS * GLCM_BINS) \
            .reshape(GLCM_BINS, GLCM_BINS)
        P += h + h.T                       # symmetric
    s = P.sum()
    if s == 0:
        return {k: float('nan') for k in GLCM_KEYS}
    P /= s
    i = np.arange(GLCM_BINS, dtype=np.float64)
    I, J = np.meshgrid(i, i, indexing='ij')
    px = P.sum(axis=1)                     # symmetric → px == py
    mu = (px * i).sum()
    sig2 = (px * (i - mu) ** 2).sum()
    nz = P[P > 0]
    corr = float(((P * (I - mu) * (J - mu)).sum() / sig2) if sig2 > 0 else 1.0)
    kd = np.abs(I - J).astype(np.int64)
    p_diff = np.bincount(kd.ravel(), weights=P.ravel(), minlength=GLCM_BINS)
    p_sum = np.bincount((I + J).astype(np.int64).ravel(), weights=P.ravel(),
                        minlength=2 * GLCM_BINS - 1)
    d_avg = (p_diff * np.arange(GLCM_BINS)).sum()
    nzd = p_diff[p_diff > 0]
    nzs = p_sum[p_sum > 0]
    hxy = float(-(nz * np.log2(nz)).sum())
    pxnz = px[px > 0]
    hx = float(-(pxnz * np.log2(pxnz)).sum())
    PXY = np.outer(px, px)
    m1 = P > 0
    hxy1 = float(-(P[m1] * np.log2(PXY[m1])).sum())
    m2 = PXY > 0
    hxy2 = float(-(PXY[m2] * np.log2(PXY[m2])).sum())
    off_diag = kd > 0
    mcc = float('nan')
    rows = px > 0
    if rows.sum() >= 2:
        Pr = P[np.ix_(rows, rows)]
        pxr = px[rows]
        Q = (Pr / pxr[:, None]) @ (Pr / pxr[None, :]).T
        ev = np.sort(np.abs(np.linalg.eigvals(Q)))
        mcc = float(np.sqrt(max(0.0, min(1.0, ev[-2]))))
    return {
        'glcm_contrast': float((P * (I - J) ** 2).sum()),
        'glcm_dissimilarity': float(d_avg),
        'glcm_homogeneity': float((P / (1.0 + (I - J) ** 2)).sum()),
        'glcm_asm': float((P ** 2).sum()),
        'glcm_energy': float(np.sqrt((P ** 2).sum())),
        'glcm_correlation': corr,
        'glcm_cluster_shade': float((P * (I + J - 2 * mu) ** 3).sum()),
        'glcm_cluster_prominence': float((P * (I + J - 2 * mu) ** 4).sum()),
        'glcm_entropy_bits': hxy,
        'glcm_max_prob': float(P.max()),
        'glcm_autocorrelation': float((P * I * J).sum()),
        'glcm_cluster_tendency': float((P * (I + J - 2 * mu) ** 2).sum()),
        'glcm_joint_average': float(mu),
        'glcm_sum_of_squares': float(sig2),
        'glcm_sum_entropy_bits': float(-(nzs * np.log2(nzs)).sum()),
        'glcm_difference_entropy_bits': float(-(nzd * np.log2(nzd)).sum()),
        'glcm_difference_variance': float(
            (p_diff * (np.arange(GLCM_BINS) - d_avg) ** 2).sum()),
        'glcm_inverse_difference': float((P / (1.0 + kd)).sum()),
        'glcm_inverse_difference_norm': float(
            (P / (1.0 + kd / GLCM_BINS)).sum()),
        'glcm_idm_norm': float((P / (1.0 + (kd / GLCM_BINS) ** 2)).sum()),
        'glcm_inverse_variance': float(
            (P[off_diag] / kd[off_diag].astype(np.float64) ** 2).sum()),
        'glcm_imc1': float((hxy - hxy1) / hx) if hx > 0 else 0.0,
        'glcm_imc2': float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - hxy))))),
        'glcm_mcc': mcc,
    }


GLRLM_KEYS = (
    'glrlm_short_run_emphasis', 'glrlm_long_run_emphasis',
    'glrlm_gray_level_non_uniformity', 'glrlm_gray_level_non_uniformity_norm',
    'glrlm_run_length_non_uniformity', 'glrlm_run_length_non_uniformity_norm',
    'glrlm_run_percentage', 'glrlm_low_gray_level_run_emphasis',
    'glrlm_high_gray_level_run_emphasis', 'glrlm_short_run_low_gray_emphasis',
    'glrlm_short_run_high_gray_emphasis', 'glrlm_long_run_low_gray_emphasis',
    'glrlm_long_run_high_gray_emphasis', 'glrlm_gray_level_variance',
    'glrlm_run_variance', 'glrlm_run_entropy_bits')


def glrlm_features(b, mask):
    """Run-length matrix over the 13 directions, merged. Run lengths are
    found by pointer-doubling on uniform shifts (no per-line Python loop)."""
    R = np.zeros((GLCM_BINS, MAX_RUN + 1), dtype=np.float64)
    maxdim = max(b.shape)
    for off in GLCM_OFFSETS:
        cont = mask & shift_full(mask, off, False) & \
            (b == shift_full(b, off, -1))
        f = mask.astype(np.int32)          # run length ahead, capped at 2^k
        c = cont.copy()                    # continues for >= 2^k more steps
        step = 1
        while c.any() and step <= maxdim:
            joff = tuple(o * step for o in off)
            f = f + np.where(c, shift_full(f, joff, 0), 0)
            c &= shift_full(c, joff, False)
            step *= 2
        neg = tuple(-o for o in off)
        prev = shift_full(mask, neg, False) & (b == shift_full(b, neg, -1))
        start = mask & ~prev
        L = np.clip(f[start], 1, MAX_RUN).astype(np.int64)
        g = b[start].astype(np.int64)
        R += np.bincount(g * (MAX_RUN + 1) + L,
                         minlength=GLCM_BINS * (MAX_RUN + 1)) \
            .reshape(GLCM_BINS, MAX_RUN + 1)
    nr = R.sum()
    if nr == 0:
        return {k: float('nan') for k in GLRLM_KEYS}
    p = R / nr
    lv = np.arange(1, GLCM_BINS + 1, dtype=np.float64)      # grey level 1..Ng
    ln = np.arange(MAX_RUN + 1, dtype=np.float64)
    ln[0] = 1.0                            # column 0 is empty; avoid /0
    LV, LN = lv[:, None], ln[None, :]
    pg, pl = p.sum(axis=1), p.sum(axis=0)
    mug = (pg * lv).sum()
    mul = (pl * ln).sum()
    nzp = p[p > 0]
    return {
        'glrlm_short_run_emphasis': float((pl / ln ** 2).sum()),
        'glrlm_long_run_emphasis': float((pl * ln ** 2).sum()),
        'glrlm_gray_level_non_uniformity': float((R.sum(1) ** 2).sum() / nr),
        'glrlm_gray_level_non_uniformity_norm': float((pg ** 2).sum()),
        'glrlm_run_length_non_uniformity': float((R.sum(0) ** 2).sum() / nr),
        'glrlm_run_length_non_uniformity_norm': float((pl ** 2).sum()),
        'glrlm_run_percentage': float(
            nr / (float(mask.sum()) * len(GLCM_OFFSETS))),
        'glrlm_low_gray_level_run_emphasis': float((pg / lv ** 2).sum()),
        'glrlm_high_gray_level_run_emphasis': float((pg * lv ** 2).sum()),
        'glrlm_short_run_low_gray_emphasis': float(
            (p / (LV ** 2 * LN ** 2)).sum()),
        'glrlm_short_run_high_gray_emphasis': float(
            (p * LV ** 2 / LN ** 2).sum()),
        'glrlm_long_run_low_gray_emphasis': float(
            (p * LN ** 2 / LV ** 2).sum()),
        'glrlm_long_run_high_gray_emphasis': float(
            (p * LV ** 2 * LN ** 2).sum()),
        'glrlm_gray_level_variance': float((pg * (lv - mug) ** 2).sum()),
        'glrlm_run_variance': float((pl * (ln - mul) ** 2).sum()),
        'glrlm_run_entropy_bits': float(-(nzp * np.log2(nzp)).sum()),
    }


GLSZM_KEYS = (
    'glszm_small_area_emphasis', 'glszm_large_area_emphasis',
    'glszm_gray_level_non_uniformity', 'glszm_gray_level_non_uniformity_norm',
    'glszm_size_zone_non_uniformity', 'glszm_size_zone_non_uniformity_norm',
    'glszm_zone_percentage', 'glszm_low_gray_level_zone_emphasis',
    'glszm_high_gray_level_zone_emphasis', 'glszm_small_area_low_gray_emphasis',
    'glszm_small_area_high_gray_emphasis', 'glszm_large_area_low_gray_emphasis',
    'glszm_large_area_high_gray_emphasis', 'glszm_gray_level_variance',
    'glszm_zone_variance', 'glszm_zone_entropy_bits')


def glszm_features(b, mask):
    """Size-zone matrix: 26-connected zones of equal grey level."""
    struct = np.ones((3, 3, 3), dtype=bool)
    sizes, levels = [], []
    for g in range(GLCM_BINS):
        m = mask & (b == g)
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=struct)
        if n == 0:
            continue
        sz = np.bincount(lab.ravel())[1:].astype(np.float64)
        sizes.append(sz)
        levels.append(np.full(sz.size, g + 1, dtype=np.float64))
    if not sizes:
        return {k: float('nan') for k in GLSZM_KEYS}
    s = np.concatenate(sizes)
    gl = np.concatenate(levels)
    nz_zones = float(s.size)
    lvl_counts = np.bincount(gl.astype(np.int64))
    _, size_counts = np.unique(s, return_counts=True)
    pair_key = gl.astype(np.int64) * (10 ** 9) + s.astype(np.int64)
    _, pair_counts = np.unique(pair_key, return_counts=True)
    pz = pair_counts / nz_zones
    return {
        'glszm_small_area_emphasis': float((1.0 / s ** 2).mean()),
        'glszm_large_area_emphasis': float((s ** 2).mean()),
        'glszm_gray_level_non_uniformity': float(
            (lvl_counts.astype(np.float64) ** 2).sum() / nz_zones),
        'glszm_gray_level_non_uniformity_norm': float(
            (lvl_counts.astype(np.float64) ** 2).sum() / nz_zones ** 2),
        'glszm_size_zone_non_uniformity': float(
            (size_counts.astype(np.float64) ** 2).sum() / nz_zones),
        'glszm_size_zone_non_uniformity_norm': float(
            (size_counts.astype(np.float64) ** 2).sum() / nz_zones ** 2),
        'glszm_zone_percentage': float(nz_zones / float(mask.sum())),
        'glszm_low_gray_level_zone_emphasis': float((1.0 / gl ** 2).mean()),
        'glszm_high_gray_level_zone_emphasis': float((gl ** 2).mean()),
        'glszm_small_area_low_gray_emphasis': float(
            (1.0 / (s ** 2 * gl ** 2)).mean()),
        'glszm_small_area_high_gray_emphasis': float(
            (gl ** 2 / s ** 2).mean()),
        'glszm_large_area_low_gray_emphasis': float(
            (s ** 2 / gl ** 2).mean()),
        'glszm_large_area_high_gray_emphasis': float(
            ((s * gl) ** 2).mean()),
        'glszm_gray_level_variance': float(gl.var()),
        'glszm_zone_variance': float(s.var()),
        'glszm_zone_entropy_bits': float(-(pz * np.log2(pz)).sum()),
    }


GLDM_KEYS = (
    'gldm_small_dependence_emphasis', 'gldm_large_dependence_emphasis',
    'gldm_gray_level_non_uniformity', 'gldm_dependence_non_uniformity',
    'gldm_dependence_non_uniformity_norm', 'gldm_gray_level_variance',
    'gldm_dependence_variance', 'gldm_dependence_entropy_bits',
    'gldm_low_gray_level_emphasis', 'gldm_high_gray_level_emphasis',
    'gldm_small_dep_low_gray_emphasis', 'gldm_small_dep_high_gray_emphasis',
    'gldm_large_dep_low_gray_emphasis', 'gldm_large_dep_high_gray_emphasis')


def gldm_features(b, mask):
    """Dependence matrix, alpha=0: a 26-neighbour is dependent when it has
    the same binned grey level. Dependence size = neighbour count + 1."""
    dep = np.zeros(b.shape, dtype=np.int16)
    for off in N26_OFFSETS:
        dep += (mask & shift_full(mask, off, False)
                & (b == shift_full(b, off, -1)))
    g = b[mask].astype(np.int64)
    k = dep[mask].astype(np.int64)
    D = np.bincount(g * 27 + k, minlength=GLCM_BINS * 27) \
        .reshape(GLCM_BINS, 27).astype(np.float64)
    nz = D.sum()
    if nz == 0:
        return {k2: float('nan') for k2 in GLDM_KEYS}
    p = D / nz
    lv = np.arange(1, GLCM_BINS + 1, dtype=np.float64)
    j = np.arange(1, 28, dtype=np.float64)          # dependence size
    LV, J = lv[:, None], j[None, :]
    pg, pj = p.sum(axis=1), p.sum(axis=0)
    mug = (pg * lv).sum()
    muj = (pj * j).sum()
    nzp = p[p > 0]
    return {
        'gldm_small_dependence_emphasis': float((pj / j ** 2).sum()),
        'gldm_large_dependence_emphasis': float((pj * j ** 2).sum()),
        'gldm_gray_level_non_uniformity': float((D.sum(1) ** 2).sum() / nz),
        'gldm_dependence_non_uniformity': float((D.sum(0) ** 2).sum() / nz),
        'gldm_dependence_non_uniformity_norm': float((pj ** 2).sum()),
        'gldm_gray_level_variance': float((pg * (lv - mug) ** 2).sum()),
        'gldm_dependence_variance': float((pj * (j - muj) ** 2).sum()),
        'gldm_dependence_entropy_bits': float(-(nzp * np.log2(nzp)).sum()),
        'gldm_low_gray_level_emphasis': float((pg / lv ** 2).sum()),
        'gldm_high_gray_level_emphasis': float((pg * lv ** 2).sum()),
        'gldm_small_dep_low_gray_emphasis': float(
            (p / (LV ** 2 * J ** 2)).sum()),
        'gldm_small_dep_high_gray_emphasis': float(
            (p * LV ** 2 / J ** 2).sum()),
        'gldm_large_dep_low_gray_emphasis': float(
            (p * J ** 2 / LV ** 2).sum()),
        'gldm_large_dep_high_gray_emphasis': float(
            (p * J ** 2 * LV ** 2).sum()),
    }


NGTDM_KEYS = ('ngtdm_coarseness', 'ngtdm_busyness', 'ngtdm_complexity',
              'ngtdm_contrast', 'ngtdm_strength')


def ngtdm_features(b, mask):
    """Neighbourhood grey-tone difference over the 26-neighbourhood."""
    bf = b.astype(np.float64)
    nsum = np.zeros(b.shape, dtype=np.float64)
    ncnt = np.zeros(b.shape, dtype=np.int16)
    for off in N26_OFFSETS:
        mm = shift_full(mask, off, False)
        nsum += np.where(mm, shift_full(bf, off, 0.0), 0.0)
        ncnt += mm
    valid = mask & (ncnt > 0)
    nv = float(valid.sum())
    if nv == 0:
        return {k: float('nan') for k in NGTDM_KEYS}
    avg = nsum[valid] / ncnt[valid]
    g = b[valid].astype(np.int64)
    diff = np.abs(bf[valid] - avg)
    n_g = np.bincount(g, minlength=GLCM_BINS).astype(np.float64)
    s_g = np.bincount(g, weights=diff, minlength=GLCM_BINS)
    p_g = n_g / nv
    pres = p_g > 0
    lv = np.arange(1, GLCM_BINS + 1, dtype=np.float64)
    pi, pj = p_g[pres][:, None], p_g[pres][None, :]
    si, sj = s_g[pres][:, None], s_g[pres][None, :]
    li, lj = lv[pres][:, None], lv[pres][None, :]
    ngp = float(pres.sum())
    s_tot = s_g.sum()
    ps = (p_g * s_g).sum()
    out = {'ngtdm_coarseness': float(1.0 / ps) if ps > 0 else 1e6}
    if ngp > 1:
        out['ngtdm_contrast'] = float(
            (pi * pj * (li - lj) ** 2).sum() / (ngp * (ngp - 1))
            * s_tot / nv)
        denom = np.abs(li * pi - lj * pj).sum()
        out['ngtdm_busyness'] = float(ps / denom) if denom > 0 else float('nan')
        out['ngtdm_complexity'] = float(
            (np.abs(li - lj) * (pi * si + pj * sj) / (pi + pj)).sum() / nv)
        out['ngtdm_strength'] = float(
            ((pi + pj) * (li - lj) ** 2).sum() / s_tot) if s_tot > 0 else 0.0
    else:
        out.update({'ngtdm_contrast': 0.0, 'ngtdm_busyness': float('nan'),
                    'ngtdm_complexity': 0.0, 'ngtdm_strength': 0.0})
    return out


BONE_KEYS = (
    'bone_volume_mm3', 'bone_surface_area_mm2', 'bone_volume_to_surface_mm',
    'bone_sphericity', 'bone_components', 'bone_largest_comp_fraction',
    'bone_elongation', 'bone_flatness', 'bone_anisotropy', 'bone_solidity')


def bone_morphometry(vol, mask, bone_thr, voxel_mm):
    """Shape of the segmented bone (HU > threshold) inside the region.

    The region itself is a rigid stamped cylinder, so only the bone mask
    carries per-sample shape information.
    """
    out = {k: float('nan') for k in BONE_KEYS}
    bone = mask & (vol > bone_thr)
    n = int(bone.sum())
    out['bone_components'] = 0.0
    out['bone_volume_mm3'] = 0.0
    if n == 0:
        return out
    voxv = float(np.prod(voxel_mm))
    vol_mm3 = n * voxv
    out['bone_volume_mm3'] = vol_mm3
    lab, nc = ndimage.label(bone, structure=np.ones((3, 3, 3), bool))
    out['bone_components'] = float(nc)
    if nc:
        out['bone_largest_comp_fraction'] = float(
            np.bincount(lab.ravel())[1:].max() / n)
    try:
        verts, faces, _, _ = skmeasure.marching_cubes(
            np.pad(bone, 1).astype(np.float32), 0.5, spacing=voxel_mm)
        sa = float(skmeasure.mesh_surface_area(verts, faces))
        if sa > 0:
            out['bone_surface_area_mm2'] = sa
            out['bone_volume_to_surface_mm'] = vol_mm3 / sa
            out['bone_sphericity'] = float(
                np.pi ** (1.0 / 3.0) * (6.0 * vol_mm3) ** (2.0 / 3.0) / sa)
    except Exception:                       # noqa: BLE001  (degenerate mask)
        pass
    if n >= 4:
        coords = np.argwhere(bone).astype(np.float64) * np.asarray(voxel_mm)
        ev = np.clip(np.linalg.eigvalsh(np.cov((coords - coords.mean(0)).T)),
                     0.0, None)             # ascending
        if ev[2] > 0:
            out['bone_elongation'] = float(np.sqrt(ev[1] / ev[2]))
            out['bone_flatness'] = float(np.sqrt(ev[0] / ev[2]))
            out['bone_anisotropy'] = float(1.0 - ev[0] / ev[2])
        try:
            from scipy.spatial import ConvexHull
            pts = coords[::max(1, n // 50000)]     # deterministic subsample
            hull = ConvexHull(pts)
            if hull.volume > 0:
                out['bone_solidity'] = float(vol_mm3 / hull.volume)
        except Exception:                   # noqa: BLE001  (coplanar points)
            pass
    return out


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
    b = bin_volume(vol)

    feats = {}
    for name, mask in (('core', core), ('ring', ring)):
        print(f'Extracting {name} features ({int(mask.sum()):,} voxels)...')
        vals = vol[mask]
        f = first_order(vals, voxel_mm3)
        f.update(otsu_features(vals, args.bone_threshold))
        for fam, fn in (('GLCM', glcm_features), ('GLRLM', glrlm_features),
                        ('GLSZM', glszm_features), ('GLDM', gldm_features),
                        ('NGTDM', ngtdm_features)):
            print(f'  {fam}...')
            f.update(fn(b, mask))
        print('  bone morphometry...')
        f.update(bone_morphometry(vol, mask, args.bone_threshold, voxel_mm))
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
        'texture_families': 'GLCM(24) GLRLM(16) GLSZM(16) GLDM(14) NGTDM(5), '
                            'shared 32-bin discretisation, 26-neighbourhood, '
                            'directions merged',
        'glrlm_max_run': MAX_RUN, 'gldm_alpha': 0,
        'bone_morphometry': 'on region mask & HU > bone_threshold',
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
