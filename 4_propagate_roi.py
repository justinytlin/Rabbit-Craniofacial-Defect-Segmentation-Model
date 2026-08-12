"""
4_propagate_roi.py
Places the defect ROI on a LATER-timepoint scan by rigid registration from an
earlier scan whose ROI is trusted, instead of trusting the U-Net's placement.

WHY: the U-Net was trained on 3-month scans only. Validated against ground truth
it places the ROI to 0.21 mm on that timepoint, but on 6-month scans — where the
defect is part-healed and looks different — registration of the same animals'
3-month trephine sites shows the network lands 2.2–3.2 mm off, biased toward the
ingrown bone edge. Registration carries the anatomically-defined site forward and
is immune to healing-stage appearance.

METHOD (whole-volume registration fails here — the field of view contains paws,
a positioning tube and a dish that stay put while the head moves):
  - a 48 mm box of skull is extracted around the ROI centre in each scan
  - bone mask (HU > 226), largest connected component (drops the tube)
  - signed distance maps registered with mean squares, multi-resolution
  - rotation initialised from the two defect axes, multi-start over the
    remaining spin about that axis (12 x 30 degrees)
  - QC gate: bone-mask dice after registration must exceed --dice-min (0.5);
    below that NOTHING is written

The target scan still needs a 3_inference.py run first — its predicted ROI is
the search hint (and the comparison baseline). The reference ROI can be a
ground-truth _output_dicom or, when none exists, a predicted one from the
animal's 3-month scan (in-distribution for the network).

Usage:
    python 4_propagate_roi.py \
        --ref-input  /path/3m/dicom_dir   --ref-roi  /path/3m/SUBJ_output_dicom \
        --target-input /path/6m/dicom_dir --target-roi /path/6m/SUBJ_6m_output_dicom \
        --output /path/6m/SUBJ_6m_output_dicom [--bone-refine] ...

Requires SimpleITK (see requirements.txt).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pydicom

SCRIPT_DIR = Path(__file__).parent
_SPEC = importlib.util.spec_from_file_location('_inference',
                                               SCRIPT_DIR / '3_inference.py')
INF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(INF)
sys.path.insert(0, str(SCRIPT_DIR))
from axial_view import AxialView  # noqa: E402

BONE = INF.BONE_THRESHOLD_HU


# ── volume loading ───────────────────────────────────────────────────────────

def load_volume(dicom_dir: Path, ds_factor: int, cache_dir: Path = None):
    """HU volume downsampled by ds_factor. Returns (vol, spacing_zrc_mm)."""
    hdr = INF.dcm_files_sorted(dicom_dir)
    ds0 = pydicom.dcmread(str(hdr[0][1]), stop_before_pixels=True)
    ps = [float(v) for v in ds0.PixelSpacing]
    p0 = np.array([float(v) for v in ds0.ImagePositionPatient])
    p1 = np.array([float(v) for v in pydicom.dcmread(
        str(hdr[1][1]), stop_before_pixels=True).ImagePositionPatient])
    sp = np.array([float(np.linalg.norm(p1 - p0)), ps[0], ps[1]])

    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f'vol_{dicom_dir.name}_ds{ds_factor}.npz'
        if cache.exists():
            return np.load(cache)['v'], sp * ds_factor

    planes = []
    for i in range(0, len(hdr), ds_factor):
        ds = pydicom.dcmread(str(hdr[i][1]))
        planes.append(INF.pixel_to_hu(ds.pixel_array, ds)[::ds_factor, ::ds_factor])
    vol = np.stack(planes, 0).astype(np.float32)
    if cache is not None:
        np.savez_compressed(cache, v=vol)
    return vol, sp * ds_factor


# ── registration (SimpleITK) ────────────────────────────────────────────────

def zrc_to_xyz(v):
    return np.array([v[2], v[1], v[0]], dtype=np.float64)


def xyz_to_zrc(v):
    return np.array([v[2], v[1], v[0]], dtype=np.float64)


def rodrigues(k, th):
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def axis_align(a_from, a_to):
    a = a_from / np.linalg.norm(a_from)
    b = a_to / np.linalg.norm(a_to)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    return rodrigues(v, np.arccos(np.clip(c, -1, 1)))


def extract_box(vol, centre_mm, sp_zrc, half_mm):
    half = np.round(half_mm / sp_zrc).astype(int)
    c = np.round(np.asarray(centre_mm) / sp_zrc).astype(int)
    lo, hi = c - half, c + half
    pad_lo = np.maximum(-lo, 0)
    pad_hi = np.maximum(hi - np.array(vol.shape), 0)
    sub = vol[max(lo[0], 0):hi[0], max(lo[1], 0):hi[1], max(lo[2], 0):hi[2]]
    sub = np.pad(sub, list(zip(pad_lo, pad_hi)), constant_values=-1000.0)
    return sub.astype(np.float32)


def register_boxes(f_box, m_box, sp_zrc, a_fixed_zrc, a_moving_zrc,
                   wide_search: bool = False):
    """Rigid transform T: fixed box physical point -> moving box physical point.

    Box centres sit at physical origin, so a perfectly consistent pair of ROI
    centres yields T with |T^-1(0)| ~ 0.

    The default multi-start spins about the axis but TRUSTS the hint axis tilt.
    When the target hint comes from a very weak network mask its tilt can be off
    by tens of degrees, outside the optimizer's basin; wide_search adds tilt
    perturbations (±12°, ±24° about two perpendicular directions) at the cost of
    ~7× more starts.
    Returns (transform, bone_dice, best_spin_deg).
    """
    import SimpleITK as sitk
    from scipy.ndimage import label as cc_label

    def skull_mask(box):
        m = box > BONE
        lab, n = cc_label(m)
        if n == 0:
            return m
        sizes = np.bincount(lab.ravel())[1:]
        return lab == (sizes.argmax() + 1)

    def as_img(arr, dtype=np.float32):
        img = sitk.GetImageFromArray(arr.astype(dtype))
        img.SetSpacing((sp_zrc[2], sp_zrc[1], sp_zrc[0]))
        n = arr.shape
        img.SetOrigin((-sp_zrc[2] * n[2] / 2,
                       -sp_zrc[1] * n[1] / 2,
                       -sp_zrc[0] * n[0] / 2))
        return img

    def dist_img(mask):
        m = as_img(mask, np.uint8)
        d = sitk.SignedMaurerDistanceMap(m, insideIsPositive=False,
                                         squaredDistance=False,
                                         useImageSpacing=True)
        return sitk.Clamp(d, sitk.sitkFloat32, -6.0, 6.0), m

    f_dist, f_mask = dist_img(skull_mask(f_box))
    m_dist, m_mask = dist_img(skull_mask(m_box))

    a_f = zrc_to_xyz(a_fixed_zrc)
    a_m = zrc_to_xyz(a_moving_zrc)
    R0 = axis_align(a_f, a_m)

    # Tilt-perturbation pre-rotations (identity only, unless wide_search)
    tilts = [np.eye(3)]
    if wide_search:
        tmp = np.array([1.0, 0, 0]) if abs(a_m[0]) < 0.9 else np.array([0, 1.0, 0])
        p1 = np.cross(a_m, tmp); p1 /= np.linalg.norm(p1)
        p2 = np.cross(a_m, p1)
        for deg in (12.0, 24.0):
            for p in (p1, p2):
                tilts += [rodrigues(p, np.radians(deg)),
                          rodrigues(p, np.radians(-deg))]

    best = None
    for Rt in tilts:
      for k in range(12):
        Rk = Rt @ rodrigues(a_m, np.radians(30.0 * k)) @ R0
        tx = sitk.Euler3DTransform()
        tx.SetCenter((0.0, 0.0, 0.0))
        tx.SetMatrix(tuple(Rk.flatten()))
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMeanSquares()
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=300,
            relaxationFactor=0.6)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SetInitialTransform(tx, inPlace=True)
        try:
            R.Execute(f_dist, m_dist)
        except RuntimeError:
            continue
        metric = R.GetMetricValue()
        if best is None or metric < best[1]:
            best = (tx, metric, 30 * k)
    if best is None:
        raise RuntimeError('all registration starts failed')
    T, _, spin = best

    res = sitk.Resample(m_mask, f_mask, T, sitk.sitkNearestNeighbor, 0)
    a = sitk.GetArrayFromImage(f_mask) > 0
    b = sitk.GetArrayFromImage(res) > 0
    dice = 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    return T, float(dice), spin


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Place a later-timepoint ROI by rigid registration from an '
                    'earlier trusted ROI')
    p.add_argument('--ref-input', required=True, help='Earlier scan DICOM dir')
    p.add_argument('--ref-roi', required=True,
                   help='Earlier ROI series (GT _output_dicom, or predicted)')
    p.add_argument('--target-input', required=True, help='Later scan DICOM dir')
    p.add_argument('--target-roi',
                   help='Predicted ROI series on the later scan (from '
                        '3_inference.py) — used as the search hint and baseline')
    p.add_argument('--target-fit',
                   help='Alternative to --target-roi: JSON from 3_inference.py '
                        '--fit-only (centre/axis hint without written series). '
                        'Faster: the network series are never written at all.')
    p.add_argument('--output', required=True,
                   help='Output series base path (may equal --target-roi to '
                        'overwrite the network placement)')
    p.add_argument('--bone-refine', action='store_true')
    p.add_argument('--bone-threshold', type=float, default=BONE)
    p.add_argument('--no-axial-preview', action='store_true')
    p.add_argument('--downsample', type=int, default=4)
    p.add_argument('--box-half-mm', type=float, default=24.0)
    p.add_argument('--wide-search', action='store_true',
                   help='Extend the rotation multi-start with tilt perturbations '
                        '(for weak network hints whose axis tilt may be far off)')
    p.add_argument('--dice-min', type=float, default=0.5,
                   help='Abort without writing if bone dice is below this')
    p.add_argument('--cache-dir', default=None,
                   help='Optional dir to cache downsampled volumes')
    args = p.parse_args()

    ref_in, ref_roi = Path(args.ref_input), Path(args.ref_roi)
    tgt_in = Path(args.target_input)
    out = Path(args.output)
    if not args.target_roi and not args.target_fit:
        print('ERROR: give either --target-roi or --target-fit')
        sys.exit(1)
    checks = [ref_in, ref_roi, tgt_in]
    checks.append(Path(args.target_fit) if args.target_fit else Path(args.target_roi))
    for d in checks:
        if not d.exists():
            print(f'ERROR: not found: {d}')
            sys.exit(1)

    print('Fitting reference ROI...')
    av_r = AxialView(str(ref_in), str(ref_roi), fov_mm=11.0)
    print(f'  centre {np.round(av_r.center_mm, 2)} mm  axis {np.round(av_r.axis, 3)}  '
          f'eig {np.round(av_r.eigvals, 2)}')
    if abs(av_r.eigvals[0] - 5.33) > 1.0:
        print('  WARNING: reference eigenvalues far from the rigid template — '
              'is this really a stamped/GT ROI series?')

    if args.target_fit:
        import json
        fit = json.loads(Path(args.target_fit).read_text())
        hint_c = np.asarray(fit['center_mm'], dtype=np.float64)
        hint_a = np.asarray(fit['axis'], dtype=np.float64)
        print(f'Target hint from {Path(args.target_fit).name}: '
              f'centre {np.round(hint_c, 2)} mm  axis {np.round(hint_a, 3)}')
    else:
        print('Fitting target (network-predicted) ROI...')
        av_t = AxialView(str(tgt_in), str(args.target_roi), fov_mm=11.0)
        hint_c, hint_a = av_t.center_mm, av_t.axis
        print(f'  centre {np.round(hint_c, 2)} mm  axis {np.round(hint_a, 3)}')

    print(f'\nLoading volumes (downsample x{args.downsample})...')
    v_ref, sp_ref = load_volume(ref_in, args.downsample,
                                args.cache_dir and Path(args.cache_dir))
    v_tgt, sp_tgt = load_volume(tgt_in, args.downsample,
                                args.cache_dir and Path(args.cache_dir))
    if not np.allclose(sp_ref, sp_tgt, rtol=0.05):
        print(f'  note: voxel spacing differs ref {sp_ref} vs target {sp_tgt} mm')

    print('Registering skull-local boxes (12 spin starts)...')
    f_box = extract_box(v_tgt, hint_c, sp_tgt, args.box_half_mm)
    m_box = extract_box(v_ref, av_r.center_mm, sp_ref, args.box_half_mm)
    T, dice, spin = register_boxes(f_box, m_box, sp_tgt, hint_a, av_r.axis,
                                   wide_search=args.wide_search)
    print(f'  bone dice = {dice:.3f}   best spin start = {spin} deg')
    if dice < args.dice_min:
        print(f'\nERROR: dice {dice:.3f} < --dice-min {args.dice_min} — '
              'registration not trusted, nothing written.')
        sys.exit(2)

    # Map centre and axis into the target frame. T: fixed(target)->moving(ref).
    site_xyz = np.array(T.GetInverse().TransformPoint((0.0, 0.0, 0.0)))
    new_c = hint_c + xyz_to_zrc(site_xyz)
    Rm = np.array(T.GetMatrix()).reshape(3, 3)          # fixed dirs -> moving dirs
    new_a = xyz_to_zrc(Rm.T @ zrc_to_xyz(av_r.axis))
    if new_a[np.argmax(np.abs(new_a))] < 0:
        new_a = -new_a

    shift_vec = new_c - hint_c
    e1, e2 = INF.perp_basis(hint_a)
    ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
        new_a / np.linalg.norm(new_a), hint_a))), 0, 1)))
    print(f'\n── Propagated ROI ───────────────────────────────────────────')
    print(f'  centre : {np.round(new_c, 2)} mm  '
          f'(network was {np.linalg.norm(shift_vec):.2f} mm away; in-plane '
          f'({np.dot(shift_vec, e1):+.2f}, {np.dot(shift_vec, e2):+.2f}) mm)')
    print(f'  axis   : {np.round(new_a / np.linalg.norm(new_a), 4)}  '
          f'({ang:.1f} deg from network axis)')

    # ── stamp + write, mirroring 3_inference.py ─────────────────────────────
    slices = INF.dcm_files_sorted(tgt_in)
    n = len(slices)
    ds_ref = pydicom.dcmread(str(slices[0][1]), stop_before_pixels=True)
    orig_h, orig_w = ds_ref.Rows, ds_ref.Columns
    iop = [float(v) for v in ds_ref.ImageOrientationPatient]
    row_cos, col_cos = np.array(iop[0:3]), np.array(iop[3:6])
    ps = [float(v) for v in ds_ref.PixelSpacing]
    spacing = sp_tgt / args.downsample
    geom = INF.OrientedCylinder(new_c, new_a, spacing)

    half = geom.bbox_half_mm()
    lo, hi = (geom.c - half) / spacing, (geom.c + half) / spacing
    z_lo, z_hi = max(0, int(np.floor(lo[0]))), min(n - 1, int(np.ceil(hi[0])))
    row_min = max(0, int(np.round(lo[1])))
    row_max = min(orig_h, int(np.round(hi[1])) + 1)
    col_min = max(0, int(np.round(lo[2])))
    col_max = min(orig_w, int(np.round(hi[2])) + 1)
    crop_h, crop_w = row_max - row_min, col_max - col_min

    voxel_mm3 = float(np.prod(spacing))
    analytic = (np.pi * INF.CYLINDER_MM ** 2 * INF.CYL_HEIGHT_MM
                + np.pi * (INF.RING_OUTER_MM ** 2 - INF.RING_INNER_MM ** 2)
                * INF.CYL_HEIGHT_MM)
    enclosed = sum(int(geom.slice_mask(z, 'union', row_min, crop_h,
                                       col_min, crop_w).sum())
                   for z in range(z_lo, z_hi + 1)) * voxel_mm3
    print(f'\nActive Z    : {z_lo} → {z_hi}')
    print(f'ROI check   : {enclosed:.2f} / {analytic:.2f} mm³ '
          f'({100 * enclosed / analytic:.2f}% of the template)')
    if enclosed / analytic < 0.999:
        print('  WARNING: crop window clips the ROI — inspect before trusting.')

    outputs = [('union', out, None),
               ('cylinder', out.parent / (out.name + '_cylinder'), None),
               ('ring',     out.parent / (out.name + '_ring'),     None)]
    if args.bone_refine:
        thr = args.bone_threshold
        outputs += [
            ('cylinder', out.parent / (out.name + '_cylinder_bone'), thr),
            ('ring',     out.parent / (out.name + '_ring_bone'),     thr)]
    print(f'\nWriting {len(outputs)} DICOM series (single pass)...')
    INF.write_all_series(slices, geom, outputs,
                         row_min=row_min, crop_h=crop_h,
                         col_min=col_min, crop_w=crop_w,
                         row_cos=row_cos, col_cos=col_cos,
                         row_sp=ps[0], col_sp=ps[1],
                         z_lo=z_lo, z_hi=z_hi)

    off = INF.report_defect_offset(slices, geom, n, orig_h, orig_w,
                                   args.bone_threshold)
    if off:
        print(f'\nDefect-void offset from the propagated (trephine-site) centre: '
              f'({off["du"]:+.2f}, {off["dv"]:+.2f}) mm → {off["dist"]:.2f} mm, '
              f'void fraction {100 * off["void_frac"]:.1f}%')

    if not args.no_axial_preview:
        print('\nRendering axial view...')
        INF.write_axial_preview(slices, geom,
                                out.parent / (out.name + '_axial_view.png'),
                                n, orig_h, orig_w)
    print('\nDone. Placement: registration from '
          f'{ref_roi.name} (dice {dice:.3f}).')


if __name__ == '__main__':
    main()
