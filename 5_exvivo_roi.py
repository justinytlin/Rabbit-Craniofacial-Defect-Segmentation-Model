#!/usr/bin/env python3
"""Place the standard ROI template on an EX VIVO specimen scan — no network.

Ex vivo scans (SCANCO µCT, ~15 µm voxels) are excised calvarial specimens: the
defect is the specimen, roughly centred in the field of view, with the plate
lying close to the slice plane. The in vivo U-Net cannot be used here — the
resolution, orientation and appearance (no soft tissue, air background) are all
far outside its training distribution, and it predicts nothing.

Localisation is instead purely geometric, using two facts about the specimen:

  1. The plate normal is the defect axis. Weighted PCA on the bone mask of a
     slab-like specimen gives the normal as the smallest-variance eigenvector
     (same math 3_inference.py uses on the network mask, applied to the bone
     itself).
  2. The trephine site is found in two passes on maps perpendicular to that
     axis: a coarse pass on the plate-thickness deficit (ring−core matched
     filter), then the real centring pass — a matched filter for the trephine
     EDGE on the HU deficit of the plate mid-slab (dark moat annulus at the
     cut radius, bright intact plate in the reference ring). The moat is the
     one feature present whether the defect is still open or already healed
     to a central bone island.

All analysis runs on a downsampled bone-fraction volume (~0.06 mm in Z,
~0.12 mm in-plane), so memory stays flat regardless of scan size. The stamped
output series are written at NATIVE resolution through 3_inference.py's
writers, so the format matches every other series in the study.

    python3 5_exvivo_roi.py --input DICOM_DIR --output OUT_BASE --bone-refine
    python3 5_exvivo_roi.py --input DICOM_DIR --output OUT_BASE --fit-only pose.json

The pose JSON is compatible with stamp_roi.py ({"center_mm": ..., "axis": ...}).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import binary_fill_holes, label as cc_label, map_coordinates
from scipy.signal import fftconvolve

_SPEC = importlib.util.spec_from_file_location(
    '_inference', Path(__file__).parent / '3_inference.py')
_INF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INF)

# Geometry-finding threshold. Deliberately higher than the 226 HU BV/TV
# threshold: at 15 µm the per-voxel noise is large, and 500 HU keeps the
# downsampled bone-fraction volume clean without losing the plate. It only
# affects WHERE the template lands, never the reported BV/TV.
PLACE_THRESHOLD_HU = 500.0

Z_STEP_TARGET_MM = 0.06     # slice sampling for the analysis volume
XY_STEP_TARGET_MM = 0.12    # in-plane block-mean for the analysis volume
MAP_FOV_MM = 16.0           # (u,v) map half-extent; wide enough to recover
                            # from a coarse start several mm off the defect
MAP_STEP_MM = 0.12
MIN_RING_COVERAGE = 0.60    # specimen must cover this much of the ring annulus
MOAT_R_IN_MM = 3.8          # trephine-edge (moat) annulus for the HU filter
MOAT_R_OUT_MM = 6.0
MISSING_RING_HU = 400.0     # ring area with no plate bone is scored as if it
                            # carried this much HU deficit (edge repulsion)


def load_analysis_volume(slices, spacing, orig_h, orig_w, place_thr):
    """Downsampled bone-fraction + HU volumes and their voxel→mm transform.

    Returns (frac, hu_ds, z_index, block, ds_spacing_mm) where full-frame mm
    coords of downsampled voxel (i,j,k) are:
        z = z_index[i] * spacing[0]
        r = (j*block + (block-1)/2) * spacing[1]
        c = (k*block + (block-1)/2) * spacing[2]
    """
    zstep = max(1, int(round(Z_STEP_TARGET_MM / spacing[0])))
    block = max(1, int(round(XY_STEP_TARGET_MM / spacing[1])))
    z_index = np.arange(0, len(slices), zstep)
    h_ds, w_ds = orig_h // block, orig_w // block

    frac = np.empty((len(z_index), h_ds, w_ds), dtype=np.float32)
    hu_ds = np.empty_like(frac)
    for i, z in enumerate(z_index):
        ds = pydicom.dcmread(str(slices[z][1]))
        hu = _INF.pixel_to_hu(ds.pixel_array, ds)
        hu = hu[:h_ds * block, :w_ds * block]
        blocks = hu.reshape(h_ds, block, w_ds, block)
        hu_ds[i] = blocks.mean(axis=(1, 3))
        frac[i] = (blocks > place_thr).mean(axis=(1, 3))
        if (i + 1) % 50 == 0 or i + 1 == len(z_index):
            print(f'  loaded {i + 1}/{len(z_index)} analysis slices')
    return frac, hu_ds, z_index, block, zstep


def ds_coords_mm(frac_shape, z_index, block, spacing):
    """Full-frame mm coordinate axes of the downsampled grid."""
    z_mm = z_index * spacing[0]
    r_mm = (np.arange(frac_shape[1]) * block + (block - 1) / 2.0) * spacing[1]
    c_mm = (np.arange(frac_shape[2]) * block + (block - 1) / 2.0) * spacing[2]
    return z_mm, r_mm, c_mm


def weighted_pca_axis(w, z_mm, r_mm, c_mm, keep=None):
    """Smallest-variance eigenvector of the bone distribution = plate normal."""
    ww = w if keep is None else w * keep
    iz, ir, ic = np.nonzero(ww > 0.05)
    wt = ww[iz, ir, ic].astype(np.float64)
    pts = np.stack([z_mm[iz], r_mm[ir], c_mm[ic]], axis=1)
    ctr = (pts * wt[:, None]).sum(axis=0) / wt.sum()
    X = pts - ctr
    cov = (X * wt[:, None]).T @ X / wt.sum()
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, 0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return ctr, axis, evals


def sample_volume(vol, pts_mm, z_index, block, spacing, zstep):
    """map_coordinates lookup of full-frame mm points in the downsampled grid."""
    iz = (pts_mm[..., 0] / spacing[0] - z_index[0]) / zstep
    ir = (pts_mm[..., 1] / spacing[1] - (block - 1) / 2.0) / block
    ic = (pts_mm[..., 2] / spacing[2] - (block - 1) / 2.0) / block
    return map_coordinates(vol, [iz, ir, ic], order=1, mode='constant', cval=0.0)


def disk_kernel(radius_mm, r_inner_mm=0.0):
    n = int(np.ceil(radius_mm / MAP_STEP_MM))
    ax = np.arange(-n, n + 1) * MAP_STEP_MM
    U, V = np.meshgrid(ax, ax, indexing='ij')
    d = np.hypot(U, V)
    k = ((d <= radius_mm) & (d >= r_inner_mm)).astype(np.float64)
    return k / k.sum()


def _uv_grid():
    uv = np.arange(-MAP_FOV_MM, MAP_FOV_MM + MAP_STEP_MM, MAP_STEP_MM)
    U, V = np.meshgrid(uv, uv, indexing='ij')
    return uv, U, V


def _ring_mid_t(bone_pts, origin, e1, e2, axis, u_c, v_c):
    """Plate mid-plane offset along the axis, from bone in the ring annulus."""
    d = bone_pts - (origin + u_c * e1 + v_c * e2)
    t_pts = d @ axis
    rad = np.sqrt(np.maximum((d ** 2).sum(-1) - t_pts ** 2, 0.0))
    in_ring = (rad >= _INF.RING_INNER_MM) & (rad <= _INF.RING_OUTER_MM)
    if not in_ring.any():
        raise RuntimeError('no bone inside the reference ring annulus')
    return float(np.median(t_pts[in_ring]))


def fit_pose(frac, hu_ds, z_index, block, zstep, spacing):
    """Locate the defect: returns dict with centre, axis, and QC metrics.

    Three stages:
      1. plate normal by weighted PCA on the bone mask; coarse centre by a
         ring−core THICKNESS matched filter (finds the general deficit region
         — plate thickness varies across a specimen, so this alone can land
         several mm off);
      2. axis re-fit on bone near that centre only, so distant curved skull
         (sagittal crest, cut edges) does not bias the normal;
      3. final centre by the trephine-edge (moat) matched filter on the HU
         DEFICIT of the plate mid-slab — see the inline comments; this is the
         step that actually centres the template.
    """
    z_mm, r_mm, c_mm = ds_coords_mm(frac.shape, z_index, block, spacing)

    # Largest connected bone component — drops loose specks and holder debris.
    lab, nlab = cc_label(frac > 0.05)
    if nlab == 0:
        raise RuntimeError('no bone found above the placement threshold')
    largest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    keep = (lab == largest).astype(np.float32)
    w = frac * keep

    core_k = disk_kernel(_INF.CYLINDER_MM)
    ring_k = disk_kernel(_INF.RING_OUTER_MM, _INF.RING_INNER_MM)
    uv, U, V = _uv_grid()

    def bone_points(weights, thr=0.10):
        iz, ir, ic = np.nonzero(weights > thr)
        return np.stack([z_mm[iz], r_mm[ir], c_mm[ic]], axis=1)

    def thickness_maps(origin, axis, e1, e2):
        pts = bone_points(w)
        t_all = (pts - origin) @ axis
        t_lo, t_hi = np.percentile(t_all, [1, 99])
        plane = (origin[None, None, :] + U[..., None] * e1[None, None, :]
                 + V[..., None] * e2[None, None, :])
        thick = np.zeros(U.shape, dtype=np.float64)
        support = np.zeros(U.shape, dtype=bool)
        for t in np.arange(t_lo - 0.5, t_hi + 0.5, MAP_STEP_MM):
            s = sample_volume(w, plane + t * axis[None, None, :],
                              z_index, block, spacing, zstep)
            thick += s * MAP_STEP_MM
            support |= s > 0.25
        return thick, support, plane, pts

    # ── Stage 1: coarse centre from the thickness deficit ────────────────────
    ctr_b, axis, _ = weighted_pca_axis(frac, z_mm, r_mm, c_mm, keep)
    e1, e2 = _INF.perp_basis(axis)
    thick, support, plane, pts = thickness_maps(ctr_b, axis, e1, e2)
    ring_cov = fftconvolve(support.astype(np.float64), ring_k, mode='same')
    score = (fftconvolve(thick, ring_k, mode='same')
             - fftconvolve(thick, core_k, mode='same'))
    score[ring_cov < MIN_RING_COVERAGE] = -np.inf
    if not np.isfinite(score).any():
        raise RuntimeError(
            'specimen never covers the reference ring — the piece is too '
            'small (or too far off-centre) for the 18 mm template')
    iu, iv = np.unravel_index(np.argmax(score), score.shape)
    t_mid = _ring_mid_t(pts, ctr_b, e1, e2, axis, uv[iu], uv[iv])
    ctr_roi = ctr_b + uv[iu] * e1 + uv[iv] * e2 + t_mid * axis

    # ── Stage 2: re-fit the axis on bone near the coarse centre only ─────────
    dz = (z_mm - ctr_roi[0]).astype(np.float32)[:, None, None]
    dr = (r_mm - ctr_roi[1]).astype(np.float32)[None, :, None]
    dc = (c_mm - ctr_roi[2]).astype(np.float32)[None, None, :]
    t = dz * axis[0] + dr * axis[1] + dc * axis[2]
    d2 = dz ** 2 + dr ** 2 + dc ** 2 - t ** 2
    restrict = ((d2 <= MAP_FOV_MM ** 2) & (np.abs(t) <= 6.0)).astype(np.float32)
    _, axis, _ = weighted_pca_axis(frac, z_mm, r_mm, c_mm, keep * restrict)
    e1, e2 = _INF.perp_basis(axis)

    # ── Stage 3: final centre from the HU deficit of the plate mid-slab ──────
    thick, support, plane, pts = thickness_maps(ctr_roi, axis, e1, e2)
    core_t = fftconvolve(thick, core_k, mode='same')
    ring_t = fftconvolve(thick, ring_k, mode='same')

    # Slab HU and slab support over ±2 mm about the plate mid-plane ONLY.
    # Using the full-thickness support here lets the beveled cut rim of the
    # specimen — thin bone reading far below plate HU — masquerade as a huge
    # deficit and drag the filter to the specimen edge; mid-slab support
    # excludes the rim because it has little bone at plate level.
    slab = np.zeros(U.shape, dtype=np.float64)
    slab_support = np.zeros(U.shape, dtype=bool)
    ts = np.arange(-2.0, 2.0 + MAP_STEP_MM, MAP_STEP_MM)
    for t in ts:
        pt = plane + t * axis[None, None, :]
        slab += sample_volume(hu_ds, pt, z_index, block, spacing, zstep)
        slab_support |= sample_volume(w, pt, z_index, block,
                                      spacing, zstep) > 0.25
    slab /= len(ts)
    if not slab_support.any():
        raise RuntimeError('no specimen support on the plate mid-slab')
    ring_cov = fftconvolve(slab_support.astype(np.float64), ring_k, mode='same')
    ref_hu = float(np.median(slab[slab_support]))

    # The specimen REGION is the bone support with its holes filled: an open
    # defect is air (no bone support), but it is enclosed by plate, and it is
    # exactly the deficit we are looking for — while everything outside the
    # specimen outline must stay neutral.
    spec2d = binary_fill_holes(slab_support)
    deficit = np.where(spec2d, np.maximum(0.0, ref_hu - slab), 0.0)

    # Matched filter for the TREPHINE EDGE: mean HU deficit in a moat annulus
    # at the cut radius, minus the deficit of the reference ring (intact
    # plate). Why an annulus and not a disk: at 6 months a treated defect is
    # often a re-mineralised central island surrounded by a dark circular
    # moat along the old cut — a disk-mean filter under-scores that and
    # wanders to any dark patch instead; the moat is present whether the
    # defect is open (dark disk ⊃ annulus) or island-healed.
    # The moat mean is normalised by its on-specimen area, while ring area
    # WITHOUT plate bone (off the specimen, or over a void) is imputed a
    # constant deficit and the whole score is weighted by the ring's bone
    # coverage. Both terms repel the specimen edge — without them the argmax
    # drifts to dark beveled cut rims, whose ring penalty would otherwise be
    # free wherever the ring hangs off the specimen. (Selected over four
    # alternative scorings on the six 261 specimens: it is the only variant
    # that matches every visually validated centre to <2.5 mm.)
    moat_k = disk_kernel(MOAT_R_OUT_MM, MOAT_R_IN_MM)
    spec_f = spec2d.astype(np.float64)
    eps = 1e-6
    moat_area = fftconvolve(spec_f, moat_k, mode='same')
    ring_area = fftconvolve(spec_f, ring_k, mode='same')
    core_area = fftconvolve(spec_f, core_k, mode='same')
    moat_mean = fftconvolve(deficit, moat_k, mode='same') / np.maximum(moat_area, eps)
    core_d = fftconvolve(deficit, core_k, mode='same') / np.maximum(core_area, eps)
    ring_pen = (fftconvolve(deficit, ring_k, mode='same')
                + MISSING_RING_HU * (1.0 - ring_area))
    d_score = (moat_mean - ring_pen) * ring_cov
    d_score[ring_cov < MIN_RING_COVERAGE] = -np.inf
    d_score[moat_area < 0.80] = -np.inf     # moat must lie inside the specimen
    if not np.isfinite(d_score).any():
        raise RuntimeError(
            'specimen never covers the reference ring — the piece is too '
            'small (or too far off-centre) for the 18 mm template')
    iu, iv = np.unravel_index(np.argmax(d_score), d_score.shape)
    u_c, v_c = uv[iu], uv[iv]
    t_mid = _ring_mid_t(pts, ctr_roi, e1, e2, axis, u_c, v_c)
    ctr_final = ctr_roi + u_c * e1 + v_c * e2 + t_mid * axis

    tilt = float(np.degrees(np.arccos(min(1.0, abs(axis[0])))))
    return {
        'center_mm': ctr_final.tolist(), 'axis': axis.tolist(),
        'tilt_deg': tilt,
        'defect_deficit_hu': float(core_d[iu, iv]),
        'plate_ref_hu': ref_hu,
        'core_thickness_mm': float(core_t[iu, iv]),
        'ring_thickness_mm': float(ring_t[iu, iv]),
        'contrast_mm': float(ring_t[iu, iv] - core_t[iu, iv]),
        'ring_coverage': float(ring_cov[iu, iv]),
        'e1': e1.tolist(), 'e2': e2.tolist(),
        '_hu_ds': hu_ds, '_z_index': z_index, '_block': block,
        '_zstep': zstep,
    }


def write_preview(pose, spacing, out_png):
    """Axial slab preview from the (already loaded) downsampled HU volume.

    3_inference.py's write_axial_preview loads a native-resolution subvolume,
    which at 15 µm would be ~10 GB — so ex vivo renders from the analysis
    volume instead. Format matches the in vivo preview (circles + centre).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ctr = np.asarray(pose['center_mm'])
    axis = np.asarray(pose['axis'])
    e1, e2 = np.asarray(pose['e1']), np.asarray(pose['e2'])
    hu_ds, z_index = pose['_hu_ds'], pose['_z_index']
    block, zstep = pose['_block'], pose['_zstep']

    fov = _INF.RING_OUTER_MM + 2.0
    uv = np.arange(-fov, fov + MAP_STEP_MM, MAP_STEP_MM)
    U, V = np.meshgrid(uv, uv, indexing='ij')
    # MEAN over the plate mid-slab, not a MIP: a maximum projection fills the
    # defect with any bright bone above/below it along the axis and hides the
    # hole (same reasoning as slab_mean in 3_inference.py).
    acc = np.zeros(U.shape, dtype=np.float64)
    ts = np.arange(-2.0, 2.0 + MAP_STEP_MM, MAP_STEP_MM)
    for t in ts:
        p = (ctr[None, None, :] + t * axis[None, None, :]
             + U[..., None] * e1[None, None, :] + V[..., None] * e2[None, None, :])
        acc += sample_volume(hu_ds, p, z_index, block, spacing, zstep)
    acc /= len(ts)

    nz = acc[acc > 0]
    vmin, vmax = (np.percentile(nz, [1, 99]) if nz.size else (0.0, 1.0))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(acc.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
              extent=[uv[0], uv[-1], uv[0], uv[-1]])
    for r, col, lab in [(_INF.CYLINDER_MM, 'cyan', '10 mm core'),
                        (_INF.RING_INNER_MM, 'yellow', '14 mm ring ID'),
                        (_INF.RING_OUTER_MM, 'lime', '18 mm ring OD')]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls='--', lw=1.6,
                                color=col, label=lab))
    ax.plot(0, 0, 'r+', ms=10)
    ax.set_xlabel('mm'); ax.set_ylabel('mm')
    ax.set_title('Ex vivo axial view perp. to plate normal '
                 '(±2 mm plate mid-slab mean)\n'
                 f'axis (Z,row,col) = {np.round(axis, 3)}', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_png.name}')


def main():
    p = argparse.ArgumentParser(
        description='Ex vivo ROI placement — geometric, no network')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fit-only', metavar='JSON',
                   help='write the pose JSON and preview only; no DICOM series')
    p.add_argument('--place-threshold', type=float, default=PLACE_THRESHOLD_HU,
                   help='HU threshold for FINDING the defect (geometry only; '
                        f'default {PLACE_THRESHOLD_HU:.0f})')
    p.add_argument('--bone-refine', action='store_true')
    p.add_argument('--bone-threshold', type=float, default=_INF.BONE_THRESHOLD_HU,
                   help='HU threshold for the _bone BV/TV series '
                        f'(default {_INF.BONE_THRESHOLD_HU:.0f})')
    p.add_argument('--margin-mm', type=float, default=_INF.BBOX_MARGIN_MM)
    p.add_argument('--no-axial-preview', action='store_true')
    args = p.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    if not input_dir.is_dir():
        print(f'ERROR: input directory not found: {input_dir}'); sys.exit(1)

    print(f'Input   : {input_dir}')
    print(f'Output  : {output_dir}')
    print('\nLoading DICOM headers...')
    slices = _INF.dcm_files_sorted(input_dir)
    n = len(slices)
    print(f'Found {n} DICOM slices')

    ds_ref = pydicom.dcmread(str(slices[0][1]), stop_before_pixels=True)
    orig_h, orig_w = ds_ref.Rows, ds_ref.Columns
    iop = [float(v) for v in ds_ref.ImageOrientationPatient]
    row_cos, col_cos = np.array(iop[0:3]), np.array(iop[3:6])
    ps = [float(v) for v in ds_ref.PixelSpacing]
    if n > 1:
        p0 = np.array([float(v) for v in ds_ref.ImagePositionPatient])
        p1 = np.array([float(v) for v in pydicom.dcmread(
            str(slices[1][1]), stop_before_pixels=True).ImagePositionPatient])
        slice_sp = float(np.linalg.norm(p1 - p0))
    else:
        slice_sp = float(getattr(ds_ref, 'SliceThickness', ps[0]))
    spacing = np.array([slice_sp, ps[0], ps[1]])
    print(f'Voxel spacing (Z,row,col): {spacing} mm')
    print(f'Field of view: {orig_w * ps[1]:.1f} × {orig_h * ps[0]:.1f} mm, '
          f'{n * slice_sp:.1f} mm stack')

    print('\n── Ex vivo geometric placement (no network) ─────────────────')
    print('Loading downsampled analysis volume...')
    frac, hu_ds, z_index, block, zstep = load_analysis_volume(
        slices, spacing, orig_h, orig_w, args.place_threshold)

    pose = fit_pose(frac, hu_ds, z_index, block, zstep, spacing)
    center_mm = np.asarray(pose['center_mm'])
    axis = np.asarray(pose['axis'])
    print(f'\n  centre (mm)          : {np.round(center_mm, 2)}')
    print(f'  axis   (Z,row,col)   : {np.round(axis, 4)}')
    print(f'  tilt from slice axis : {pose["tilt_deg"]:.1f}°')
    print(f'  defect HU deficit    : {pose["defect_deficit_hu"]:.0f} HU '
          f'(core mean vs plate median {pose["plate_ref_hu"]:.0f} HU)')
    print(f'  plate thickness — ring: {pose["ring_thickness_mm"]:.2f} mm, '
          f'core: {pose["core_thickness_mm"]:.2f} mm')
    print(f'  ring bone coverage   : {100 * pose["ring_coverage"]:.1f}%')
    if pose['defect_deficit_hu'] < 200:
        print('  WARNING: weak defect HU deficit — the defect may be fully '
              'bridged; verify placement on the preview before trusting it.')

    if not args.no_axial_preview:
        print('\nRendering axial view (⟂ to axis)...')
        write_preview(pose, spacing,
                      output_dir.parent / (output_dir.name + '_axial_view.png'))

    if args.fit_only:
        out = {k: v for k, v in pose.items() if not k.startswith('_')}
        out['note'] = 'ex vivo geometric placement (5_exvivo_roi.py)'
        Path(args.fit_only).parent.mkdir(parents=True, exist_ok=True)
        Path(args.fit_only).write_text(json.dumps(out, indent=1))
        print(f'\nPose written to {args.fit_only} — no DICOM series (--fit-only).')
        return

    geom = _INF.OrientedCylinder(center_mm, axis, spacing)

    # Crop box + enclosure check, same math and prints as stamp_roi.py.
    half = geom.bbox_half_mm() + args.margin_mm
    lo = (geom.c - half) / spacing
    hi = (geom.c + half) / spacing
    z_lo = max(0, int(np.floor(lo[0])))
    z_hi = min(n - 1, int(np.ceil(hi[0])))
    row_min = max(0, int(np.round(lo[1])))
    row_max = min(orig_h, int(np.round(hi[1])) + 1)
    col_min = max(0, int(np.round(lo[2])))
    col_max = min(orig_w, int(np.round(hi[2])) + 1)
    crop_h, crop_w = row_max - row_min, col_max - col_min

    print(f'\nActive Z    : {z_lo} → {z_hi}  ({z_hi - z_lo + 1} slices)')
    voxel_mm3 = float(np.prod(spacing))
    analytic = (np.pi * _INF.CYLINDER_MM ** 2 * _INF.CYL_HEIGHT_MM
                + np.pi * (_INF.RING_OUTER_MM ** 2 - _INF.RING_INNER_MM ** 2)
                * _INF.CYL_HEIGHT_MM)
    enclosed = sum(int(geom.slice_mask(z, 'union', row_min, crop_h,
                                       col_min, crop_w).sum())
                   for z in range(z_lo, z_hi + 1)) * voxel_mm3
    frac_enc = enclosed / analytic
    print(f'ROI check   : {enclosed:.2f} / {analytic:.2f} mm³ enclosed '
          f'({100 * frac_enc:.2f}% of the template)')
    if frac_enc < 0.999:
        print(f'  WARNING: the crop window clips {100 * (1 - frac_enc):.2f}% of '
              'the ROI. Volumes and BV/TV from this run will be under-reported.')

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [('union', output_dir, None),
               ('cylinder', output_dir.parent / (output_dir.name + '_cylinder'), None),
               ('ring', output_dir.parent / (output_dir.name + '_ring'), None)]
    if args.bone_refine:
        thr = args.bone_threshold
        outputs += [
            ('cylinder', output_dir.parent / (output_dir.name + '_cylinder_bone'), thr),
            ('ring', output_dir.parent / (output_dir.name + '_ring_bone'), thr)]
    print(f'\nWriting {len(outputs)} DICOM series (single pass, native '
          f'{spacing[1] * 1000:.0f} µm resolution)...')
    _INF.write_all_series(slices, geom, outputs,
                          row_min=row_min, crop_h=crop_h,
                          col_min=col_min, crop_w=crop_w,
                          row_cos=row_cos, col_cos=col_cos,
                          row_sp=ps[0], col_sp=ps[1], z_lo=z_lo, z_hi=z_hi)
    print('\nDone.')


if __name__ == '__main__':
    main()
