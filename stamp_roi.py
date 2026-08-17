#!/usr/bin/env python3
"""Stamp the rigid ROI template at an explicitly given pose — no network.

Used by the web app's manual-adjustment tool: the pose JSON holds a centre and
axis (typically a small in-plane nudge of an existing run's fit), and this
script writes the same five DICOM series 3_inference.py would, at that pose.

    python3 stamp_roi.py --input DICOM_DIR --output OUT_BASE \
        --pose pose.json --bone-refine

pose.json: {"center_mm": [z,r,c], "axis": [z,r,c], "note": "optional"}

Reuses 3_inference.py's geometry and writers so the output format is
byte-compatible with the rest of the pipeline.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pydicom

_SPEC = importlib.util.spec_from_file_location(
    '_inference', Path(__file__).parent / '3_inference.py')
_INF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INF)


def main():
    p = argparse.ArgumentParser(description='Stamp ROI template at a given pose')
    p.add_argument('--input',  required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--pose',   required=True, help='JSON with center_mm and axis')
    p.add_argument('--bone-refine', action='store_true')
    p.add_argument('--bone-threshold', type=float, default=_INF.BONE_THRESHOLD_HU)
    p.add_argument('--margin-mm', type=float, default=_INF.BBOX_MARGIN_MM)
    p.add_argument('--no-axial-preview', action='store_true')
    args = p.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    if not input_dir.is_dir():
        print(f'ERROR: input directory not found: {input_dir}'); sys.exit(1)
    pose = json.loads(Path(args.pose).read_text())
    center_mm = np.asarray(pose['center_mm'], dtype=float)
    axis = np.asarray(pose['axis'], dtype=float)
    axis /= np.linalg.norm(axis)

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

    print('\n── Stamping template at given pose (no network) ─────────────')
    print(f'  centre (mm)          : {np.round(center_mm, 2)}')
    print(f'  axis   (Z,row,col)   : {np.round(axis, 4)}')
    tilt = float(np.degrees(np.arccos(min(1.0, abs(axis[0])))))
    print(f'  tilt from slice axis : {tilt:.1f}°')
    if pose.get('note'):
        print(f'  note                 : {pose["note"]}')

    geom = _INF.OrientedCylinder(center_mm, axis, spacing)

    # Crop box + ROI-enclosure check, same math and prints as 3_inference.py.
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
    frac = enclosed / analytic
    print(f'ROI check   : {enclosed:.2f} / {analytic:.2f} mm³ enclosed '
          f'({100 * frac:.2f}% of the template)')
    if frac < 0.999:
        print(f'  WARNING: the crop window clips {100 * (1 - frac):.2f}% of the '
              'ROI. Volumes and BV/TV from this run will be under-reported.')

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [('union', output_dir, None),
               ('cylinder', output_dir.parent / (output_dir.name + '_cylinder'), None),
               ('ring', output_dir.parent / (output_dir.name + '_ring'), None)]
    if args.bone_refine:
        thr = args.bone_threshold
        outputs += [
            ('cylinder', output_dir.parent / (output_dir.name + '_cylinder_bone'), thr),
            ('ring', output_dir.parent / (output_dir.name + '_ring_bone'), thr)]
    print(f'\nWriting {len(outputs)} DICOM series (single pass)...')
    _INF.write_all_series(slices, geom, outputs,
                          row_min=row_min, crop_h=crop_h,
                          col_min=col_min, crop_w=crop_w,
                          row_cos=row_cos, col_cos=col_cos,
                          row_sp=ps[0], col_sp=ps[1], z_lo=z_lo, z_hi=z_hi)

    if not args.no_axial_preview:
        print('\nRendering axial view (⟂ to axis)...')
        _INF.write_axial_preview(slices, geom,
                                 output_dir.parent / (output_dir.name + '_axial_view.png'),
                                 n, orig_h, orig_w)
    print('\nDone.')


if __name__ == '__main__':
    main()
