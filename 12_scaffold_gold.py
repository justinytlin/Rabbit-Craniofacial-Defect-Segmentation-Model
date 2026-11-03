#!/usr/bin/env python3
"""Gold-label quantification for ONE in vitro scaffold µCT scan.

Dr. Ren's nanogold study (`Radiomics/Dr. Ren/`): PLL-AuNP-labelled hMSCs
seeded on 6–8 mm CG / MC / PEG-MC scaffolds, scanned on the SCANCO µCT at
37 µm. These scans hold no calvarial defect, so the rabbit pipeline (U-Net,
18 mm template, ex vivo moat filter) can never place an ROI on them — the
2026-10-27 web-app attempt failed with "specimen never covers the reference
ring". This script is what the web app runs instead, for the `scaffold`
mode: it re-uses the study's own analysis code
(`Dr. Ren/analysis/microct_pipeline.py`, unchanged) on a single scan folder
and writes per-scan results next to the app's other outputs.

    python 12_scaffold_gold.py --input SCAN_DIR --output OUT_BASE \
        [--scan-id 6061] [--manifest .../analysis/manifest.csv] [--k 5]

Method (identical to the study report, 2026-10-26):
  scaffold envelope (> 800 raw) → bright floor plane cut at the high-z end →
  per-slice scaffold baseline + one-sided robust σ → gold = baseline + k·σ
  inside the envelope. k = 5 is the threshold calibrated on the 20
  unlabeled / no-cell controls (95th-percentile control gold ≤ 0.02 % of
  scaffold volume). Fixed raw thresholds and k = 3–6 are written too.

Writes
  OUT_BASE.json                  everything below + geometry + flags
  OUT_BASE_metrics.csv           one row per threshold (fixed + adaptive)
  OUT_BASE_depth_profile.csv     0.1 mm depth bins at the primary threshold
  OUT_BASE_qc.png                coronal / sagittal / axial overlay + depth histogram

Nothing here is comparable to a rabbit-defect BV/TV; these runs are never
added to the radiomics spreadsheet.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom

HERE = Path(__file__).resolve().parent
# defect_segmentation → PDLLA RAW… → In Vivo CT Data → Radiomics / Dr. Ren
DEFAULT_SCAFFOLD_ROOT = HERE.parent.parent.parent / 'Dr. Ren'

PRIMARY_K = 5.0
CONTROL_FP_MAX = 0.02            # % of scaffold voxels — control ceiling (k=5 calibration)
# Flag rules copied from Dr. Ren/analysis/group_stats.py
FLAG_BOTTOM_FRAC = 0.3
FLAG_BOTTOM_MIN_VOX = 500
FLAG_FLOOR_VOX = 5000
FLAG_TOP_CUT = 0.3
FLAG_THIN_MM = 2.0
CONTROL_LABELS = {'none', 'unlabeled', 'unlabelled', 'control'}


def load_pipeline(analysis_dir: Path):
    src = analysis_dir / 'microct_pipeline.py'
    if not src.is_file():
        raise SystemExit(f'scaffold pipeline not found: {src}\n'
                         '(expected Dr. Ren/analysis/microct_pipeline.py — pass --pipeline-dir)')
    spec = importlib.util.spec_from_file_location('microct_pipeline', str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_volume(scan_dir: Path):
    files = sorted(f for f in os.listdir(scan_dir)
                   if f.lower().endswith('.dcm') and not f.startswith('.'))
    if not files:
        raise SystemExit(f'no .dcm files in {scan_dir}')
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(scan_dir / f))
        except pydicom.errors.InvalidDicomError:
            ds = pydicom.dcmread(str(scan_dir / f), force=True)
        slices.append(ds)

    def zkey(s):
        if getattr(s, 'SliceLocation', None) is not None:
            return float(s.SliceLocation)
        ipp = getattr(s, 'ImagePositionPatient', None)
        if ipp is not None:
            return float(ipp[2])
        return float(getattr(s, 'InstanceNumber', 0) or 0)

    slices.sort(key=zkey)
    vol = np.stack([s.pixel_array.astype(np.int32) for s in slices])
    locs = np.array([zkey(s) for s in slices])
    dz = float(abs(np.median(np.diff(locs)))) if len(locs) > 1 else 0.0
    if not dz or math.isnan(dz):
        dz = float(getattr(slices[0], 'SliceThickness', 0) or 0) or float(slices[0].PixelSpacing[0])
    px = float(slices[0].PixelSpacing[0])
    return vol, dz, px, len(files)


def manifest_row(manifest: Path, scan_id: str):
    if not manifest or not Path(manifest).is_file() or not scan_id:
        return None
    with open(manifest, newline='') as fh:
        for row in csv.DictReader(fh):
            if (row.get('scan') or '').strip() == str(scan_id).strip():
                return row
    return None


def clean(o):
    """JSON-safe: NaN/inf → None, numpy scalars → python."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(description='Gold quantification for one scaffold µCT scan')
    ap.add_argument('--input', required=True, help='scan folder of .dcm slices')
    ap.add_argument('--output', required=True,
                    help='output base path, e.g. …/outputs/6061_scaffold/6061_scaffold')
    ap.add_argument('--scan-id', default=None, help='manifest scan id (default: input folder name)')
    ap.add_argument('--manifest', default=None, help='analysis/manifest.csv (sample identities)')
    ap.add_argument('--meta', default=None, help='JSON of user-entered metadata (web app)')
    ap.add_argument('--k', type=float, default=PRIMARY_K, help='primary adaptive threshold k (default 5)')
    ap.add_argument('--pipeline-dir', default=str(DEFAULT_SCAFFOLD_ROOT / 'analysis'),
                    help='folder holding microct_pipeline.py')
    a = ap.parse_args()

    scan_dir = Path(a.input)
    out = Path(a.output)
    scan_id = a.scan_id or scan_dir.name
    manifest = Path(a.manifest) if a.manifest else (DEFAULT_SCAFFOLD_ROOT / 'analysis' / 'manifest.csv')
    meta_user = json.loads(a.meta) if a.meta else {}

    print(f'Input   : {scan_dir}')
    print(f'Output  : {out}')
    print(f'Scan id : {scan_id}')
    row = manifest_row(manifest, scan_id)
    if row:
        cells = row.get('cells') or '0'
        try:
            cells_k = f'{int(float(cells)) // 1000}k'
        except ValueError:
            cells_k = str(cells)
        label = (f"{row.get('run')} {row.get('scaffold')} {row.get('labeled')} "
                 f"{cells_k} rep{row.get('replicate')}")
        labeled = (row.get('labeled') or '').strip().lower()
        print(f'Manifest: {label}  (batch {row.get("batch")}, Ø {row.get("diameter_mm")} mm'
              + (f', note: {row.get("note")}' if row.get('note') else '') + ')')
    else:
        label = ' '.join(str(x) for x in (meta_user.get('sample'), meta_user.get('treatment'),
                                          meta_user.get('notes')) if x) or scan_id
        labeled = ''
        print('Manifest: scan not listed — labelling unknown' if manifest.is_file()
              else 'Manifest: none')
    control = (True if labeled in CONTROL_LABELS else
               False if labeled else None)

    print('\nLoading DICOM slices...')
    vol, dz, px, n = load_volume(scan_dir)
    print(f'Found {n} DICOM slices')
    print(f'Voxel spacing (Z,row,col): [{dz:.3f} {px:.3f} {px:.3f}] mm')
    print(f'Field of view: {vol.shape[2] * px:.1f} × {vol.shape[1] * px:.1f} mm, '
          f'{vol.shape[0] * dz:.1f} mm stack')
    vox_mm3 = dz * px * px

    mp = load_pipeline(Path(a.pipeline_dir))
    print(f'\n── Scaffold gold quantification (Dr. Ren pipeline: {Path(a.pipeline_dir) / "microct_pipeline.py"}) ──')
    voi, env, depth, rnorm, geo = mp.build_voi(vol, dz)
    if int(voi.sum()) == 0:
        raise SystemExit('no scaffold envelope found — nothing above the 800 raw '
                         'scaffold threshold (is this really a scaffold scan?)')
    mode = mp.scaffold_mode(vol[voi])
    base, sigma = mp.slice_baseline(vol, voi)
    excess = vol - base[:, None, None]
    zv = slice(0, geo['z_end'])
    geo.update(dict(scan=scan_id, dz_mm=dz, px_mm=px, scaffold_vox=int(voi.sum()),
                    scaffold_volume_mm3=float(voi.sum() * vox_mm3), mode=mode, sigma=sigma,
                    baseline_median=float(np.nanmedian(base[zv])),
                    baseline_top=float(np.nanmean(base[zv][:5])),
                    baseline_bottom=float(np.nanmean(base[zv][-5:])),
                    floor_zone_gold_vox_T3000=int(((vol > 3000) & env)[geo['z_end']:].sum()),
                    label=label))
    print(f'scaffold volume   : {geo["scaffold_volume_mm3"]:.2f} mm³ ({geo["scaffold_vox"]:,} voxels)')
    print(f'analysed depth    : {geo["analysed_thickness_mm"]:.2f} mm '
          f'(VOI slices 0–{geo["z_end"]} of {geo["nz"]}; floor plane from slice {geo["floor_start"]})')
    print(f'baseline mode     : {mode:.0f} raw, σ {sigma:.0f}')
    print(f'top-cut ratio     : {geo["top_cut_ratio"]:.2f} (first-slice area / mid-stack area)')

    thresholds = ([('fixed', None, float(t)) for t in mp.FIXED_T] +
                  [('adaptive', k, float(np.nanmedian(base[zv]) + k * sigma)) for k in mp.K_VALUES])
    if a.k not in mp.K_VALUES:
        thresholds.append(('adaptive', a.k, float(np.nanmedian(base[zv]) + a.k * sigma)))
    rows, primary, gold_primary, prof_primary = [], None, None, None
    for kind, k, T in thresholds:
        gold = ((vol > T) & voi) if kind == 'fixed' else ((excess > k * sigma) & voi)
        m, gold, prof = mp.gold_metrics(vol, voi, depth, rnorm, excess, gold, T, vox_mm3)
        if gold.any():
            gz = gold.sum(axis=(1, 2))
            cut = int(0.85 * geo['z_end'])
            m['frac_gold_bottom15'] = float(gz[cut:].sum() / gz.sum())
        else:
            m['frac_gold_bottom15'] = float('nan')
        # gold deeper than 1 mm as % of scaffold deeper than 1 mm (group_stats.py's deep_density_pct)
        if prof is not None:
            deep = np.asarray(mp.DEPTH_EDGES[:-1]) >= 1.0
            g_deep = float(np.sum(np.asarray(prof['frac_of_gold'])[deep]) * m['gold_vox'])
            s_deep = float(np.sum(np.asarray(prof['scaffold_vox'])[deep]))
            m['deep_density_pct'] = 100.0 * g_deep / s_deep if s_deep else float('nan')
        else:
            m['deep_density_pct'] = 0.0
        m.update(dict(scan=scan_id, kind=kind, k=k))
        rows.append(m)
        if kind == 'adaptive' and k == a.k:
            primary, gold_primary, prof_primary = m, gold, prof

    p = primary
    print(f'\ngold threshold    : {p["T"]:.0f} raw (baseline + {a.k:g}σ)')
    print(f'gold fraction     : {p["gold_frac_pct"]:.4f}% of scaffold '
          f'({p["gold_vox"]:,} voxels, {p["gold_volume_mm3"]:.4f} mm³)')
    if p['gold_vox']:
        print(f'D50 depth         : {p["D50_mm"]:.3f} mm   (D90 {p["D90_mm"]:.3f} mm, mean {p["mean_depth_mm"]:.3f} mm)')
        print(f'within 0.25 mm    : {100 * p["frac_surface"]:.1f}%')
        print(f'deeper than 1 mm  : {100 * p["frac_deeper_1mm"]:.1f}%  (deep density {p["deep_density_pct"]:.4f}% of deep scaffold)')
        print(f'mean radial pos   : {p["mean_rnorm"]:.2f} (0 centre … 1 edge), core fraction {100 * p["frac_core"]:.0f}%')
        print(f'clusters          : {p["n_clusters"]}, largest holds {100 * p["largest_cluster_frac"]:.0f}% of the gold')
    else:
        print('no gold voxels at the primary threshold')

    flags = {
        'bottom_gold': bool(p['gold_vox'] > FLAG_BOTTOM_MIN_VOX
                            and (p['frac_gold_bottom15'] or 0) > FLAG_BOTTOM_FRAC),
        'floor_gold': bool(geo['floor_zone_gold_vox_T3000'] > FLAG_FLOOR_VOX),
        'top_cut': bool(geo['top_cut_ratio'] > FLAG_TOP_CUT),
        'thin_voi': bool(geo['analysed_thickness_mm'] < FLAG_THIN_MM),
    }
    if control is True and p['gold_frac_pct'] > CONTROL_FP_MAX:
        flags['control_above_ceiling'] = True
    if control is False and p['gold_frac_pct'] <= CONTROL_FP_MAX:
        flags['labelled_at_control_level'] = True
    for name, on in flags.items():
        if on:
            print(f'FLAG: {name}')

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ['scan', 'kind', 'k', 'T', 'gold_vox', 'gold_frac_pct', 'gold_volume_mm3',
            'gold_intensity_sum', 'D50_mm', 'D50w_mm', 'D90_mm', 'mean_depth_mm',
            'frac_surface', 'frac_0p25_1mm', 'frac_deeper_1mm', 'deep_density_pct',
            'mean_rnorm', 'frac_core', 'n_clusters', 'largest_cluster_frac',
            'median_cluster_vox', 'frac_gold_bottom15']
    csv_path = out.parent / (out.name + '_metrics.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({c: ('' if r.get(c) is None or (isinstance(r.get(c), float) and math.isnan(r[c]))
                            else r.get(c)) for c in cols})

    prof_path = out.parent / (out.name + '_depth_profile.csv')
    with open(prof_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['depth_lo_mm', 'depth_hi_mm', 'frac_of_gold', 'local_gold_pct', 'scaffold_vox'])
        if prof_primary is not None:
            for b in range(len(mp.DEPTH_EDGES) - 1):
                w.writerow([f'{mp.DEPTH_EDGES[b]:.1f}', f'{mp.DEPTH_EDGES[b + 1]:.1f}',
                            f'{prof_primary["frac_of_gold"][b]:.6f}',
                            f'{prof_primary["local_gold_pct"][b]:.6f}',
                            int(prof_primary['scaffold_vox'][b])])

    qc_path = out.parent / (out.name + '_qc.png')
    qc_meta = dict(geo, T=p['T'], k=a.k, gold_frac_pct=p['gold_frac_pct'])
    mp.qc_figure(scan_id, vol, voi, env, gold_primary, depth, qc_meta, str(qc_path))

    result = {
        'scan': scan_id, 'label': label, 'control': control,
        'manifest': row, 'user_meta': meta_user or None,
        'geometry': geo, 'primary': p, 'thresholds': rows, 'flags': flags,
        'meta': {'k': a.k, 'T': p['T'], 'control_ceiling_pct': CONTROL_FP_MAX,
                 'voxel_mm': [dz, px, px], 'n_slices': n,
                 'input': str(scan_dir), 'pipeline': str(Path(a.pipeline_dir) / 'microct_pipeline.py'),
                 'extracted': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 'files': {'metrics_csv': str(csv_path), 'depth_profile_csv': str(prof_path),
                           'qc_png': str(qc_path)}},
    }
    json_path = out.parent / (out.name + '.json')
    json_path.write_text(json.dumps(clean(result), indent=1))
    print(f'\nWrote {json_path.name}, {csv_path.name}, {prof_path.name}, {qc_path.name}')
    print(f'  in {out.parent}')


if __name__ == '__main__':
    main()
