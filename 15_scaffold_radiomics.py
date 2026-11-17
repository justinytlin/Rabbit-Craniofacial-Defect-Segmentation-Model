#!/usr/bin/env python3
"""Radiomic features for Dr. Ren's in vitro scaffold scans (CG / MC / PEG-MC).

The gold-label analysis (12_scaffold_gold.py) answers "where are the cells".
This script answers the radiomics question instead: what does the scaffold
itself look like on µCT — intensity, texture, dense-phase morphometry — so
the three materials can be compared feature by feature.

It re-uses the feature functions of 6_extract_features.py unchanged (the same
113 IBSI-style features as the rabbit database: first-order, Otsu, GLCM,
GLRLM, GLSZM, GLDM, NGTDM, morphometry of the supra-threshold phase), on two
regions per scan:

  whole     the scaffold volume of interest (envelope, floor plane cut,
            2 px lateral erosion) — exactly the VOI of the gold analysis
  interior  the same VOI deeper than 0.25 mm below the seeded surface: the
            scaffold matrix WITHOUT the labelled-cell crust, i.e. the
            material-intrinsic region

Differences from the rabbit extraction, all forced by the specimen:
  * no cylinder template — the region is the irregular scaffold mask;
  * texture grey levels: 32 bins over a fixed −600…1000 HU window (50 HU
    bins). The bone window (−1000…3000 HU, 125 HU bins) would put ~87 % of
    scaffold voxels into three grey levels;
  * the 226 HU "bone" threshold here selects the DENSE PHASE (mineral
    particles, gold), not bone: bvtv_fixed = dense fraction, bone_* = dense-
    phase morphometry.
Working grid: 2×2 in-plane block average + Z stride 2 → 0.074 mm voxels, the
extractor's own rule for 37 µm scans (TARGET_VOXEL_MM = 0.06).

    python 15_scaffold_radiomics.py [--workers 4] [--scans 6061 6064 …] [--force]

Writes Dr. Ren/results/radiomics/per_scan/<scan>.json and
       Dr. Ren/results/radiomics/scaffold_radiomics.csv (one row per scan,
       whole_<feature> / interior_<feature> columns + manifest identity).
Compare features only between scaffold runs extracted this way — never with
the rabbit database (different region, window and voxel size).
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pydicom

REPO = Path(__file__).resolve().parent
SCAFFOLD_ROOT = REPO.parent.parent.parent / 'Dr. Ren'
OUT_DIR = SCAFFOLD_ROOT / 'results' / 'radiomics'

TEX_LO, TEX_HI = -600.0, 1000.0      # fixed texture window (HU) → 32 bins of 50 HU
DENSE_THR_HU = 226.0                 # the extractor's fixed threshold (dense phase here)
CRUST_MM = 0.25                      # interior = deeper than this below the surface
BLOCK = 2                            # 0.037 mm → 0.074 mm working voxels
MIN_REGION_VOX = 2000


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, REPO / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _modules():
    sys.path.insert(0, str(REPO))
    sg = _load('scaffold_gold', '12_scaffold_gold.py')
    fx = _load('extract_features', '6_extract_features.py')
    mp = sg.load_pipeline(SCAFFOLD_ROOT / 'analysis')
    return sg, fx, mp


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return o


def region_features(fx, hu, b, mask, voxel_mm):
    """The extractor's full feature set on one region mask."""
    vals = hu[mask]
    f = fx.first_order(vals, float(np.prod(voxel_mm)))
    f.update(fx.otsu_features(vals, DENSE_THR_HU))
    for fn in (fx.glcm_features, fx.glrlm_features, fx.glszm_features,
               fx.gldm_features, fx.ngtdm_features):
        f.update(fn(b, mask))
    f.update(fx.bone_morphometry(hu, mask, DENSE_THR_HU, voxel_mm))
    return f


def process(row):
    scan, force = row['scan'], row.pop('_force', False)
    out = OUT_DIR / 'per_scan' / f'{scan}.json'
    if out.exists() and not force:
        return json.loads(out.read_text())
    t0 = time.time()
    sg, fx, mp = _modules()
    scan_dir = SCAFFOLD_ROOT / str(row['batch']) / scan
    vol, dz, px, n = sg.load_volume(scan_dir)
    f0 = sorted(f for f in os.listdir(scan_dir)
                if f.lower().endswith('.dcm') and not f.startswith('.'))[0]
    ds = pydicom.dcmread(str(scan_dir / f0), stop_before_pixels=True)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    inter = float(getattr(ds, 'RescaleIntercept', 0.0))
    voi, env, depth, rnorm, geo = mp.build_voi(vol, dz)
    hu = vol.astype(np.float32) * slope + inter
    interior = voi & (depth > CRUST_MM)

    k = BLOCK
    nz, ny, nx = hu.shape
    ny2, nx2 = ny // k, nx // k
    zs = list(range(0, nz, k))

    def down_mean(a):
        return a[zs][:, :ny2 * k, :nx2 * k].reshape(len(zs), ny2, k, nx2, k).mean(axis=(2, 4))

    w = down_mean(hu).astype(np.float32)
    masks = {'whole': down_mean(voi.astype(np.float32)) >= 0.5,
             'interior': down_mean(interior.astype(np.float32)) >= 0.5}
    voxel_mm = (dz * k, px * k, px * k)
    b = np.clip(((w - TEX_LO) / (TEX_HI - TEX_LO) * fx.GLCM_BINS).astype(np.int16),
                0, fx.GLCM_BINS - 1)
    feats = {}
    for name, m in masks.items():
        feats[name] = region_features(fx, w, b, m, voxel_mm) if int(m.sum()) >= MIN_REGION_VOX else None
    res = {
        'scan': scan, 'manifest': {k_: v for k_, v in row.items()},
        'features': feats,
        'meta': {
            'regions': {'whole': 'scaffold VOI (gold-analysis VOI)',
                        'interior': f'VOI deeper than {CRUST_MM} mm below the seeded surface'},
            'voxel_mm': list(voxel_mm), 'native_voxel_mm': [dz, px, px], 'block': k,
            'texture_window_hu': [TEX_LO, TEX_HI], 'texture_bins': int(fx.GLCM_BINS),
            'dense_threshold_hu': DENSE_THR_HU, 'rescale': [slope, inter],
            'region_voxels': {k_: int(m.sum()) for k_, m in masks.items()},
            'n_slices': n, 'seconds': round(time.time() - t0, 1),
            'extracted': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'feature_code': '6_extract_features.py (functions re-used unchanged)'},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(res), indent=1))
    return clean(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--scans', nargs='*', help='subset of scan ids')
    ap.add_argument('--force', action='store_true', help='re-extract even if a per-scan JSON exists')
    a = ap.parse_args()
    with open(SCAFFOLD_ROOT / 'analysis' / 'manifest.csv', newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r.get('on_disk') == '1']
    if a.scans:
        rows = [r for r in rows if r['scan'] in set(a.scans)]
    for r in rows:
        r['_force'] = a.force
    (OUT_DIR / 'per_scan').mkdir(parents=True, exist_ok=True)
    print(f'{len(rows)} scans → {OUT_DIR}')
    results = []
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process, rows), 1):
            m = res['meta']
            print(f'[{i}/{len(rows)}] {res["scan"]}: whole {m["region_voxels"]["whole"]:,} vox, '
                  f'interior {m["region_voxels"]["interior"]:,} vox, {m["seconds"]} s', flush=True)
            results.append(res)
    results.sort(key=lambda r: r['scan'])
    ident = ['scan', 'run', 'batch', 'scaffold', 'labeled', 'cells', 'replicate', 'diameter_mm', 'note']
    feat_names = next(list(r['features']['whole'].keys()) for r in results if r['features']['whole'])
    cols = ident + ['voxel_mm', 'whole_region_voxels', 'interior_region_voxels'] \
        + [f'whole_{k}' for k in feat_names] + [f'interior_{k}' for k in feat_names]
    with open(OUT_DIR / 'scaffold_radiomics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            rec = {k: r['manifest'].get(k, '') for k in ident}
            rec['voxel_mm'] = r['meta']['voxel_mm'][1]
            for reg in ('whole', 'interior'):
                rec[f'{reg}_region_voxels'] = r['meta']['region_voxels'][reg]
                for k in feat_names:
                    v = (r['features'][reg] or {}).get(k)
                    rec[f'{reg}_{k}'] = '' if v is None else v
            w.writerow(rec)
    print(f'Wrote scaffold_radiomics.csv: {len(results)} scans × {len(feat_names)} features × 2 regions')


if __name__ == '__main__':
    main()
