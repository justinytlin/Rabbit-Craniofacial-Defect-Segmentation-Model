#!/usr/bin/env python3
"""Build the study-wide radiomics database.

ONE sheet, one row per scan. The spreadsheet is built incrementally and
AUTOMATICALLY from the web app: every run that finishes without a FAILed
check is added by the app the moment it completes (and a manual readjustment
of a run re-adds it, flagged in the `manually_adjusted` column). Runs that
error or fail a check are never added.

  radiomics_database/radiomics_database.xlsx
      README              what the columns mean, how to use the table
      database            one row per SCAN: identity, placement, QC, and
                          core_<feature> / ring_<feature> /
                          core_to_ring_<feature> for all 113 features
      feature_dictionary  family / description / units for every feature
  plus radiomics_database.csv (the `database` sheet as CSV).

Rows are keyed by the scanner StudyID (falling back to the scan folder), so
re-adding a scan — from any export folder of the same scan — replaces its row
rather than duplicating it.

A full rebuild (`python 7_build_database.py`) is also available: it walks the
archive, extracts any missing features, and writes one row per scan using the
best series on disk (ground truth first, then the standard automatic series;
a manual `_adj` series is used only when nothing else exists). Scans with no
usable series get a stub row so nothing silently disappears.

Rules baked in (see README.md):
  * Series superseded by the 2026-08-04 axis fix are excluded.
  * Ex vivo (15 µm) rows are flagged and must not be compared with in vivo.

Usage:  python 7_build_database.py [--no-extract] [--out DIR]
        python 7_build_database.py --add-job JOB_ID   # upsert one web-app run
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings('ignore', category=FutureWarning)   # pandas concat dtype notice

REPO_DIR = Path(__file__).resolve().parent
DATA_ROOT = REPO_DIR.parent
EXVIVO_ROOT = DATA_ROOT.parent / 'Ex Vivo CT Data'
PYTHON = sys.executable
JOBS_FILE = REPO_DIR / 'logs' / 'webapp' / 'jobs.json'
QC_DIR = DATA_ROOT / 'segmentation_qc_2026-08-11'
SERIES_SUFFIXES = ('_cylinder', '_ring', '_cylinder_bone', '_ring_bone')
# 'outputs' is the central per-run outputs folder (DATA_ROOT/outputs/...) —
# it holds ROI series, never raw scans, and must not be walked as one.
SKIP_DIRS = {'defect_segmentation', '.venv', 'radiomics_database', 'outputs'}
MIN_SLICES = 300
# Series known to be superseded (README of the 2026-08-11 QC batch).
SUPERSEDED = {'37951_9m_v2_output_dicom'}
LOOSE = []          # (dir, n_files) of stray DICOM files found outside scan folders
BONE_THRESHOLD_HU = 226.0
EXPECTED_FEATURES = 113   # per region; older JSONs (38 features) are re-extracted
# Human visual review of the axial preview (date: reviewer note). Keyed by
# series name; shown in the `visual_qc_note` column.
VISUAL_QC_NOTES = {
    '184_4194_exvivo_output_dicom': '2026-09-07 preview review: defect and moat clearly visible, core centred on the re-mineralised island; specimen is only 5 mm thick so the 8 mm template is clipped (volumes under-reported).',
    '261_5809_exvivo_output_dicom': '2026-09-07 preview review: defect centred in the core; the specimen crop clips the reference ring (partial annulus).',
    '261_5810_exvivo_output_dicom': '2026-09-07 preview review: SUSPECT — core sits on bright intact bone at the specimen edge and no defect is visible in the preview; the specimen fills only one quadrant of the field. Do not use without re-review / manual placement.',
    '261_5811_exvivo_output_dicom': '2026-09-07 preview review: defect centred in the core, ring on bone.',
    't60939_3m_pred_output_dicom': 'Anonymised 3-month scan of unknown animal; mask-eigenvalue warning — inspect the axial preview before use.',
    't60940_3m_pred_output_dicom': 'Anonymised 3-month scan of unknown animal; all checks passed.',
}

FEATURE_FAMILIES = [
    ('voxels', 'region', 'Number of voxels in the region on the working grid', 'count'),
    ('volume_mm3', 'region', 'Region volume (voxels x voxel volume)', 'mm^3'),
    ('mean_hu', 'first-order', 'Mean intensity', 'HU'),
    ('std_hu', 'first-order', 'Standard deviation of intensity', 'HU'),
    ('min_hu', 'first-order', 'Minimum intensity', 'HU'),
    ('max_hu', 'first-order', 'Maximum intensity', 'HU'),
    ('p10_hu', 'first-order', '10th percentile', 'HU'),
    ('p25_hu', 'first-order', '25th percentile', 'HU'),
    ('median_hu', 'first-order', 'Median', 'HU'),
    ('p75_hu', 'first-order', '75th percentile', 'HU'),
    ('p90_hu', 'first-order', '90th percentile', 'HU'),
    ('iqr_hu', 'first-order', 'Interquartile range (p75 - p25)', 'HU'),
    ('range_hu', 'first-order', 'max - min', 'HU'),
    ('skewness', 'first-order', 'Skewness of the intensity distribution', '-'),
    ('kurtosis', 'first-order', 'Excess kurtosis of the intensity distribution', '-'),
    ('rms_hu', 'first-order', 'Root mean square intensity', 'HU'),
    ('mean_abs_dev_hu', 'first-order', 'Mean absolute deviation from the mean', 'HU'),
    ('entropy_bits', 'first-order', 'Histogram entropy (25 HU bins, -1000..3000 HU)', 'bits'),
    ('uniformity', 'first-order', 'Histogram uniformity (sum of squared bin probabilities)', '-'),
    ('bvtv_fixed', 'Otsu / bone', 'Bone volume fraction at the fixed study threshold (226 HU) — the headline BV/TV', 'fraction'),
    ('bone_mean_hu', 'Otsu / bone', 'Mean HU of voxels above the fixed threshold (tissue-mineral-density proxy)', 'HU'),
    ('otsu_threshold_hu', 'Otsu / bone', 'Otsu threshold computed inside the region (diagnostic: in an in vivo core with air it separates air from tissue, not bone from soft tissue)', 'HU'),
    ('bvtv_otsu', 'Otsu / bone', 'Fraction of voxels above the per-region Otsu threshold', 'fraction'),
    ('multiotsu_t1_hu', 'Otsu / bone', '3-class multi-Otsu lower threshold', 'HU'),
    ('multiotsu_t2_hu', 'Otsu / bone', '3-class multi-Otsu upper threshold', 'HU'),
    ('fraction_low', 'Otsu / bone', 'Fraction of voxels in the low multi-Otsu class', 'fraction'),
    ('fraction_mid', 'Otsu / bone', 'Fraction of voxels in the middle multi-Otsu class', 'fraction'),
    ('fraction_high', 'Otsu / bone', 'Fraction of voxels in the high multi-Otsu class', 'fraction'),
]
FAMILY_PREFIX = {
    'glcm_': ('GLCM', '3D grey-level co-occurrence, 13 directions merged, distance 1, 32 bins'),
    'glrlm_': ('GLRLM', '3D grey-level run-length, directions merged, 32 bins'),
    'glszm_': ('GLSZM', '3D grey-level size-zone, 26-connectivity, 32 bins'),
    'gldm_': ('GLDM', '3D grey-level dependence, 26-neighbourhood, alpha 0, 32 bins'),
    'ngtdm_': ('NGTDM', '3D neighbourhood grey-tone difference, 26-neighbourhood, 32 bins'),
    'bone_': ('bone morphometry', 'Shape of the segmented bone (HU > 226) inside the region'),
}
BONE_MORPH_DESC = {
    'bone_volume_mm3': 'Volume of bone voxels in the region',
    'bone_surface_area_mm2': 'Marching-cubes surface area of the bone mask',
    'bone_volume_to_surface_mm': 'Bone volume / surface area',
    'bone_sphericity': 'Sphericity of the bone mask (1 = sphere)',
    'bone_components': 'Number of 26-connected bone components',
    'bone_largest_comp_fraction': 'Fraction of bone volume in the largest component',
    'bone_elongation': 'Inertia-eigenvalue elongation of the bone mask',
    'bone_flatness': 'Inertia-eigenvalue flatness of the bone mask',
    'bone_anisotropy': 'Inertia-eigenvalue anisotropy of the bone mask',
    'bone_solidity': 'Bone volume / convex-hull volume',
}


# ────────────────────────────────────────────────────────────── helpers

def dcm_names(d: Path):
    try:
        return [f for f in os.listdir(d)
                if f.lower().endswith('.dcm') and not f.startswith('.')]
    except OSError:
        return []


def is_union_series(name: str) -> bool:
    return 'output_dicom' in name and not name.endswith(SERIES_SUFFIXES)


def is_ground_truth(name: str) -> bool:
    return re.fullmatch(r'\d+_output_dicom', name) is not None


def read_header(d: Path) -> dict:
    fs = sorted(dcm_names(d))
    if not fs:
        return {}
    ds = pydicom.dcmread(str(d / fs[0]), stop_before_pixels=True)
    ps = getattr(ds, 'PixelSpacing', None)
    out = {
        'n_slices': len(fs),
        'study_id': str(getattr(ds, 'StudyID', '') or ''),
        'patient_id': str(getattr(ds, 'PatientID', '') or ''),
        'patient_name': str(getattr(ds, 'PatientName', '') or ''),
        'study_date': str(getattr(ds, 'StudyDate', '') or ''),
        'manufacturer': str(getattr(ds, 'Manufacturer', '') or ''),
        'pixel_spacing_mm': float(ps[0]) if ps is not None else None,
        'rows': int(getattr(ds, 'Rows', 0)),
        'columns': int(getattr(ds, 'Columns', 0)),
    }
    if len(fs) > 1:
        try:
            p0 = np.array([float(v) for v in ds.ImagePositionPatient])
            ds1 = pydicom.dcmread(str(d / fs[1]), stop_before_pixels=True)
            p1 = np.array([float(v) for v in ds1.ImagePositionPatient])
            out['slice_spacing_mm'] = float(np.linalg.norm(p1 - p0))
        except Exception:                               # noqa: BLE001
            out['slice_spacing_mm'] = float(getattr(ds, 'SliceThickness', 0) or 0)
    return out


def load_jobs() -> dict:
    """Web-app run history keyed by output path."""
    try:
        jobs = json.loads(JOBS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    by_out = {}
    for j in jobs.values():
        out = (j.get('params') or {}).get('output')
        if out:
            by_out[str(Path(out))] = j
    return by_out


def load_qc() -> dict:
    """2026-08-11 batch QC keyed by (subject, months)."""
    qc = {}
    f = QC_DIR / 'placement_results.csv'
    if f.exists():
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                qc[(r['sid'], int(r['tp']))] = r
    # Cohort table: covers the 5 registration runs placed before the
    # placement log started, and the 41122 3-month prediction.
    f = QC_DIR / 'cohort_bvtv_226HU.csv'
    if f.exists():
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                key = (r['sid'], int(r['tp']))
                if key not in qc and r['status'] == 'ok':
                    qc[key] = {'sid': r['sid'], 'tp': r['tp'], 'group': r['group'],
                               'mode': 'propagated' if int(r['tp']) != 3 else 'infer-only',
                               'dice': '', 'shift_mm': '',
                               'void': f'cohort table: ok, tilt {r["tilt"]} deg, core BV/TV {r["core_bvtv"]}%, ring {r["ring_bvtv"]}%'}
    return qc


def tilt_from_axis(axis):
    a = abs(float(axis[0]))
    return float(np.degrees(np.arccos(min(1.0, a))))


# ────────────────────────────────────────────────────────────── inventory

def find_scans():
    """Yield dicts describing every raw scan (>= MIN_SLICES .dcm files)."""
    scans = []
    roots = [DATA_ROOT] + ([EXVIVO_ROOT] if EXVIVO_ROOT.is_dir() else [])
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if not d.startswith('.')
                           and d not in SKIP_DIRS
                           and not d.startswith('segmentation_qc')
                           and 'output_dicom' not in d)
            p = Path(dp)
            n = sum(1 for f in fn if f.lower().endswith('.dcm') and not f.startswith('.'))
            if n < MIN_SLICES:
                continue
            if dn:
                # Loose DICOM files sitting in a group / timepoint folder that
                # also holds subject folders — not a scan; keep descending.
                LOOSE.append((str(p.relative_to(root.parent)), n))
                continue
            scans.append(classify_scan(p, root))
    return scans


def classify_scan(d: Path, root: Path) -> dict:
    rel = d.relative_to(root.parent)
    parts = d.parts
    hdr = read_header(d)
    info = {'scan_dir': str(d), 'scan_rel': str(rel), 'scan_type': 'in vivo',
            'treatment': None, 'subject': None, 'timepoint': None,
            'timepoint_months': None, 'note': '', 'include': True}
    info.update(hdr)
    if root == EXVIVO_ROOT or (hdr.get('pixel_spacing_mm') or 1) <= 0.03 \
            or 'SCANCO' in hdr.get('manufacturer', '').upper():
        info['scan_type'] = 'ex vivo'
        info['subject'] = d.name
        info['treatment'] = 'unknown (ex vivo specimen)'
        pn = hdr.get('patient_name', '')
        m = re.search(r'(\d+)\s*month', pn, re.IGNORECASE)
        if m:
            info['timepoint'] = f'{m.group(1)} MONTH (ex vivo)'
            info['timepoint_months'] = int(m.group(1))
        info['note'] = f'PatientID {hdr.get("patient_id")}: {pn}'
        if 'Other Samples' in parts:
            info['treatment'] = 'unknown (Other Samples)'
            info['note'] += ' — not part of the PDLLA in vivo study'
        return info

    month_idx = next((i for i, p in enumerate(parts)
                      if re.fullmatch(r'\d+\s*MONTHS?', p.strip(), re.I)), None)
    if month_idx is not None:
        months = int(re.match(r'(\d+)', parts[month_idx].strip()).group(1))
        info['timepoint'] = parts[month_idx]
        info['timepoint_months'] = months
        info['treatment'] = parts[month_idx - 1]
        cand = parts[month_idx + 1] if month_idx + 1 < len(parts) else ''
        if re.fullmatch(r'\d+', cand):
            info['subject'] = cand
        elif re.fullmatch(r'\d+_cropped', cand):
            info['include'] = False
            info['subject'] = cand.split('_')[0]
            info['note'] = 'cropped duplicate export of an archived scan — excluded'
        else:
            info['subject'] = d.name
            info['treatment'] = 'unknown (unlabelled scan)'
            info['note'] = (f'anonymised export in "{cand}" — animal and treatment '
                            'not recorded in the DICOM header')
    if info['subject'] is None:
        info['subject'] = d.name
        info['treatment'] = info['treatment'] or 'unknown'
    # Duplicate exports: keep only the canonical dicom_t* directory when the
    # same subject/timepoint folder holds several full scans.
    return info


def dedupe_scans(scans):
    by_key = {}
    for s in scans:
        key = (s['scan_type'], s['treatment'], s['subject'], s['timepoint'], Path(s['scan_dir']).parent)
        by_key.setdefault(key, []).append(s)
    out = []
    for key, group in by_key.items():
        if len(group) == 1:
            out.append(group[0]); continue
        pref = [s for s in group if Path(s['scan_dir']).name.lower().startswith('dicom')]
        keep = (pref or group)[0]
        for s in group:
            if s is not keep:
                s['include'] = False
                s['note'] = f'duplicate export of {Path(keep["scan_dir"]).name} — excluded'
            out.append(s)
    # "3 months-selected 2" mirrors "3 months-selected" (same anonymised files)
    seen = {}
    for s in out:
        if s['treatment'].startswith('unknown (unlabelled'):
            k = (s['subject'], s['n_slices'])
            if k in seen and s['include']:
                s['include'] = False
                s['note'] = f'duplicate of {seen[k]} — excluded'
            else:
                seen.setdefault(k, s['scan_rel'])
    return out


def series_for_scan(scan: dict):
    """Union ROI series that belong to this raw scan."""
    d = Path(scan['scan_dir'])
    parent = d.parent
    out = []
    for e in sorted(os.scandir(parent), key=lambda e: e.name):
        if not e.is_dir() or e.name.startswith('.') or not is_union_series(e.name):
            continue
        if scan['scan_type'] == 'ex vivo':
            if not e.name.startswith(f'{scan["patient_id"]}_{d.name}_'):
                continue
        elif scan['treatment'].startswith('unknown (unlabelled'):
            if not e.name.startswith(d.name + '_'):
                continue
        n = len(dcm_names(Path(e.path)))
        out.append({'series_dir': e.path, 'series_name': e.name, 'series_files': n})
    return out


def placement_method(scan, name, job):
    if is_ground_truth(name):
        return 'manual annotation (ground truth)'
    if '_adj' in name or (job and job.get('manually_adjusted')):
        return 'manual adjustment of automatic placement'
    if 'exvivo' in name or scan['scan_type'] == 'ex vivo':
        return 'ex vivo geometric placement (no network)'
    if job:
        return {'3m': 'network (3-month, direct)',
                'later_reg': 'registration from 3-month ROI',
                'later_raw': 'network (raw, non-3-month — biased)',
                'manual': 'manual adjustment of automatic placement',
                'exvivo': 'ex vivo geometric placement (no network)'}.get(job['mode'], job['mode'])
    if scan['timepoint_months'] == 3:
        return 'network (3-month, direct)'
    return 'registration from 3-month ROI (2026-08-11 batch)'


def choose_primary(scan, series):
    """Index of the recommended series, or None (full-rebuild mode only —
    a web-app add always uses the run the user clicked)."""
    names = [s['series_name'] for s in series]
    subj = scan['subject']
    cands = [i for i, s in enumerate(series)
             if s['series_files'] > 0 and '_adj' not in s['series_name']
             and s['series_name'] not in SUPERSEDED]
    if not cands:
        # Nothing automatic on disk — fall back to a manual _adj series
        # rather than dropping the scan (the row is flagged manually_adjusted).
        cands = [i for i, s in enumerate(series)
                 if s['series_files'] > 0 and s['series_name'] not in SUPERSEDED]
    if not cands:
        return None
    for i in cands:                                  # ground truth first
        if is_ground_truth(names[i]):
            return i
    tp = scan['timepoint_months']
    prefer = []
    if scan['scan_type'] == 'ex vivo':
        prefer = [f'{scan["patient_id"]}_{Path(scan["scan_dir"]).name}_exvivo_output_dicom']
    elif tp == 3:
        prefer = [f'{subj}_predicted_output_dicom', f'{subj}_3m_pred_output_dicom']
    elif tp:
        prefer = [f'{subj}_{tp}m_output_dicom']
    for p in prefer:
        if p in names and names.index(p) in cands:
            return names.index(p)
    # else: the newest automatic series
    return max(cands, key=lambda i: os.stat(series[i]['series_dir']).st_mtime)


# ───────────────────────────────────────────────────────────── extraction

def features_path(series_dir: Path) -> Path:
    return series_dir.parent / (series_dir.name + '_features.json')


def newest_mtime(d: Path) -> float:
    fs = dcm_names(d)
    return max((os.stat(d / f).st_mtime for f in fs), default=0.0)


def ensure_features(scan, ser, jobs, do_extract: bool, log) -> str:
    sdir = Path(ser['series_dir'])
    fjson = features_path(sdir)
    job = jobs.get(str(sdir))
    if job and job.get('status') in ('queued', 'running'):
        return 'skipped: web-app job still running'
    if ser['series_files'] == 0:
        return 'skipped: empty series'
    if fjson.exists() and os.stat(fjson).st_mtime >= newest_mtime(sdir):
        try:
            n_keys = len(json.loads(fjson.read_text()).get('core', {}))
        except (OSError, json.JSONDecodeError):
            n_keys = 0
        if n_keys >= EXPECTED_FEATURES:
            return 'existing'
        print(f'  {sdir.name}: stale features file ({n_keys} features) — re-extracting', flush=True)
    if not do_extract:
        return 'missing (extraction disabled)'
    cmd = [PYTHON, str(REPO_DIR / '6_extract_features.py'),
           '--input', scan['scan_dir'], '--roi', str(sdir),
           '--bone-threshold', str(BONE_THRESHOLD_HU)]
    t0 = time.time()
    print(f'  extracting {sdir.name} ...', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write(f'\n### {sdir}\n{r.stdout}\n{r.stderr}\n')
    log.flush()
    if r.returncode != 0 or not fjson.exists():
        print(f'    FAILED ({r.returncode}) — see extraction log', flush=True)
        return f'failed: exit {r.returncode}'
    print(f'    done in {time.time() - t0:.0f} s', flush=True)
    return 'extracted now'


# ───────────────────────────────────────────────────────────── assembly

def series_row(scan, ser, jobs, qc, feat_status):
    sdir = Path(ser['series_dir'])
    job = jobs.get(str(sdir))
    row = {
        'treatment': scan['treatment'],
        'subject': scan['subject'],
        'timepoint': scan['timepoint'],
        'timepoint_months': scan['timepoint_months'],
        'scan_type': scan['scan_type'],
        'roi_series': ser['series_name'],
        'placement_method': placement_method(scan, ser['series_name'], job),
        'manually_adjusted': ('_adj' in ser['series_name']
                              or bool(job and job.get('manually_adjusted'))),
        'study_id': scan.get('study_id'),
        'patient_id': scan.get('patient_id'),
        'study_date': scan.get('study_date'),
        'scanner': scan.get('manufacturer'),
        'n_slices': scan.get('n_slices'),
        'pixel_spacing_mm': scan.get('pixel_spacing_mm'),
        'slice_spacing_mm': scan.get('slice_spacing_mm'),
        'scan_dir': scan['scan_rel'],
        'roi_series_dir': str(sdir.relative_to(DATA_ROOT.parent)),
        'roi_series_date': datetime.fromtimestamp(newest_mtime(sdir)).strftime('%Y-%m-%d') if ser['series_files'] else None,
        'feature_status': feat_status,
        'visual_qc_note': VISUAL_QC_NOTES.get(ser['series_name'], ''),
        'note': scan.get('note') or '',
        'added_via': 'full rebuild',
    }
    # QC from the web-app run
    if job:
        checks = job.get('checks') or []
        rank = {'pass': 0, 'warn': 1, 'fail': 2}
        worst = max((rank.get(c.get('status'), 0) for c in checks), default=None)
        row['webapp_job_id'] = job['id']
        row['webapp_status'] = job.get('status')
        row['qc_overall'] = {0: 'pass', 1: 'warn', 2: 'fail', None: None}[worst]
        row['qc_flags'] = '; '.join(f'{c["name"]}: {c["status"]}' for c in checks if c.get('status') != 'pass') or ''
        m = job.get('metrics') or {}
        row['registration_dice'] = m.get('dice')
        row['webapp_tilt_deg'] = m.get('tilt_deg')
        res = job.get('results') or {}
        row['webapp_core_to_ring_bvtv'] = res.get('core_to_ring')
    else:
        row.update({'webapp_job_id': None, 'webapp_status': None, 'qc_overall': None,
                    'qc_flags': '', 'registration_dice': None, 'webapp_tilt_deg': None,
                    'webapp_core_to_ring_bvtv': None})
    q = qc.get((scan['subject'], scan['timepoint_months'])) if scan['scan_type'] == 'in vivo' else None
    base_batch = (scan['timepoint_months'] != 3 and ser['series_name'] == f'{scan["subject"]}_{scan["timepoint_months"]}m_output_dicom') \
        or (scan['timepoint_months'] == 3 and ser['series_name'] == f'{scan["subject"]}_predicted_output_dicom')
    if q and base_batch:
        row['qc_batch_2026_08_11_mode'] = q['mode']
        row['qc_batch_dice'] = float(q['dice']) if q['dice'] else None
        row['qc_batch_network_vs_registered_shift_mm'] = float(q['shift_mm']) if q['shift_mm'] else None
        row['qc_batch_note'] = q['void']
        if row['qc_overall'] is None:
            row['qc_overall'] = 'pass (batch QC)' if q['mode'].startswith('propagated') or q['mode'] == 'infer-only' else q['mode']
    elif is_ground_truth(ser['series_name']):
        row['qc_batch_2026_08_11_mode'] = 'ground truth'
        row['qc_batch_dice'] = None
        row['qc_batch_network_vs_registered_shift_mm'] = None
        row['qc_batch_note'] = 'hand-labelled annotation'
        row['qc_overall'] = row['qc_overall'] or 'ground truth'
    else:
        row['qc_batch_2026_08_11_mode'] = None
        row['qc_batch_dice'] = None
        row['qc_batch_network_vs_registered_shift_mm'] = None
        row['qc_batch_note'] = ''
    # features
    fjson = features_path(sdir)
    if fjson.exists():
        d = json.loads(fjson.read_text())
        meta = d.get('meta', {})
        row['features_extracted_utc'] = meta.get('extracted_utc')
        row['feature_voxel_mm'] = meta.get('voxel_mm', [None])[0]
        row['bone_threshold_hu'] = meta.get('bone_threshold_hu')
        c = meta.get('center_mm'); a = meta.get('axis')
        row['roi_center_z_mm'], row['roi_center_row_mm'], row['roi_center_col_mm'] = (c if c else (None, None, None))
        row['roi_axis_z'], row['roi_axis_row'], row['roi_axis_col'] = (a if a else (None, None, None))
        row['roi_tilt_deg'] = tilt_from_axis(a) if a else None
        core, ring = d.get('core', {}), d.get('ring', {})
        for k in core:
            row[f'core_{k}'] = core.get(k)
        for k in ring:
            row[f'ring_{k}'] = ring.get(k)
        for k in core:
            cv, rv = core.get(k), ring.get(k)
            try:
                row[f'core_to_ring_{k}'] = float(cv) / float(rv) if (cv is not None and rv not in (None, 0) and np.isfinite(cv) and np.isfinite(rv) and rv != 0) else None
            except (TypeError, ValueError):
                row[f'core_to_ring_{k}'] = None
    return row


def feature_dictionary(feature_names):
    known = {k: (fam, desc, unit) for k, fam, desc, unit in FEATURE_FAMILIES}
    rows = []
    for k in feature_names:
        if k in known:
            fam, desc, unit = known[k]
        else:
            fam, desc, unit = 'other', '', '-'
            for pre, (f, base) in FAMILY_PREFIX.items():
                if k.startswith(pre):
                    fam = f
                    desc = BONE_MORPH_DESC.get(k, f'{k[len(pre):].replace("_", " ")} — {base}')
                    unit = 'mm^3' if k.endswith('_mm3') else 'mm^2' if k.endswith('_mm2') else 'mm' if k.endswith('_mm') else 'HU' if k.endswith('_hu') else 'bits' if k.endswith('_bits') else 'count' if k == 'bone_components' else '-'
        rows.append({'feature': k, 'family': fam, 'description': desc, 'units': unit,
                     'columns': f'core_{k}, ring_{k}, core_to_ring_{k}'})
    return pd.DataFrame(rows)


README_TEXT = """Rabbit calvarial defect study — radiomics database
Built by defect_segmentation/7_build_database.py on {built}

WHAT IS IN HERE
  database           ONE row per scan. Each row is normally the scan's latest successful
                     web-app run: the app adds every run AUTOMATICALLY when it finishes
                     without a FAILed check (re-running or manually readjusting a scan
                     replaces its row — rows are keyed by the scanner StudyID, so any
                     export folder of the same scan maps to the same row). Runs that
                     error or fail a check are never added.
                     In a full rebuild the row is the best series on disk (ground truth
                     first, then the standard automatic series; a manual _adj nudge only
                     when nothing else exists). Scans with no usable series appear as stub
                     rows with empty features, so nothing silently disappears.
  feature_dictionary Family, description and units for each of the 113 features.
  (radiomics_database.csv next to this file mirrors the `database` sheet.)

REGIONS
  core  = 10 mm diameter x 8 mm tall cylinder over the defect ("in the defect")
  ring  = 14 mm ID / 18 mm OD x 8 mm annulus of reference bone ("around the defect")
  (a 5-7 mm radius gap between them is excluded). The ROI is one rigid oblique cylinder
  template; only its centre and axis vary per scan (roi_center_*, roi_axis_*, roi_tilt_deg).
  Columns: core_<feature>, ring_<feature>, core_to_ring_<feature> = core / ring.

KEY COLUMNS
  treatment          Study group folder: Defect, Defect +PDLLA, MC, MC+PDLLA (unknown for
                     unlabelled / ex vivo / Other Samples scans).
  subject            Animal ID (in vivo) or specimen scan number (ex vivo).
  timepoint_months   3, 6, 9 (in vivo). Ex vivo rows carry the month from the specimen label.
  roi_series         The series this row's numbers come from.
  placement_method   How the ROI was placed. NEVER mix methods in one comparison.
  manually_adjusted  TRUE when the row comes from a hand-nudged _adj series — never mix
                     TRUE and FALSE rows in one comparison.
  visual_qc_note     Human review of the axial preview where one was done (2026-09-07).
  qc_overall / qc_flags   From the web-app sanity checks (pass / warn / fail) or the 2026-08-11
                     batch QC. Treat 'fail' rows as unusable; read 'warn' details before use.
  registration_dice  Registration overlap for 6/9-month placements (gate is 0.5).
  bone_threshold_hu  226 HU fixed study threshold behind bvtv_fixed and bone_* features.
  feature_voxel_mm   Working voxel size of the extraction (0.1 mm in vivo, 0.06 mm ex vivo).
  other_series_on_disk  Alternative ROI series that exist for the scan but are not this row.
  added_via          'full rebuild' or the web-app run that added / refreshed the row
                     automatically when it finished.

RULES (from defect_segmentation/README.md)
  * Report core_to_ring_bvtv_fixed, not absolute BV/TV — the 8 mm template is taller than the
    calvarial plate, so absolute values are geometrically diluted.
  * Never compare ex vivo (15 µm SCANCO) rows with in vivo (100 µm) rows, even for the same
    feature and threshold. scan_type separates them.
  * Never mix ground-truth, network, registration and manually adjusted ROIs in one comparison.
  * Compare texture features only between rows with the same feature_voxel_mm.
  * Per-region Otsu thresholds are diagnostics, not bone thresholds (see feature_dictionary).
  * bvtv_fixed is recomputed by the feature extractor on the template re-fitted from the written
    series; it agrees with the 2026-08-11 cohort table to within ~0.5 pp for most scans (max 1.4 pp,
    37950 at 3 months). Use the values here consistently rather than mixing the two sources.

SCANS WITHOUT FEATURES
  Stub rows with an empty roi_series and a feature_status note. Known: 41122 at 9 months —
  registration failed its dice gate four times (0.2 mm re-export, large head-pose change,
  healed defect); needs a re-export at 0.1 mm or manual template placement.

UPDATE
  Normal path: run a scan in the web app — when the run finishes without a FAILed check
  it is added to this spreadsheet automatically (the run page shows "in spreadsheet").
  Re-running a scan, or nudging its placement, updates that scan's row in place.
  Full rebuild from disk: python defect_segmentation/7_build_database.py
  (re-extracts features only for series that have none; add --no-extract to just re-assemble)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.environ.get(
        'DEFECT_DB_DIR', str(DATA_ROOT / 'radiomics_database')),
        help='database directory (env DEFECT_DB_DIR overrides the default)')
    ap.add_argument('--no-extract', action='store_true', help='assemble only; do not run 6_extract_features.py')
    ap.add_argument('--add-job', metavar='JOB_ID', action='append',
                    help='incremental: add/refresh one finished web-app run (repeatable)')
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.add_job:
        for jid in args.add_job:
            add_job(jid, out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs()
    qc = load_qc()
    print('Scanning archive ...', flush=True)
    scans = dedupe_scans(find_scans())
    scans.sort(key=lambda s: (s['scan_type'], str(s['treatment']), str(s['subject']), s['timepoint_months'] or 0))
    print(f'  {len(scans)} raw scans ({sum(s["include"] for s in scans)} included)')

    rows = []
    with open(out_dir / 'extraction_log.txt', 'a') as log:
        log.write(f'\n===== build {datetime.now().isoformat(timespec="seconds")} =====\n')
        for scan in scans:
            if not scan['include']:
                continue                    # duplicate exports etc. — skipped
            series = series_for_scan(scan)
            prim = choose_primary(scan, series) if series else None
            print(f'{scan["scan_type"]:7s} {str(scan["treatment"]):28s} {scan["subject"]:>8s} {str(scan["timepoint"]):22s} {len(series)} series', flush=True)
            others = [s['series_name'] for i, s in enumerate(series) if i != prim]
            if prim is None:
                # Stub row so scans without a usable series stay visible.
                r = {'treatment': scan['treatment'], 'subject': scan['subject'],
                     'timepoint': scan['timepoint'], 'timepoint_months': scan['timepoint_months'],
                     'scan_type': scan['scan_type'], 'roi_series': None,
                     'placement_method': None, 'manually_adjusted': False,
                     'study_id': scan.get('study_id'), 'patient_id': scan.get('patient_id'),
                     'study_date': scan.get('study_date'), 'scanner': scan.get('manufacturer'),
                     'n_slices': scan.get('n_slices'),
                     'pixel_spacing_mm': scan.get('pixel_spacing_mm'),
                     'slice_spacing_mm': scan.get('slice_spacing_mm'),
                     'scan_dir': scan['scan_rel'], 'roi_series_dir': None,
                     'feature_status': 'no ROI series' if not series else 'no usable series',
                     'note': scan['note'] or '', 'added_via': 'full rebuild',
                     'other_series_on_disk': '; '.join(others)}
                rows.append(r)
                continue
            ser = series[prim]
            status = ensure_features(scan, ser, jobs, not args.no_extract, log)
            r = series_row(scan, ser, jobs, qc, status)
            r['other_series_on_disk'] = '; '.join(others)
            rows.append(r)

    for d, n in LOOSE:
        print(f'  note: {n} loose .dcm files in "{d}" — not a scan, ignored')
    if not rows:
        print('No scans found — nothing to write.'); return
    write_outputs(pd.DataFrame(rows), out_dir)


def order_columns(df):
    meta_cols = [c for c in df.columns if not c.startswith(('core_', 'ring_'))]
    feat_names = [c[5:] for c in df.columns if c.startswith('core_') and not c.startswith('core_to_ring_')]
    ordered = meta_cols + [f'core_{k}' for k in feat_names] + [f'ring_{k}' for k in feat_names] \
        + [f'core_to_ring_{k}' for k in feat_names]
    return df[[c for c in ordered if c in df.columns]], feat_names


def write_outputs(df, out_dir):
    """Write the CSV and the single-sheet workbook (one row per scan)."""
    df, feat_names = order_columns(df)
    df = df.sort_values(['scan_type', 'treatment', 'subject', 'timepoint_months'],
                        na_position='last').reset_index(drop=True)
    dict_df = feature_dictionary(feat_names)

    df.to_csv(out_dir / 'radiomics_database.csv', index=False)
    # Retire the old four-table layout so nobody reads stale copies.
    for legacy in ('radiomics_all_series.csv', 'radiomics_primary.csv',
                   'radiomics_long.csv', 'scan_inventory.csv'):
        f = out_dir / legacy
        if f.exists():
            f.unlink()

    # Workbook
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active; ws.title = 'README'
    ws.column_dimensions['A'].width = 120
    for i, line in enumerate(README_TEXT.format(built=datetime.now().strftime('%Y-%m-%d %H:%M')).splitlines(), 1):
        c = ws.cell(row=i, column=1, value=line)
        c.font = Font(name='Arial', size=10, bold=(i == 1 or (line and not line.startswith(' ') and line.isupper())))
        c.alignment = Alignment(wrap_text=False)

    def write_df(name, frame, freeze_col=8, widths=None):
        sh = wb.create_sheet(name)
        head_font = Font(name='Arial', bold=True, size=10)
        body_font = Font(name='Arial', size=10)
        fill = PatternFill('solid', fgColor='DDEBF7')
        cols = list(frame.columns)
        for j, col in enumerate(cols, 1):
            c = sh.cell(row=1, column=j, value=col); c.font = head_font; c.fill = fill
            if col.startswith('core_to_ring_'):
                c.fill = PatternFill('solid', fgColor='FCE4D6')
            elif col.startswith('ring_'):
                c.fill = PatternFill('solid', fgColor='E2EFDA')
        for i, rec in enumerate(frame.itertuples(index=False), 2):
            for j, v in enumerate(rec, 1):
                if isinstance(v, float) and np.isnan(v):
                    v = None
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
                c = sh.cell(row=i, column=j, value=v); c.font = body_font
                if isinstance(v, float):
                    c.number_format = '0.0000' if abs(v) < 100 else '0.00'
        for j, col in enumerate(cols, 1):
            w = (widths or {}).get(col, max(10, min(28, len(col) + 2)))
            sh.column_dimensions[get_column_letter(j)].width = w
        sh.freeze_panes = sh.cell(row=2, column=freeze_col + 1)
        sh.auto_filter.ref = f'A1:{get_column_letter(len(cols))}{len(frame) + 1}'
        return sh

    write_df('database', df, widths={'placement_method': 34, 'qc_flags': 40, 'roi_series': 34, 'scan_dir': 40, 'roi_series_dir': 44, 'qc_batch_note': 40, 'visual_qc_note': 50, 'note': 50, 'other_series_on_disk': 44, 'added_via': 40})
    write_df('feature_dictionary', dict_df, freeze_col=0, widths={'description': 90, 'family': 18, 'columns': 60, 'feature': 36})
    xlsx = out_dir / 'radiomics_database.xlsx'
    wb.save(xlsx)

    print(f'\nWrote {xlsx}')
    print(f'  database : {len(df)} scans, {len(feat_names)} features x (core, ring, core/ring)')
    missing = df[df['feature_status'].astype(str).str.startswith(('failed', 'missing', 'skipped', 'no '))]
    if len(missing):
        print('  scans without features:')
        for _, r in missing.iterrows():
            print(f'    {str(r["roi_series"] or r["scan_dir"]):40s} {r["feature_status"]}')


# ───────────────────────────────────────────── incremental add (web app)

STR_COLS = ['subject', 'study_id', 'patient_id', 'study_date', 'timepoint', 'treatment',
            'roi_series', 'roi_series_dir', 'scan_dir', 'webapp_job_id']


def load_existing(out_dir: Path):
    f = out_dir / 'radiomics_database.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={c: str for c in STR_COLS}, keep_default_na=True)
    if 'manually_adjusted' in df.columns:
        df['manually_adjusted'] = \
            df['manually_adjusted'].astype(str).str.lower().eq('true')
    return df


def scan_for_job(job: dict) -> dict:
    """Describe the scan behind a web-app run, including uploaded scans that
    live outside the archive layout. For uploads the user-entered metadata
    form (job['user_meta']) is the identity of record — it is preferred over
    the 'unknown (uploaded scan)' placeholders."""
    input_dir = Path(job['params']['input'])
    root = EXVIVO_ROOT if EXVIVO_ROOT.is_dir() and EXVIVO_ROOT in input_dir.parents else DATA_ROOT
    scan = classify_scan(input_dir, root)
    ctx = job.get('context') or {}
    um = job.get('user_meta') or {}
    in_uploads = 'uploads' in input_dir.parts
    if in_uploads:
        scan['treatment'] = (um.get('treatment') or ctx.get('group')
                             or 'unknown (uploaded scan)')
        scan['subject'] = str(um.get('sample') or ctx.get('subject')
                              or job.get('label') or input_dir.parent.name)
        note = f'uploaded through the web app (run {job["id"]}); not in the study archive'
        if um.get('notes'):
            note += f' — user notes: {um["notes"]}'
        scan['note'] = note + (f' — {scan["note"]}' if scan['note'] else '')
        tp = um.get('timepoint') or ctx.get('timepoint')
        if tp and not scan['timepoint']:
            scan['timepoint'] = tp
            m = re.match(r'(\d+)', str(tp))
            scan['timepoint_months'] = int(m.group(1)) if m else None
        elif um.get('timepoint_months') and not scan['timepoint_months']:
            scan['timepoint_months'] = um['timepoint_months']
    else:
        if ctx.get('group') and scan['treatment'] in (None, 'unknown'):
            scan['treatment'] = ctx['group']
        if ctx.get('subject') and (scan['subject'] in (None, input_dir.name)):
            scan['subject'] = ctx['subject']
    scan['include'] = True
    return scan


def add_job(job_id: str, out_dir: Path) -> dict:
    """Upsert one finished web-app run into the database."""
    try:
        jobs_raw = json.loads(JOBS_FILE.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f'cannot read run history: {e}')
    job = jobs_raw.get(job_id)
    if not job:
        raise SystemExit(f'no such run: {job_id}')
    if job.get('status') not in ('done', 'done_warn'):
        raise SystemExit(f'run {job_id} is {job.get("status")} — only runs that '
                         'finished without a FAILed check can be added')
    jobs = load_jobs()
    qc = load_qc()
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_for_job(job)
    sdir = Path(job['params']['output'])
    if not sdir.is_dir():
        raise SystemExit(f'series directory missing: {sdir}')
    ser = {'series_dir': str(sdir), 'series_name': sdir.name, 'series_files': len(dcm_names(sdir))}
    if ser['series_files'] == 0:
        raise SystemExit(f'series {sdir.name} contains no .dcm files')

    with open(out_dir / 'extraction_log.txt', 'a') as log:
        log.write(f'\n===== add run {job_id} {datetime.now().isoformat(timespec="seconds")} =====\n')
        status = ensure_features(scan, ser, jobs, True, log)
    if status.startswith(('failed', 'skipped')):
        raise SystemExit(f'feature extraction: {status}')

    row = series_row(scan, ser, jobs, qc, status)
    row['other_series_on_disk'] = ''
    row['added_via'] = f'web app run {job_id} added {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    if 'core_voxels' not in row:
        raise SystemExit('no features found for this series after extraction')

    # One row per SCAN: replace any existing row for the same scan. The key
    # is the scanner StudyID when the header carries one — so the same scan
    # added from a different export folder still replaces its row — with the
    # scan folder as the fallback for anonymised headers.
    df = load_existing(out_dir)
    replaced = 0
    if df is not None and not df.empty:
        sid = str(row.get('study_id') or '').strip()
        same_scan = df['scan_dir'].astype(str) == str(row['scan_dir'])
        if sid:
            same_scan |= df['study_id'].fillna('').astype(str).str.strip() == sid
        # A series dir can only ever describe one scan.
        same_scan |= df['roi_series_dir'].fillna('').astype(str) == str(row['roi_series_dir'])
        replaced = int(same_scan.sum())
        df = df[~same_scan]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False)
    else:
        df = pd.DataFrame([row])

    write_outputs(df, out_dir)
    result = {'ok': True, 'job': job_id, 'roi_series': ser['series_name'],
              'replaced': bool(replaced),
              'manually_adjusted': bool(row['manually_adjusted']),
              'feature_status': status, 'n_scans': int(len(df)),
              'xlsx': str(out_dir / 'radiomics_database.xlsx'),
              'treatment': scan['treatment'], 'subject': scan['subject'],
              'timepoint_months': scan['timepoint_months']}
    print('RESULT ' + json.dumps(result))
    return result


if __name__ == '__main__':
    main()
