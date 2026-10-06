#!/usr/bin/env python3
"""Re-place 6-month in vivo ROIs by registration from each animal's 3-month ROI.

The MR525xx 6-month ROIs in the database were placed by raw network inference,
which is biased at non-3-month timepoints (see 4_propagate_roi.py). This runs
the web app's validated 'later_reg' path for each animal:

  3_inference.py --fit-only (network hint)  ->  4_propagate_roi.py (dice gate 0.5)
  on a dice-gate failure: one retry from the reference-pose hint + --wide-search

into a NEW series <animal>_6m_reg_output_dicom next to the old network series
(which is kept), then extracts features (6_) and the density sidecar (8_).

    python 10_replace_6m_by_registration.py            # place + extract
    python 10_replace_6m_by_registration.py --update-db  # then swap the DB rows

--update-db backs up the database (write_outputs does), keeps workbook hand
edits, and replaces ONLY the registered animals' 6-month rows: ROI series,
placement method, registration dice, QC, and all feature columns.
"""

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
PY = sys.executable
ANIMALS = ['MR52521', 'MR52522', 'MR52523', 'MR52524', 'MR52525', 'MR52526',
           'MR52527', 'MR52528', 'MR52529', 'MR52530']


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, REPO / fname)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def rescue_snippet():
    tree = ast.parse((REPO / 'webapp.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'RESCUE_HINT_SNIPPET':
            return ast.literal_eval(node.value)
    raise RuntimeError('RESCUE_HINT_SNIPPET not found in webapp.py')


def run(cmd, log):
    log.write('\n$ ' + ' '.join(map(str, cmd)) + '\n'); log.flush()
    r = subprocess.run(list(map(str, cmd)), cwd=REPO, capture_output=True, text=True)
    log.write(r.stdout + r.stderr); log.flush()
    return r


def place(animal, df, base, log):
    rows = df[(df.subject == animal) & (df.scan_type == 'in vivo')]
    r3 = rows[rows.timepoint_months == 3].iloc[0]
    r6 = rows[rows.timepoint_months == 6].iloc[0]
    ref_in, ref_roi = base / r3.scan_dir, base / r3.roi_series_dir
    tgt_in = base / r6.scan_dir
    out = (base / r6.roi_series_dir).parent / f'{animal}_6m_reg_output_dicom'
    fit = out.parent / f'{animal}_6m_fit.json'
    res = {'animal': animal, 'output': out, 'rescue': False}
    if out.exists() and (out.parent / (out.name + '_features.json')).exists():
        res.update(status='existing', dice=json.loads(
            (out.parent / (out.name + '_registration.json')).read_text())['dice'])
        return res
    print(f'{animal}: network hint...', flush=True)
    r = run([PY, '3_inference.py', '--input', tgt_in, '--output', out,
             '--fit-only', fit, '--fast-fit'], log)
    if r.returncode:
        return {**res, 'status': f'hint failed ({r.returncode})'}
    base_cmd = [PY, '4_propagate_roi.py', '--ref-input', ref_in, '--ref-roi', ref_roi,
                '--target-input', tgt_in, '--output', out, '--bone-refine']
    print(f'{animal}: registering 3 m -> 6 m...', flush=True)
    r = run(base_cmd + ['--target-fit', fit], log)
    if r.returncode and 'dice' in (r.stdout + r.stderr).lower():
        print(f'{animal}: dice gate failed — retrying from the reference pose', flush=True)
        synth = out.parent / f'{animal}_6m_rescue_hint.json'
        r2 = run([PY, '-c', rescue_snippet(), ref_in, ref_roi, synth], log)
        if r2.returncode == 0:
            r = run(base_cmd + ['--target-fit', synth, '--wide-search'], log)
            res['rescue'] = r.returncode == 0
    m = re.findall(r'bone dice = ([0-9.]+)', r.stdout)
    res['dice'] = float(m[-1]) if m else None
    sh = re.findall(r'network was ([0-9.]+) mm away', r.stdout)
    res['shift_mm'] = float(sh[-1]) if sh else None
    ang = re.findall(r'\(([0-9.]+) deg from network axis\)', r.stdout)
    res['axis_change_deg'] = float(ang[-1]) if ang else None
    if r.returncode:
        return {**res, 'status': f'registration refused (dice {res["dice"]})'}
    (out.parent / (out.name + '_registration.json')).write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in res.items()}
        | {'ref_roi': str(ref_roi), 'utc': datetime.utcnow().isoformat(timespec='seconds')}, indent=1))
    print(f'{animal}: dice {res["dice"]:.3f}, network was {res["shift_mm"]} mm off; extracting features...', flush=True)
    r = run([PY, '6_extract_features.py', '--input', tgt_in, '--roi', out, '--bone-threshold', '226'], log)
    if r.returncode:
        return {**res, 'status': 'feature extraction failed'}
    run([PY, '8_density_particles.py', '--input', tgt_in, '--roi', out], log)
    return {**res, 'status': 'registered'}


def update_db(results, db, dp, base):
    out_dir = db.DATA_ROOT / 'radiomics_database'
    df = db.load_existing(out_dir)
    df = dp.carry_over_xlsx_edits(df, out_dir, db.STR_COLS)
    n = 0
    for res in results:
        if res['status'] not in ('registered', 'existing'):
            continue
        sdir = Path(res['output'])
        m = (df.subject.astype(str) == res['animal']) & (df.scan_type == 'in vivo') & (df.timepoint_months == 6)
        assert m.sum() == 1, res['animal']
        i = df.index[m][0]
        old = df.loc[i, 'roi_series']
        scan = {'scan_type': 'in vivo', 'timepoint_months': 6, 'treatment': df.loc[i, 'treatment'],
                'subject': res['animal'], 'scan_dir': str(base / df.loc[i, 'scan_dir']),
                'scan_rel': df.loc[i, 'scan_dir']}
        for k in ('timepoint', 'study_id', 'patient_id', 'study_date', 'manufacturer',
                  'n_slices', 'pixel_spacing_mm', 'slice_spacing_mm', 'note'):
            scan[k] = df.loc[i, k] if k in df.columns else None
        scan['manufacturer'] = df.loc[i, 'scanner']
        ser = {'series_dir': str(sdir), 'series_name': sdir.name,
               'series_files': len(db.dcm_names(sdir))}
        row = db.series_row(scan, ser, {}, {}, 'extracted now')
        feat = {k: v for k, v in row.items() if k.startswith(('core_', 'ring_'))
                or k in ('features_extracted_utc', 'feature_voxel_mm', 'bone_threshold_hu',
                         'roi_center_z_mm', 'roi_center_row_mm', 'roi_center_col_mm',
                         'roi_axis_z', 'roi_axis_row', 'roi_axis_col', 'roi_tilt_deg',
                         'roi_series', 'roi_series_dir', 'roi_series_date')}
        for k, v in feat.items():
            if k not in df.columns:
                df[k] = None
            df[k] = df[k].astype(object)
            df.at[i, k] = v
        df.at[i, 'placement_method'] = 'registration from 3-month ROI'
        df.at[i, 'manually_adjusted'] = False
        df.at[i, 'registration_dice'] = res['dice']
        df.at[i, 'qc_overall'] = 'pass (registration dice gate)'
        df.at[i, 'qc_flags'] = ('registration rescue: reference-pose hint + wide search'
                                if res.get('rescue') else '')
        df.at[i, 'webapp_job_id'] = None
        df.at[i, 'added_via'] = (f'10_replace_6m_by_registration.py {datetime.now():%Y-%m-%d %H:%M} '
                                 f'(replaced raw-network series {old})')
        oth = str(df.at[i, 'other_series_on_disk'] or '').replace('nan', '')
        df.at[i, 'other_series_on_disk'] = '; '.join(x for x in (oth, old) if x)
        n += 1
    print(f'updated {n} rows')
    db.write_outputs(df, out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--update-db', action='store_true')
    ap.add_argument('--only', action='append')
    args = ap.parse_args()
    db = _load('_db', '7_build_database.py')
    dp = _load('_dp', '8_density_particles.py')
    base = db.DATA_ROOT.parent
    df = pd.read_csv(db.DATA_ROOT / 'radiomics_database' / 'radiomics_database.csv', dtype={'subject': str})
    logf = REPO / 'logs' / f'replace_6m_{datetime.now():%Y%m%d_%H%M}.log'
    logf.parent.mkdir(exist_ok=True)
    results = []
    with open(logf, 'w') as log:
        for a in args.only or ANIMALS:
            try:
                results.append(place(a, df, base, log))
            except Exception as e:                      # noqa: BLE001
                results.append({'animal': a, 'status': f'error: {e}'})
            print(f'  -> {results[-1]["status"]}', flush=True)
    summ = pd.DataFrame(results)
    print(summ.drop(columns=['output'], errors='ignore').to_string(index=False))
    summ.to_csv(logf.with_suffix('.csv'), index=False)
    print(f'log: {logf}')
    if args.update_db:
        update_db(results, db, dp, base)


if __name__ == '__main__':
    main()
