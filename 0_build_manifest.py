"""
0_build_manifest.py
Builds data/subjects.csv mapping each of the 12 labeled 3-month subjects to
their original DICOM directory and ground-truth _output_dicom directory.
Verifies file counts and flags any issues.
"""

import argparse
import csv
import sys
from pathlib import Path

# Root holding the per-group subject folders. Override with --base on another machine.
BASE = Path('/Volumes/justinytlin/Craniofacial/Radiomics/In Vivo CT Data/PDLLA RAW DICOM VIVO DATA')

SUBJECTS = [
    ('37951', 'Defect',       'Defect/3 MONTH/37951/dicom_t57860',
                               'Defect/3 MONTH/37951/37951_output_dicom'),
    ('37952', 'Defect',       'Defect/3 MONTH/37952/dicom_t57863_1',
                               'Defect/3 MONTH/37952/37952_output_dicom'),
    ('38743', 'Defect+PDLLA', 'Defect +PDLLA/3 MONTH/38743/dicom_t57862',
                               'Defect +PDLLA/3 MONTH/38743/38743_output_dicom'),
    ('38744', 'Defect+PDLLA', 'Defect +PDLLA/3 MONTH/38744/dicom_t57864',
                               'Defect +PDLLA/3 MONTH/38744/38744_output_dicom'),
    ('38750', 'Defect+PDLLA', 'Defect +PDLLA/3 MONTH/38750/dicom_t57939',
                               'Defect +PDLLA/3 MONTH/38750/38750_output_dicom'),
    ('44266', 'Defect+PDLLA', 'Defect +PDLLA/3 MONTH/44266/dicom_t59116',
                               'Defect +PDLLA/3 MONTH/44266/44266_output_dicom'),
    ('38747', 'MC',           'MC/3 MONTH/38747/dicom_t57936',
                               'MC/3 MONTH/38747/38747_output_dicom'),
    ('38748', 'MC',           'MC/3 MONTH/38748/dicom_t57937',
                               'MC/3 MONTH/38748/38748_output_dicom'),
    ('38749', 'MC',           'MC/3 MONTH/38749/dicom_t57938',
                               'MC/3 MONTH/38749/38749_output_dicom'),
    ('37949', 'MC+PDLLA',     'MC+PDLLA/3 MONTH/37949/dicom_t57854',
                               'MC+PDLLA/3 MONTH/37949/37949_output_dicom'),
    ('38746', 'MC+PDLLA',     'MC+PDLLA/3 MONTH/38746/dicom_t57861',
                               'MC+PDLLA/3 MONTH/38746/38746_output_dicom'),
    ('44267', 'MC+PDLLA',     'MC+PDLLA/3 MONTH/44267/dicom_T59117',
                               'MC+PDLLA/3 MONTH/44267/44267_output_dicom'),
]

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)
OUT_CSV    = DATA_DIR / 'subjects.csv'


def count_dcm(directory: Path) -> int:
    if not directory.exists():
        return -1
    return sum(1 for f in directory.iterdir()
               if f.suffix.lower() == '.dcm' and not f.name.startswith('._'))


def main():
    parser = argparse.ArgumentParser(
        description='Build data/subjects.csv from the raw DICOM tree')
    parser.add_argument('--base', default=str(BASE),
                        help='Root holding the Defect/, MC/, ... group folders '
                             '(default: the path this study was authored against)')
    args = parser.parse_args()
    base = Path(args.base)

    if not base.exists():
        print(f'ERROR: base directory not found: {base}')
        print('Pass --base /path/to/PDLLA RAW DICOM VIVO DATA')
        sys.exit(1)

    rows = []
    errors = []

    for sid, group, orig_rel, out_rel in SUBJECTS:
        orig_dir = base / orig_rel
        out_dir  = base / out_rel

        n_orig = count_dcm(orig_dir)
        n_out  = count_dcm(out_dir)

        ok_orig = n_orig == 1200
        ok_out  = n_out  == 1200

        status = 'OK'
        if not ok_orig:
            msg = f'{sid}: orig DICOM count={n_orig} (expected 1200) at {orig_dir}'
            errors.append(msg)
            status = 'ERROR'
        if not ok_out:
            msg = f'{sid}: output DICOM count={n_out} (expected 1200) at {out_dir}'
            errors.append(msg)
            status = 'ERROR'

        rows.append({
            'subject_id':       sid,
            'group':            group,
            'orig_dicom_dir':   str(orig_dir),
            'output_dicom_dir': str(out_dir),
            'n_orig_files':     n_orig,
            'n_output_files':   n_out,
            'status':           status,
        })

        symbol = '✓' if status == 'OK' else '✗'
        print(f'  {symbol} {sid:5s}  [{group:12s}]  orig={n_orig:4d}  output={n_out:4d}  {status}')

    fieldnames = ['subject_id', 'group', 'orig_dicom_dir', 'output_dicom_dir',
                  'n_orig_files', 'n_output_files', 'status']
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nManifest written → {OUT_CSV}')
    print(f'Subjects: {len(rows)} total, {sum(1 for r in rows if r["status"]=="OK")} OK')

    if errors:
        print('\nISSUES:')
        for e in errors:
            print(f'  ! {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
