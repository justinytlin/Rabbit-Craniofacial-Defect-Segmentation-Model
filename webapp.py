#!/usr/bin/env python3
"""Lab web app for the rabbit calvarial defect segmentation pipeline.

Wraps 3_inference.py and 4_propagate_roi.py behind a local browser UI so lab
members can segment a scan without touching the terminal:

    python3 webapp.py            # then open http://127.0.0.1:8765
    python3 webapp.py --open     # opens the browser for you

The app enforces the study's placement rules:
  * 3-month scans        -> direct network placement (in-distribution)
  * any other timepoint  -> registration from the animal's 3-month ROI
                            (4_propagate_roi.py); raw network placement is
                            allowed only with an explicit warning, because the
                            network has a measured 2-3 mm bias there.

It also runs the README's sanity checklist on every run (tilt, eigenvalues,
ROI enclosure, registration dice) and computes ROI volumes / BV/TV from the
written series, so results are read off the screen instead of the notebook.

Standard library only — no dependencies beyond requirements.txt.
"""

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DATA_ROOT = REPO_DIR.parent
APP_DIR = REPO_DIR / 'logs' / 'webapp'
JOBS_FILE = APP_DIR / 'jobs.json'
HTML_FILE = REPO_DIR / 'webapp.html'
PYTHON = sys.executable

DEFAULT_BONE_THRESHOLD = 226.0

# Ranges from README.md: all 12 ground-truth fits land inside these.
TILT_RANGE = (74.0, 88.0)
EIG1_RANGE = (3.5, 7.5)        # GT 5.33 mm², predictions ~5.47
EIG23_RANGE = (14.0, 30.0)     # GT 20.98 / 20.99 mm²
SERIES_SUFFIXES = ['', '_cylinder', '_ring', '_cylinder_bone', '_ring_bone']

ALLOWED_ROOTS = [DATA_ROOT]

JOBS = {}          # id -> job dict
JOB_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()
CURRENT_PROC = {}  # job_id -> subprocess.Popen


# ─────────────────────────────────────────────────────────────── path helpers

def path_allowed(p: Path) -> bool:
    try:
        rp = p.resolve()
    except OSError:
        return False
    return any(rp == root or root in rp.parents for root in ALLOWED_ROOTS)


def count_dcm(d: Path, cap: int = 5000) -> int:
    n = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.lower().endswith('.dcm'):
                    n += 1
                    if n >= cap:
                        break
    except OSError:
        pass
    return n


def is_roi_series_name(name: str) -> bool:
    return ('output_dicom' in name
            and not any(name.endswith(s) for s in
                        ('_cylinder', '_ring', '_cylinder_bone', '_ring_bone')))


def is_ground_truth_name(name: str) -> bool:
    """The 12 hand-labeled GT series are named exactly <digits>_output_dicom."""
    return re.fullmatch(r'\d+_output_dicom', name) is not None


# ───────────────────────────────────────────────────────── subject detection

def detect_context(input_dir: Path) -> dict:
    """Infer subject / group / timepoint / reference from the folder layout:
    <GROUP>/<N MONTH>/<SUBJECT>/dicom_t*  with GT at <GROUP>/3 MONTH/<SUBJ>/<SUBJ>_output_dicom
    """
    parts = list(input_dir.parts)
    info = {'subject': None, 'group': None, 'timepoint': None, 'is_3m': None,
            'tp_short': None, 'ref_input': None, 'ref_roi': None,
            'suggested_output': None, 'n_dcm': count_dcm(input_dir)}

    month_idx = None
    for i, p in enumerate(parts):
        if re.fullmatch(r'\d+\s*MONTH[S]?', p.strip(), re.IGNORECASE):
            month_idx = i
            break
    if month_idx is not None:
        m = re.match(r'(\d+)', parts[month_idx].strip())
        months = int(m.group(1))
        info['timepoint'] = parts[month_idx]
        info['is_3m'] = (months == 3)
        info['tp_short'] = f'{months}m'
        if month_idx >= 1:
            info['group'] = parts[month_idx - 1]
        if month_idx + 1 < len(parts):
            cand = parts[month_idx + 1]
            if re.fullmatch(r'\d+', cand):
                info['subject'] = cand
    if info['subject'] is None:
        for p in reversed(parts):
            if re.fullmatch(r'\d{4,6}', p):
                info['subject'] = p
                break

    # Subject directory = where outputs conventionally live.
    if info['subject'] and info['subject'] in parts:
        subj_dir = Path(*parts[:parts.index(info['subject']) + 1])
    else:
        subj_dir = input_dir.parent

    # Suggested output name follows the existing convention
    # (37951_6m_output_dicom); at 3 months add _pred so the hand-labeled
    # ground truth 37951_output_dicom can never be collided with.
    subj = info['subject'] or 'SUBJECT'
    tp = info['tp_short'] or 'tp'
    name = f'{subj}_{tp}_pred_output_dicom' if info['is_3m'] else f'{subj}_{tp}_output_dicom'
    info['suggested_output'] = str(subj_dir / name)

    # Reference scan+ROI for registration: same group, 3 MONTH, same subject.
    if info['group'] and info['subject'] and month_idx is not None and not info['is_3m']:
        group_dir = Path(*parts[:month_idx])
        for tp_name in ('3 MONTH', '3 MONTHS', '3 month'):
            ref_subj = group_dir / tp_name / info['subject']
            if ref_subj.is_dir():
                rois, dicoms = [], []
                try:
                    for e in sorted(os.scandir(ref_subj), key=lambda e: e.name):
                        if not e.is_dir():
                            continue
                        n = e.name
                        if is_roi_series_name(n):
                            rois.append(Path(e.path))
                        elif count_dcm(Path(e.path), cap=600) >= 500:
                            dicoms.append(Path(e.path))
                except OSError:
                    pass
                gt = [r for r in rois if is_ground_truth_name(r.name)]
                if gt:
                    info['ref_roi'] = str(gt[0])
                elif rois:
                    info['ref_roi'] = str(rois[0])
                pref = [d for d in dicoms if d.name.startswith('dicom')]
                if pref or dicoms:
                    info['ref_input'] = str((pref or dicoms)[0])
                break
    return info


# ──────────────────────────────────────────────────────────── job execution

def save_jobs():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with JOB_LOCK:
        slim = {jid: {k: v for k, v in j.items() if k != 'log_text'}
                for jid, j in JOBS.items()}
    JOBS_FILE.write_text(json.dumps(slim, indent=1))


def load_jobs():
    if JOBS_FILE.exists():
        try:
            data = json.loads(JOBS_FILE.read_text())
            for jid, j in data.items():
                if j.get('status') in ('queued', 'running'):
                    j['status'] = 'interrupted'
                JOBS[jid] = j
        except (json.JSONDecodeError, OSError):
            pass


def job_log_path(jid: str) -> Path:
    return APP_DIR / f'{jid}.log'


def run_step(job, cmd, log_fh):
    log_fh.write(f'\n$ {" ".join(str(c) for c in cmd)}\n\n')
    log_fh.flush()
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(REPO_DIR),
                            stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    CURRENT_PROC[job['id']] = proc
    rc = proc.wait()
    CURRENT_PROC.pop(job['id'], None)
    return rc


def clear_previous_output(out: Path):
    """Remove series dirs + preview from an earlier run of the same output, so
    stale files (e.g. a different bone threshold) can't survive underneath."""
    if is_ground_truth_name(out.name):
        raise RuntimeError(f'{out.name} matches the ground-truth naming pattern '
                           '— refusing to overwrite it. Choose another name.')
    for suf in SERIES_SUFFIXES:
        d = out.parent / (out.name + suf)
        if d.is_dir():
            shutil.rmtree(d)
    png = out.parent / (out.name + '_axial_view.png')
    if png.exists():
        png.unlink()


def worker():
    while True:
        jid = JOB_QUEUE.get()
        job = JOBS.get(jid)
        if job is None or job.get('status') == 'cancelled':
            continue
        job['status'] = 'running'
        job['started'] = time.time()
        save_jobs()
        try:
            _run_job(job)
        except Exception as e:                          # noqa: BLE001
            job['status'] = 'failed'
            job['error'] = str(e)
            with open(job_log_path(jid), 'a') as fh:
                fh.write(f'\nAPP ERROR: {e}\n')
        job['finished'] = time.time()
        save_jobs()


def _run_job(job):
    p = job['params']
    out = Path(p['output'])
    thr = str(p.get('bone_threshold', DEFAULT_BONE_THRESHOLD))
    with open(job_log_path(job['id']), 'a') as fh:
        if p.get('overwrite'):
            clear_previous_output(out)
            fh.write(f'Cleared previous series for {out.name}\n')

        if job['mode'] == 'later_reg':
            fit_json = APP_DIR / f'{job["id"]}_fit.json'
            rc = run_step(job, [PYTHON, '3_inference.py',
                                '--input', p['input'], '--output', out,
                                '--fit-only', fit_json, '--fast-fit'], fh)
            if rc != 0:
                raise RuntimeError(f'network detection step exited with code {rc}')
            cmd = [PYTHON, '4_propagate_roi.py',
                   '--ref-input', p['ref_input'], '--ref-roi', p['ref_roi'],
                   '--target-input', p['input'], '--target-fit', fit_json,
                   '--output', out, '--bone-refine', '--bone-threshold', thr]
            if p.get('wide_search'):
                cmd.append('--wide-search')
            rc = run_step(job, cmd, fh)
            if rc != 0:
                raise RuntimeError(f'registration step exited with code {rc} '
                                   '(see log — a dice below --dice-min refuses to write)')
        else:
            cmd = [PYTHON, '3_inference.py', '--input', p['input'],
                   '--output', out, '--bone-refine', '--bone-threshold', thr]
            if job['mode'] == 'later_raw':
                cmd.append('--report-defect-offset')
            rc = run_step(job, cmd, fh)
            if rc != 0:
                raise RuntimeError(f'inference exited with code {rc}')

        fh.write('\nComputing ROI volumes and BV/TV from the written series...\n')
        fh.flush()
        try:
            job['results'] = compute_results(out, float(thr))
            fh.write('done.\n')
        except Exception as e:                          # noqa: BLE001
            fh.write(f'volume computation failed: {e}\n')

    log_text = job_log_path(job['id']).read_text(errors='replace')
    job['metrics'] = parse_metrics(log_text)
    job['checks'] = build_checks(job)
    worst = min((c['status'] for c in job['checks']), default='pass',
                key=lambda s: {'fail': 0, 'warn': 1, 'pass': 2}[s])
    job['status'] = 'done' if worst == 'pass' else ('done_warn' if worst == 'warn' else 'done_fail')
    preview = out.parent / (out.name + '_axial_view.png')
    if preview.exists():
        job['preview'] = str(preview)


# ─────────────────────────────────────────────── metrics, checks, BV/TV

def parse_metrics(log: str) -> dict:
    m = {}
    def grab(pattern, cast=float, group=1, last=False):
        found = re.findall(pattern, log)
        if found:
            v = found[-1] if last else found[0]
            if isinstance(v, tuple):
                v = v[group - 1]
            try:
                return cast(v)
            except ValueError:
                return None
        return None

    m['n_slices'] = grab(r'Found (\d+) DICOM slices', int)
    m['device'] = grab(r'Device\s*:\s*(\S+)', str)
    m['raw_voxels'] = grab(r'raw predicted voxels\s*:\s*([\d,]+)',
                           lambda s: int(s.replace(',', '')))
    m['tilt_deg'] = grab(r'tilt[^:\n]*:?\s*([\d.]+)\s*°', last=True)
    eig = re.search(r'eigenvalues \(mm²\)\s*:\s*\[([^\]]+)\]', log)
    if eig:
        try:
            m['eigenvalues'] = [float(x) for x in eig.group(1).split()]
        except ValueError:
            pass
    roi = re.search(r'ROI check\s*:\s*([\d.]+) / ([\d.]+) mm³ enclosed '
                    r'\(([\d.]+)% of the template\)', log)
    if roi:
        m['roi_enclosed_pct'] = float(roi.group(3))
    m['active_slices'] = grab(r'Active Z\s*:\s*\d+\s*→\s*\d+\s*\((\d+) slices\)', int)
    m['low_active_warning'] = grab(r'WARNING: only (\d+) active slices', int)
    m['dice'] = grab(r'bone dice = ([\d.]+)')
    m['spin_deg'] = grab(r'best spin start = (\-?\d+) deg', int)
    m['no_prediction'] = 'No defect region predicted' in log
    off = re.search(r'low-density centroid[^\n]*→\s*([\d.]+) mm', log)
    if off:
        m['void_offset_mm'] = float(off.group(1))
    return {k: v for k, v in m.items() if v is not None and v is not False}


def compute_results(output_base: Path, bone_threshold: float) -> dict:
    """Voxel-count each written series with pydicom -> volumes, BV/TV, ratio."""
    import numpy as np
    import pydicom

    def series_stats(d: Path):
        files = sorted(f for f in os.listdir(d) if f.lower().endswith('.dcm'))
        if not files:
            return None
        vox = 0
        spacing = None
        z_positions = []
        for f in files:
            ds = pydicom.dcmread(str(d / f))
            arr = ds.pixel_array
            vox += int(np.count_nonzero(arr))
            if spacing is None:
                ps = getattr(ds, 'PixelSpacing', [1, 1])
                st = getattr(ds, 'SliceThickness', None)
                spacing = [float(st) if st else None, float(ps[0]), float(ps[1])]
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp is not None and len(z_positions) < 3:
                z_positions.append([float(x) for x in ipp])
        if spacing[0] is None and len(z_positions) >= 2:
            import math
            spacing[0] = math.dist(z_positions[0], z_positions[1])
        if spacing[0] is None:
            spacing[0] = 1.0
        return vox, spacing, len(files)

    res = {'bone_threshold': bone_threshold, 'series': {}}
    for suf in SERIES_SUFFIXES:
        d = output_base.parent / (output_base.name + suf)
        if d.is_dir():
            st = series_stats(d)
            if st:
                vox, sp, nf = st
                voxel_mm3 = sp[0] * sp[1] * sp[2]
                res['series'][suf or 'union'] = {
                    'voxels': vox, 'volume_mm3': round(vox * voxel_mm3, 2),
                    'files': nf}
    s = res['series']
    def ratio(a, b):
        return round(s[a]['voxels'] / s[b]['voxels'], 4) if \
            a in s and b in s and s[b]['voxels'] else None
    res['bvtv_core'] = ratio('_cylinder_bone', '_cylinder')
    res['bvtv_ring'] = ratio('_ring_bone', '_ring')
    if res['bvtv_core'] and res['bvtv_ring']:
        res['core_to_ring'] = round(res['bvtv_core'] / res['bvtv_ring'], 3)
    return res


def build_checks(job) -> list:
    m = job.get('metrics', {})
    checks = []
    def add(name, status, detail):
        checks.append({'name': name, 'status': status, 'detail': detail})

    if job['mode'] == '3m':
        add('Placement method', 'pass',
            'Direct network placement — 3-month scans are the training '
            'timepoint (centre accurate to ~0.21 mm on GT).')
    elif job['mode'] == 'later_reg':
        add('Placement method', 'pass',
            'Registration from the 3-month reference ROI — the validated '
            'workflow for non-3-month scans.')
    else:
        add('Placement method', 'warn',
            'RAW network placement on a non-3-month scan. The network has a '
            'measured 2–3 mm placement bias at later timepoints. Use this ROI '
            'for detection only, or re-run with a 3-month reference.')

    if m.get('no_prediction'):
        add('Defect detected', 'fail', 'The network predicted no defect region.')
        return checks

    tilt = m.get('tilt_deg')
    if tilt is not None:
        lo, hi = TILT_RANGE
        if lo <= tilt <= hi:
            add('Axis tilt', 'pass', f'{tilt:.1f}° — inside the GT range {lo}–{hi}°.')
        elif tilt < 30:
            add('Axis tilt', 'fail',
                f'{tilt:.1f}° — near slice-aligned. GT axes are {lo}–{hi}° oblique; '
                'this fit is almost certainly a misplaced blob. Discard the run.')
        else:
            add('Axis tilt', 'warn',
                f'{tilt:.1f}° — outside the GT range {lo}–{hi}°. Inspect the axial preview.')

    eig = m.get('eigenvalues')
    if eig and len(eig) == 3:
        # Judge the smallest (axis, GT 5.33 mm²) and largest (transverse, GT
        # ~21 mm²) eigenvalues only: raw network masks are always elongated
        # (middle/largest 0.35-0.41 on all 12 GT subjects), and that is NOT
        # evidence of a displaced centre.
        e1, e2, e3 = sorted(eig)
        ok = EIG1_RANGE[0] <= e1 <= EIG1_RANGE[1] and \
            EIG23_RANGE[0] <= e3 <= EIG23_RANGE[1]
        if ok:
            add('Mask eigenvalues', 'pass',
                f'({e1:.2f}, {e2:.2f}, {e3:.2f}) mm² vs GT (5.33, 20.98, 20.99). '
                'A small middle value is normal — raw masks are elongated.')
        elif e1 > 50 or e3 > 200:
            add('Mask eigenvalues', 'fail',
                f'({e1:.2f}, {e2:.2f}, {e3:.2f}) mm² — an order of magnitude off the '
                'template: the mask was a diffuse blob, the ROI is misplaced.')
        else:
            add('Mask eigenvalues', 'warn',
                f'({e1:.2f}, {e2:.2f}, {e3:.2f}) mm² vs GT (5.33, 20.98, 20.99) — '
                'inspect the axial preview before trusting placement.')

    pct = m.get('roi_enclosed_pct')
    if pct is not None:
        if pct >= 99.9:
            add('ROI fully enclosed', 'pass', f'{pct:.2f}% of the rigid template written.')
        else:
            add('ROI fully enclosed', 'warn',
                f'Crop clips the template to {pct:.2f}% — volumes and BV/TV are '
                'under-reported.')

    if m.get('low_active_warning'):
        add('Active slices', 'warn',
            f'Only {m["low_active_warning"]} active slices predicted — weak detection.')
    elif m.get('active_slices'):
        add('Active slices', 'pass', f'{m["active_slices"]} slices in the ROI span.')

    if job['mode'] == 'later_reg':
        dice = m.get('dice')
        if dice is None:
            add('Registration dice', 'warn', 'No dice value found in the log.')
        elif dice >= 0.7:
            add('Registration dice', 'pass',
                f'{dice:.3f} — comparable to the validated 6-month runs (0.72–0.75).')
        elif dice >= 0.5:
            add('Registration dice', 'warn',
                f'{dice:.3f} — above the write threshold (0.5) but below the '
                'validated runs (0.72–0.75). Inspect the axial preview.')
        else:
            add('Registration dice', 'fail', f'{dice:.3f} — registration failed.')

    r = job.get('results') or {}
    if r.get('bvtv_ring') is not None and r['bvtv_ring'] < 0.30:
        add('Reference ring', 'warn',
            f'Ring BV/TV {100 * r["bvtv_ring"]:.1f}% is below the expected '
            '~40–50% dilution ceiling — the ring may not be seated in intact bone.')
    return checks


# ─────────────────────────────────────────────────────────────── HTTP layer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):                  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._json({'error': msg}, code)

    def do_GET(self):                                    # noqa: N802
        url = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(url.query))
        route = url.path

        if route == '/':
            body = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif route == '/api/config':
            self._json({'data_root': str(DATA_ROOT),
                        'bone_threshold': DEFAULT_BONE_THRESHOLD})

        elif route == '/api/browse':
            p = Path(q.get('path') or DATA_ROOT)
            if not p.is_dir():
                return self._err(f'not a directory: {p}')
            if not path_allowed(p):
                return self._err('path outside the allowed data root', 403)
            dirs = []
            try:
                entries = sorted(os.scandir(p), key=lambda e: e.name.lower())
            except OSError as e:
                return self._err(str(e))
            for e in entries:
                if not e.is_dir() or e.name.startswith('.'):
                    continue
                d = Path(e.path)
                n = count_dcm(d, cap=1500)
                dirs.append({'name': e.name, 'path': str(d), 'n_dcm': n,
                             'is_roi': is_roi_series_name(e.name),
                             'is_gt': is_ground_truth_name(e.name)})
            parent = str(p.parent) if path_allowed(p.parent) else None
            self._json({'path': str(p), 'parent': parent, 'dirs': dirs})

        elif route == '/api/suggest':
            p = Path(q.get('input', ''))
            if not p.is_dir():
                return self._err('input directory not found')
            self._json(detect_context(p))

        elif route == '/api/jobs':
            with JOB_LOCK:
                jobs = [_job_summary(JOBS[j]) for j in
                        sorted(JOBS, key=lambda j: JOBS[j]['created'], reverse=True)]
            self._json({'jobs': jobs})

        elif route.startswith('/api/jobs/'):
            jid = route.split('/')[3]
            job = JOBS.get(jid)
            if not job:
                return self._err('no such job', 404)
            out = dict(job)
            lp = job_log_path(jid)
            frm = int(q.get('log_from', 0))
            if lp.exists():
                text = lp.read_text(errors='replace')
                out['log'] = text[frm:]
                out['log_len'] = len(text)
            else:
                out['log'] = ''
                out['log_len'] = 0
            self._json(out)

        elif route == '/api/image':
            p = Path(q.get('path', ''))
            if not (p.suffix == '.png' and p.is_file() and path_allowed(p)):
                return self._err('not an allowed image', 403)
            body = p.read_bytes()
            self.send_response(200)
            self.send_header('Cache-Control', 'max-age=60')
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self._err('unknown route', 404)

    def do_POST(self):                                   # noqa: N802
        route = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError:
            return self._err('bad JSON')

        if route == '/api/run':
            return self._start_job(payload)

        if route.startswith('/api/jobs/') and route.endswith('/cancel'):
            jid = route.split('/')[3]
            job = JOBS.get(jid)
            if not job:
                return self._err('no such job', 404)
            proc = CURRENT_PROC.get(jid)
            if proc:
                proc.terminate()
            job['status'] = 'cancelled'
            save_jobs()
            return self._json({'ok': True})

        self._err('unknown route', 404)

    def _start_job(self, p):
        mode = p.get('mode')
        if mode not in ('3m', 'later_reg', 'later_raw'):
            return self._err('mode must be 3m, later_reg or later_raw')
        input_dir = Path(p.get('input') or '')
        if not p.get('input') or not input_dir.is_dir() or count_dcm(input_dir, cap=10) == 0:
            return self._err('input directory does not exist or contains no .dcm files')
        if not path_allowed(input_dir):
            return self._err('input is outside the allowed data root')
        output = Path(p.get('output') or '')
        if not output.name:
            return self._err('output path required')
        if not path_allowed(output.parent):
            return self._err('output is outside the allowed data root')
        if is_ground_truth_name(output.name):
            return self._err(f'"{output.name}" matches the hand-labeled ground-truth '
                             'naming pattern — choose a different output name')
        if output.resolve() == input_dir.resolve() or \
                input_dir.resolve() in output.resolve().parents:
            return self._err('output must not be the input directory')
        exists = any((output.parent / (output.name + s)).exists()
                     for s in SERIES_SUFFIXES)
        if exists and not p.get('overwrite'):
            return self._err(f'output series "{output.name}" already exists — '
                             'tick Overwrite to replace it')
        if mode == 'later_reg':
            for key in ('ref_input', 'ref_roi'):
                val = p.get(key)
                if not val or not Path(val).is_dir():
                    return self._err(f'{key} is required for registration placement')
                if not path_allowed(Path(val)):
                    return self._err(f'{key} is outside the allowed data root')
            if count_dcm(Path(p['ref_input']), cap=10) == 0:
                return self._err('ref_input contains no .dcm files')
        try:
            thr = float(p.get('bone_threshold', DEFAULT_BONE_THRESHOLD))
        except (TypeError, ValueError):
            return self._err('bone_threshold must be a number')

        jid = uuid.uuid4().hex[:12]
        job = {'id': jid, 'created': time.time(), 'status': 'queued',
               'mode': mode,
               'label': p.get('label') or output.name,
               'params': {'input': str(input_dir), 'output': str(output),
                          'ref_input': p.get('ref_input'),
                          'ref_roi': p.get('ref_roi'),
                          'bone_threshold': thr,
                          'wide_search': bool(p.get('wide_search')),
                          'overwrite': bool(p.get('overwrite'))}}
        APP_DIR.mkdir(parents=True, exist_ok=True)
        job_log_path(jid).write_text('')
        with JOB_LOCK:
            JOBS[jid] = job
        save_jobs()
        JOB_QUEUE.put(jid)
        self._json({'id': jid})


def _job_summary(j):
    return {k: j.get(k) for k in
            ('id', 'created', 'started', 'finished', 'status', 'mode', 'label')}


# ────────────────────────────────────────────────────────────────────── main

def main():
    ap = argparse.ArgumentParser(description='Defect segmentation lab web app')
    ap.add_argument('--host', default='127.0.0.1',
                    help='bind address (0.0.0.0 to allow other lab machines)')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--extra-root', action='append', default=[],
                    help='additional directory the app may browse/write')
    ap.add_argument('--open', action='store_true', help='open the browser')
    args = ap.parse_args()

    for r in args.extra_root:
        ALLOWED_ROOTS.append(Path(r).resolve())

    APP_DIR.mkdir(parents=True, exist_ok=True)
    load_jobs()
    threading.Thread(target=worker, daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f'http://{"127.0.0.1" if args.host == "0.0.0.0" else args.host}:{args.port}'
    print(f'Defect segmentation app: {url}')
    print(f'Data root: {DATA_ROOT}')
    if args.open:
        import webbrowser
        threading.Timer(0.6, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nbye')


if __name__ == '__main__':
    main()
