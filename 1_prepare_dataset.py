"""
1_prepare_dataset.py
For each of the 12 labeled subjects, pairs original DICOM slices with their
corresponding _output_dicom ground-truth slices, reconstructs full 1200x1200
binary masks from the cropped output DICOM using ImagePositionPatient math,
then resizes both image and mask to 512x512 and saves as per-subject NPZ files.

Output: data/prepared/<subject_id>.npz
  images           : (N, 512, 512) float32, normalized to [0,1]
  masks            : (N, 512, 512) uint8 binary
  instance_numbers : (N,) int32
  has_defect       : (N,) bool  (True = slice has at least one foreground pixel)
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import pydicom
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / 'data'
PREPARED_DIR = DATA_DIR / 'prepared'
SUBJECTS_CSV = DATA_DIR / 'subjects.csv'

PREPARED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE      = 512
HU_MIN           = -500.0
HU_MAX           = 3000.0
N_EMPTY_PER_SUBJ = 50
RANDOM_SEED      = 42


def dcm_files_sorted(directory: Path):
    """Return .dcm files sorted by InstanceNumber (skips macOS resource forks)."""
    files = [f for f in directory.glob('*.dcm') if not f.name.startswith('._')]
    if not files:
        raise FileNotFoundError(f'No .dcm files in {directory}')
    headers = []
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        headers.append((int(ds.InstanceNumber), f))
    headers.sort(key=lambda x: x[0])
    return headers


def pixel_to_hu(pixel_array, ds):
    """Convert raw pixel values to HU using DICOM rescale tags (if present)."""
    arr = pixel_array.astype(np.float32)
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return arr * slope + intercept


def normalize_hu(hu_array, hu_min=HU_MIN, hu_max=HU_MAX):
    """Window HU to [hu_min, hu_max] and scale to [0, 1] float32."""
    clipped = np.clip(hu_array, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)


def compute_otsu_map(img_norm: np.ndarray) -> np.ndarray:
    """
    Binary bone mask via Otsu's method on a [0,1]-normalised image.
    Pure NumPy — no scikit-image dependency.
    Returns float32 array with values 0.0 (background) or 1.0 (bone/foreground).
    """
    hist, edges = np.histogram(img_norm.ravel(), bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(hist.sum())
    if total == 0:
        return np.zeros_like(img_norm, dtype=np.float32)
    p    = hist.astype(np.float64) / total
    w0   = np.cumsum(p)
    w1   = 1.0 - w0
    mc   = np.cumsum(p * centers)
    mT   = mc[-1]
    mu0  = mc / np.maximum(w0, 1e-10)
    mu1  = (mT - mc) / np.maximum(w1, 1e-10)
    sigma = w0 * w1 * (mu0 - mu1) ** 2
    thresh = float(centers[np.argmax(sigma)])
    return (img_norm >= thresh).astype(np.float32)


def resize2d(arr, size, order=1):
    """Resize a 2D numpy array to (size, size) using scipy zoom."""
    from scipy.ndimage import zoom
    zy = size / arr.shape[0]
    zx = size / arr.shape[1]
    return zoom(arr, (zy, zx), order=order)


def compute_crop_offset(ds_orig, ds_crop):
    """
    Recover (row_min, col_min) in the original 1200x1200 image where the
    cropped DICOM's top-left corner sits.

    The extraction notebook shifts ImagePositionPatient by:
        shift = col_min * col_sp * row_cosine + row_min * row_sp * col_cosine
    where row_cosine = IOP[0:3], col_cosine = IOP[3:6].

    Inverse: project shift onto each direction cosine.
    """
    iop = [float(v) for v in ds_orig.ImageOrientationPatient]
    row_cosine = np.array(iop[0:3])
    col_cosine = np.array(iop[3:6])

    ps = [float(v) for v in ds_orig.PixelSpacing]
    row_sp = ps[0]
    col_sp = ps[1]

    ipp_orig = np.array([float(v) for v in ds_orig.ImagePositionPatient])
    ipp_crop = np.array([float(v) for v in ds_crop.ImagePositionPatient])
    shift = ipp_crop - ipp_orig

    col_min = int(round(np.dot(shift, row_cosine) / col_sp))
    row_min = int(round(np.dot(shift, col_cosine) / row_sp))
    return row_min, col_min


def process_subject(subject_id, orig_dir, out_dir, rng):
    orig_dir = Path(orig_dir)
    out_dir  = Path(out_dir)

    print(f'\n─── {subject_id} ───')
    print(f'  orig : {orig_dir.name}')
    print(f'  mask : {out_dir.name}')

    orig_sorted = dcm_files_sorted(orig_dir)
    out_sorted  = dcm_files_sorted(out_dir)

    # Build instance_number → path maps
    orig_map = {inst: path for inst, path in orig_sorted}
    out_map  = {inst: path for inst, path in out_sorted}

    common_insts = sorted(set(orig_map) & set(out_map))
    if len(common_insts) != 1200:
        print(f'  WARNING: only {len(common_insts)} matching instance numbers')

    # --- Determine crop offset from first output DICOM that has pixel data ---
    print('  Computing crop offset from ImagePositionPatient...')
    row_min = col_min = None
    crop_h  = crop_w  = None
    for inst in common_insts:
        ds_out = pydicom.dcmread(str(out_map[inst]))
        arr_out = ds_out.pixel_array
        if arr_out.any():
            ds_orig_ref = pydicom.dcmread(str(orig_map[inst]), stop_before_pixels=True)
            row_min, col_min = compute_crop_offset(ds_orig_ref, ds_out)
            crop_h, crop_w   = ds_out.Rows, ds_out.Columns
            print(f'  Crop offset  : row_min={row_min}, col_min={col_min}')
            print(f'  Crop shape   : {crop_h}×{crop_w} px')
            break

    if row_min is None:
        print('  ERROR: no non-zero output slice found, skipping subject')
        return None

    # --- Identify active vs empty instance numbers ---
    print('  Scanning output DICOM for active slices (fast header scan)...')
    active_insts = []
    empty_insts  = []
    for inst in tqdm(common_insts, desc='  Scanning', leave=False):
        ds_out = pydicom.dcmread(str(out_map[inst]))
        if ds_out.pixel_array.any():
            active_insts.append(inst)
        else:
            empty_insts.append(inst)

    print(f'  Active slices : {len(active_insts)}  ({active_insts[0]}–{active_insts[-1]})')
    print(f'  Empty slices  : {len(empty_insts)}')

    # Sample empty slices for negative examples
    sampled_empty = rng.sample(empty_insts, min(N_EMPTY_PER_SUBJ, len(empty_insts)))

    all_insts = sorted(active_insts + sampled_empty)

    # --- Load reference header for HU conversion (from first orig slice) ---
    ds_ref = pydicom.dcmread(str(orig_map[common_insts[0]]), stop_before_pixels=True)
    orig_rows = ds_ref.Rows
    orig_cols = ds_ref.Columns

    images      = []
    masks       = []
    otsu_maps   = []
    inst_nums   = []
    has_defect  = []

    print(f'  Processing {len(all_insts)} slices...')
    for inst in tqdm(all_insts, desc='  Loading', leave=False):
        # --- Load original pixel data ---
        ds_orig = pydicom.dcmread(str(orig_map[inst]))
        hu = pixel_to_hu(ds_orig.pixel_array, ds_orig)
        img_norm = normalize_hu(hu)
        img_512  = resize2d(img_norm, TARGET_SIZE, order=1)
        otsu_512 = compute_otsu_map(img_512)

        # --- Reconstruct full-resolution binary mask ---
        ds_out   = pydicom.dcmread(str(out_map[inst]))
        out_arr  = ds_out.pixel_array
        is_active = bool(out_arr.any())

        full_mask = np.zeros((orig_rows, orig_cols), dtype=np.uint8)
        if is_active:
            binary_crop = (out_arr > 0).astype(np.uint8)
            r0, c0 = row_min, col_min
            r1 = min(r0 + crop_h, orig_rows)
            c1 = min(c0 + crop_w, orig_cols)
            full_mask[r0:r1, c0:c1] = binary_crop[:r1-r0, :c1-c0]

        mask_512 = resize2d(full_mask.astype(np.float32), TARGET_SIZE, order=0)
        mask_512 = (mask_512 > 0.5).astype(np.uint8)

        images.append(img_512)
        masks.append(mask_512)
        otsu_maps.append(otsu_512)
        inst_nums.append(inst)
        has_defect.append(is_active)

    images     = np.stack(images,    axis=0)   # (N, 512, 512) float32
    masks      = np.stack(masks,     axis=0)   # (N, 512, 512) uint8
    otsu_maps  = np.stack(otsu_maps, axis=0)   # (N, 512, 512) float32
    inst_nums  = np.array(inst_nums, dtype=np.int32)
    has_defect = np.array(has_defect, dtype=bool)

    pos_px  = masks.sum()
    tot_px  = masks.size
    pos_pct = 100.0 * pos_px / tot_px
    print(f'  Saved {len(images)} slices  |  foreground pixels: {pos_px:,} / {tot_px:,} ({pos_pct:.2f}%)')

    return {
        'images':           images,
        'masks':            masks,
        'otsu_maps':        otsu_maps,
        'instance_numbers': inst_nums,
        'has_defect':       has_defect,
    }


def main():
    if not SUBJECTS_CSV.exists():
        print(f'ERROR: {SUBJECTS_CSV} not found. Run 0_build_manifest.py first.')
        sys.exit(1)

    with open(SUBJECTS_CSV) as f:
        reader = csv.DictReader(f)
        subjects = list(reader)

    rng = random.Random(RANDOM_SEED)

    summary = []
    for row in subjects:
        sid      = row['subject_id']
        orig_dir = row['orig_dicom_dir']
        out_dir  = row['output_dicom_dir']
        out_npz  = PREPARED_DIR / f'{sid}.npz'

        if out_npz.exists():
            print(f'  SKIP {sid}: {out_npz.name} already exists')
            summary.append((sid, 'skipped'))
            continue

        result = process_subject(sid, orig_dir, out_dir, rng)
        if result is None:
            summary.append((sid, 'failed'))
            continue

        np.savez_compressed(
            str(out_npz),
            images=result['images'],
            masks=result['masks'],
            otsu_maps=result['otsu_maps'],
            instance_numbers=result['instance_numbers'],
            has_defect=result['has_defect'],
        )
        size_mb = out_npz.stat().st_size / 1e6
        print(f'  → {out_npz.name}  ({size_mb:.1f} MB)')
        summary.append((sid, 'ok'))

    print('\n═══ Summary ═══')
    for sid, status in summary:
        print(f'  {sid}: {status}')
    print(f'\nDone. NPZ files in {PREPARED_DIR}')


if __name__ == '__main__':
    main()
