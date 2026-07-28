"""
3_inference.py
Runs the trained U-Net on a new in-vivo DICOM scan to localize the calvarial
defect, then stamps exact geometric masks matching the ground-truth annotation
protocol: a 10 mm cylinder (central defect zone) and an 18 mm ring (surrounding
reference bone annulus).

Usage:
    python 3_inference.py --input  /path/to/original_dicom_dir
                          --output /path/to/SUBJECTID_output_dicom
                          [--model  /path/to/best_model.pth]
                          [--threshold 0.5]
                          [--min-active-slices 10]

Three output directories are produced automatically:
  <output>/             — union  (cylinder ∪ ring), same format as GT _output_dicom
  <output>_cylinder/    — 10 mm cylinder only
  <output>_ring/        — 18 mm ring only (annulus between cylinder and outer edge)
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import zoom, label as cc_label, uniform_filter1d
import torch

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL = SCRIPT_DIR / 'models' / 'best_model.pth'

TARGET_SIZE     = 256
HU_MIN          = -500.0
HU_MAX          = 3000.0
CYLINDER_MM     = 5.0    # radius of 10 mm cylinder (mm)
RING_OUTER_MM   = 9.0    # outer radius of 18 mm ring (mm)
BBOX_MARGIN_MM  = 2.0    # extra padding around bounding box (mm)


def compute_otsu_map(img_norm: np.ndarray) -> np.ndarray:
    """
    Binary bone mask via Otsu's method on a [0,1]-normalised image.
    Pure NumPy — no additional dependencies.
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def dcm_files_sorted(directory: Path):
    files = [f for f in directory.glob('*.dcm') if not f.name.startswith('._')]
    if not files:
        raise FileNotFoundError(f'No .dcm files found in {directory}')
    headers = []
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        headers.append((int(ds.InstanceNumber), f))
    headers.sort(key=lambda x: x[0])
    return headers


def pixel_to_hu(pixel_array, ds):
    arr       = pixel_array.astype(np.float32)
    slope     = float(getattr(ds, 'RescaleSlope',     1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return arr * slope + intercept


def normalize_hu(hu_array):
    clipped = np.clip(hu_array, HU_MIN, HU_MAX)
    return ((clipped - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def resize2d(arr, size, order=1):
    zy = size / arr.shape[0]
    zx = size / arr.shape[1]
    return zoom(arr, (zy, zx), order=order)


def upsample_mask(mask_small: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    zy = orig_h / mask_small.shape[0]
    zx = orig_w / mask_small.shape[1]
    up = zoom(mask_small.astype(np.float32), (zy, zx), order=1)
    return (up > 0.5).astype(np.uint8)


def crop_ipp(original_ipp, col_offset_px, row_offset_px, col_sp, row_sp, row_cos, col_cos):
    origin = np.array(original_ipp)
    shift  = col_offset_px * col_sp * row_cos + row_offset_px * row_sp * col_cos
    return (origin + shift).tolist()


def make_geometric_masks(pred_masks, active_z, orig_h, orig_w,
                         cyl_r_px: float, ring_r_px: float):
    """
    From predicted masks find per-slice centroid, smooth it, then generate
    a cylinder (solid circle) and ring (annulus) at the smoothed centroid.

    Returns
    -------
    cyl_masks  : (n, H, W) uint8  — 10 mm cylinder
    ring_masks : (n, H, W) uint8  — 18 mm ring (annulus)
    rows_c, cols_c : smoothed centroid arrays (length n)
    """
    centroids = []
    for zi in active_z:
        m = pred_masks[zi]
        if m.any():
            ys, xs = np.where(m)
            centroids.append((float(ys.mean()), float(xs.mean())))
        else:
            centroids.append(None)

    # fill None entries with nearest valid neighbour
    valid_idx = [i for i, c in enumerate(centroids) if c is not None]
    for i, c in enumerate(centroids):
        if c is None:
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            centroids[i] = centroids[nearest]

    rows_c = np.array([c[0] for c in centroids], dtype=np.float64)
    cols_c = np.array([c[1] for c in centroids], dtype=np.float64)

    # smooth centroid trajectory
    w = max(3, min(21, len(rows_c) // 5))
    if len(rows_c) >= w:
        rows_c = uniform_filter1d(rows_c, size=w)
        cols_c = uniform_filter1d(cols_c, size=w)

    ys_g, xs_g = np.mgrid[0:orig_h, 0:orig_w].astype(np.float32)
    n          = len(active_z)
    cyl_masks  = np.zeros((n, orig_h, orig_w), dtype=np.uint8)
    ring_masks = np.zeros((n, orig_h, orig_w), dtype=np.uint8)

    for k in range(n):
        cy, cx = rows_c[k], cols_c[k]
        d2 = (ys_g - cy) ** 2 + (xs_g - cx) ** 2
        cyl_masks[k]  = (d2 <= cyl_r_px ** 2).astype(np.uint8)
        ring_masks[k] = ((d2 > cyl_r_px ** 2) & (d2 <= ring_r_px ** 2)).astype(np.uint8)

    return cyl_masks, ring_masks, rows_c, cols_c


def write_series(slices, mask_vol, active_z_set, output_dir,
                 row_min, crop_h, col_min, crop_w,
                 row_cos, col_cos, row_sp, col_sp,
                 apply_otsu: bool = False):
    """Write one DICOM series (cropped, masked) to output_dir.

    When apply_otsu=True the geometric mask is further AND-ed with a per-slice
    Otsu bone map so only calcified/bone voxels survive within the ROI.  This
    is useful for bone-volume-fraction (BV/TV) quantification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (inst, path) in enumerate(slices):
        ds     = pydicom.dcmread(str(path))
        pixels = ds.pixel_array.copy().astype(np.int32)
        if i in active_z_set:
            geom = mask_vol[i].astype(bool)
            if apply_otsu:
                hu        = pixel_to_hu(ds.pixel_array, ds)
                img_norm  = normalize_hu(hu)
                otsu_bone = compute_otsu_map(img_norm).astype(bool)
                pixels[~(geom & otsu_bone)] = 0
            else:
                pixels[~geom] = 0
        else:
            pixels[:] = 0
        cropped = pixels[row_min:row_min + crop_h, col_min:col_min + crop_w]

        ds_out         = copy.deepcopy(ds)
        ds_out.Rows    = crop_h
        ds_out.Columns = crop_w
        ds_out.ImagePositionPatient = [
            f'{v:.6f}' for v in crop_ipp(
                [float(v) for v in ds.ImagePositionPatient],
                col_min, row_min, col_sp, row_sp, row_cos, col_cos)
        ]
        ds_out.SOPInstanceUID = generate_uid()
        ds_out.file_meta.MediaStorageSOPInstanceUID = ds_out.SOPInstanceUID
        orig_dtype           = ds.pixel_array.dtype
        ds_out.PixelData     = cropped.astype(orig_dtype).tobytes()
        ds_out.BitsAllocated = orig_dtype.itemsize * 8
        ds_out.BitsStored    = ds_out.BitsAllocated
        ds_out.HighBit       = ds_out.BitsAllocated - 1
        pydicom.dcmwrite(str(output_dir / path.name), ds_out)

        if (i + 1) % 200 == 0 or i + 1 == len(slices):
            print(f'  Written {i+1}/{len(slices)}')

    out_files = list(output_dir.glob('*.dcm'))
    total_mb  = sum(f.stat().st_size for f in out_files) / 1e6
    print(f'  → {len(out_files)} files  ({total_mb:.1f} MB)  in {output_dir.name}')


def load_model(model_path: Path, device: torch.device):
    from model import UNet
    model = UNet(in_channels=2, out_channels=1, features=(32, 64, 128, 256))
    ckpt  = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Loaded model from {model_path.name}  (epoch {epoch})')
    return model


@torch.no_grad()
def predict_slice(model, img_norm: np.ndarray, device: torch.device, threshold: float) -> np.ndarray:
    """Returns binary mask at TARGET_SIZE×TARGET_SIZE."""
    img_t    = resize2d(img_norm, TARGET_SIZE, order=1)
    otsu_t   = compute_otsu_map(img_t)                       # bone binary map at same size
    inp      = np.stack([img_t, otsu_t], axis=0)             # (2, H, W)
    t = torch.from_numpy(inp).unsqueeze(0).to(device)        # (1, 2, H, W)
    logit = model(t)
    prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Defect region inference → _output_dicom')
    parser.add_argument('--input',       required=True,  help='Path to original DICOM directory')
    parser.add_argument('--output',      required=True,  help='Path to write predicted _output_dicom')
    parser.add_argument('--model',       default=str(DEFAULT_MODEL), help='Path to model checkpoint')
    parser.add_argument('--threshold',   type=float, default=0.5,  help='Sigmoid threshold (default 0.5)')
    parser.add_argument('--min-active-slices', type=int, default=10,
                        help='Minimum predicted active slices before writing output (default 10)')
    parser.add_argument('--min-blob-area', type=int, default=200,
                        help='Min non-zero px per slice to keep (removes noise blobs, default 200)')
    parser.add_argument('--otsu-refine', action='store_true',
                        help='Also write *_cylinder_bone and *_ring_bone series '
                             'where the geometric mask is AND-ed with a per-slice '
                             'Otsu bone map (useful for BV/TV quantification)')
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    model_path = Path(args.model)

    if not input_dir.exists():
        print(f'ERROR: input directory not found: {input_dir}'); sys.exit(1)
    if not model_path.exists():
        print(f'ERROR: model not found: {model_path}'); sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f'Device  : {device}')
    print(f'Input   : {input_dir}')
    print(f'Output  : {output_dir}')

    model = load_model(model_path, device)

    print('\nLoading DICOM headers...')
    slices = dcm_files_sorted(input_dir)
    n      = len(slices)
    print(f'Found {n} DICOM slices')

    ds_ref = pydicom.dcmread(str(slices[0][1]), stop_before_pixels=True)
    orig_h = ds_ref.Rows
    orig_w = ds_ref.Columns
    iop    = [float(v) for v in ds_ref.ImageOrientationPatient]
    row_cos = np.array(iop[0:3])
    col_cos = np.array(iop[3:6])
    ps      = [float(v) for v in ds_ref.PixelSpacing]
    row_sp, col_sp = ps[0], ps[1]

    # ── Pass 1: predict all masks, collect full-res binary volumes ──────────
    print('\nPass 1 — predicting masks...')
    pred_masks = np.zeros((n, orig_h, orig_w), dtype=np.uint8)

    for i, (inst, path) in enumerate(slices):
        ds    = pydicom.dcmread(str(path))
        hu    = pixel_to_hu(ds.pixel_array, ds)
        img_n = normalize_hu(hu)
        mask_small = predict_slice(model, img_n, device, args.threshold)
        full_mask  = upsample_mask(mask_small, orig_h, orig_w)
        # Keep only the largest connected component; drop masks below min area
        if full_mask.any():
            labeled, n_comps = cc_label(full_mask)
            if n_comps > 0:
                sizes  = [(labeled == k).sum() for k in range(1, n_comps + 1)]
                biggest = sizes.index(max(sizes)) + 1
                if max(sizes) >= args.min_blob_area:
                    full_mask = (labeled == biggest).astype(np.uint8)
                else:
                    full_mask[:] = 0
        pred_masks[i] = full_mask

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f'  {i+1}/{n}  active so far: {pred_masks[:i+1].any(axis=(1,2)).sum()}')

    # ── Compute global bounding box from predicted volume ────────────────────
    active_z = np.where(pred_masks.any(axis=(1, 2)))[0]

    # Keep only the largest contiguous Z block (gap tolerance = 20 slices).
    # This discards isolated false-positive predictions far from the main region.
    if len(active_z) > 1:
        MAX_Z_GAP = 20
        gaps   = np.diff(active_z)
        splits = np.where(gaps > MAX_Z_GAP)[0] + 1
        blocks = np.split(active_z, splits)
        active_z = max(blocks, key=len)
        if len(blocks) > 1:
            dropped = sum(len(b) for b in blocks) - len(active_z)
            print(f'  [CC filter] kept largest Z block ({len(active_z)} slices), '
                  f'dropped {dropped} isolated slices')

    if len(active_z) < args.min_active_slices:
        print(f'\nWARNING: only {len(active_z)} active slices predicted '
              f'(threshold={args.threshold}). Output will be written but may be empty.')

    if len(active_z) == 0:
        print('No defect region predicted. Aborting — no output written.')
        return

    # ── Build geometric cylinder + ring masks ────────────────────────────────
    cyl_r_px  = CYLINDER_MM   / row_sp   # 10 mm cylinder radius in pixels
    ring_r_px = RING_OUTER_MM / row_sp   # 18 mm ring outer radius in pixels
    margin_px = int(np.ceil(BBOX_MARGIN_MM / row_sp))
    print(f'\nGeometry  : cylinder r={cyl_r_px:.1f} px ({CYLINDER_MM*2:.0f} mm diameter)')
    print(f'            ring outer r={ring_r_px:.1f} px ({RING_OUTER_MM*2:.0f} mm diameter)')

    cyl_vol, ring_vol, rows_c, cols_c = make_geometric_masks(
        pred_masks, active_z, orig_h, orig_w, cyl_r_px, ring_r_px)

    # union volume mapped back to full n-slice index
    union_full = np.zeros((n, orig_h, orig_w), dtype=np.uint8)
    cyl_full   = np.zeros((n, orig_h, orig_w), dtype=np.uint8)
    ring_full  = np.zeros((n, orig_h, orig_w), dtype=np.uint8)
    for k, zi in enumerate(active_z):
        cyl_full[zi]   = cyl_vol[k]
        ring_full[zi]  = ring_vol[k]
        union_full[zi] = np.clip(cyl_vol[k] + ring_vol[k], 0, 1)

    # Global bounding box: centroid range + ring outer radius + margin
    row_min = max(0,      int(rows_c.min()) - int(ring_r_px) - margin_px)
    row_max = min(orig_h, int(rows_c.max()) + int(ring_r_px) + margin_px + 1)
    col_min = max(0,      int(cols_c.min()) - int(ring_r_px) - margin_px)
    col_max = min(orig_w, int(cols_c.max()) + int(ring_r_px) + margin_px + 1)
    crop_h  = row_max - row_min
    crop_w  = col_max - col_min

    z_start = int(active_z[0])
    z_end   = int(active_z[-1])
    print(f'\nActive Z   : {z_start} → {z_end}  ({len(active_z)} slices)')
    print(f'Bounding box: rows {row_min}→{row_max} ({crop_h} px)  '
          f'cols {col_min}→{col_max} ({crop_w} px)')

    active_set   = set(active_z.tolist())
    union_dir    = output_dir
    cyl_dir      = output_dir.parent / (output_dir.name + '_cylinder')
    ring_dir     = output_dir.parent / (output_dir.name + '_ring')

    write_kw = dict(row_min=row_min, crop_h=crop_h, col_min=col_min, crop_w=crop_w,
                    row_cos=row_cos, col_cos=col_cos, row_sp=row_sp, col_sp=col_sp)

    # ── Pass 2: write all three DICOM series ─────────────────────────────────
    print('\nPass 2 — writing union (cylinder ∪ ring) series...')
    write_series(slices, union_full, active_set, union_dir, **write_kw)

    print('\nPass 3 — writing 10 mm cylinder series...')
    write_series(slices, cyl_full, active_set, cyl_dir, **write_kw)

    print('\nPass 4 — writing 18 mm ring series...')
    write_series(slices, ring_full, active_set, ring_dir, **write_kw)

    if args.otsu_refine:
        cyl_bone_dir  = output_dir.parent / (output_dir.name + '_cylinder_bone')
        ring_bone_dir = output_dir.parent / (output_dir.name + '_ring_bone')
        print('\nPass 5 — writing Otsu-refined 10 mm cylinder (bone only)...')
        write_series(slices, cyl_full, active_set, cyl_bone_dir,
                     apply_otsu=True, **write_kw)
        print('\nPass 6 — writing Otsu-refined 18 mm ring (bone only)...')
        write_series(slices, ring_full, active_set, ring_bone_dir,
                     apply_otsu=True, **write_kw)

    print('\nDone.')


if __name__ == '__main__':
    main()
