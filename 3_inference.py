"""
3_inference.py
Runs the trained U-Net on a new in-vivo DICOM scan to localize the calvarial
defect, then stamps exact geometric masks matching the ground-truth annotation
protocol.

GEOMETRY (reverse-engineered from the 12 labeled _output_dicom volumes; all
subjects share an identical rigid template of ~1,432,500 voxels):

    oblique cylinder, axis  = normal to the local skull surface (subject-specific)
    height along axis       = 8.0 mm   (uniform)
    core cylinder           = r <= 5.0 mm          (10 mm diameter defect zone)
    gap                     = 5.0 < r < 7.0 mm     (excluded)
    reference ring          = 7.0 <= r <= 9.0 mm   (14 mm ID / 18 mm OD annulus)

The cylinder axis is NOT the slice-stacking (Z) axis — it is oblique, and tilts
by 20-25 degrees from subject to subject. It is recovered here by PCA on the raw
predicted mask: the U-Net is trained on these oblique cylinders, so the smallest-
variance eigenvector of its raw output is the cylinder axis (validated against GT:
smallest eigenvalue 5.47 mm^2 predicted vs 5.33 mm^2 ground truth, where
h^2/12 = 5.33 for h = 8 mm).

Usage:
    python 3_inference.py --input  /path/to/original_dicom_dir
                          --output /path/to/SUBJECTID_output_dicom
                          [--model  /path/to/best_model.pth]
                          [--threshold 0.5]
                          [--min-active-slices 10]

Three output directories are produced automatically:
  <output>/             — union  (core ∪ ring), same format as GT _output_dicom
  <output>_cylinder/    — 10 mm core cylinder only
  <output>_ring/        — 18 mm reference ring only (7–9 mm annulus)

Plus <output>_axial_view.png — a reslice PERPENDICULAR to the fitted axis, i.e.
the true top-down axial view in which the defect reads as a circle surrounded by
bone.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import zoom, label as cc_label, map_coordinates
import torch

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL = SCRIPT_DIR / 'models' / 'best_model.pth'

TARGET_SIZE     = 256
HU_MIN          = -500.0
HU_MAX          = 3000.0

# ── Ground-truth annotation template (mm) ───────────────────────────────────
CYL_HEIGHT_MM   = 8.0    # total height along the cylinder axis
CYLINDER_MM     = 5.0    # core radius        (10 mm diameter defect zone)
RING_INNER_MM   = 7.0    # ring inner radius  (14 mm ID)
RING_OUTER_MM   = 9.0    # ring outer radius  (18 mm OD)
BBOX_MARGIN_MM  = 0.0    # GT crop is tight to the cylinder bbox — no margin


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
    """Return [(key, path)] ordered along the stack.

    Prefers InstanceNumber. Some exports (e.g. ORS Dragonfly) omit it entirely,
    so fall back to spatial order: ImagePositionPatient projected onto the slice
    normal. Also accepts upper-case .DCM.
    """
    files = [f for f in directory.glob('*.dcm') if not f.name.startswith('._')]
    files += [f for f in directory.glob('*.DCM')
              if not f.name.startswith('._') and f not in files]
    if not files:
        raise FileNotFoundError(f'No .dcm/.DCM files found in {directory}')

    headers, missing = [], 0
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        if 'InstanceNumber' in ds and ds.InstanceNumber is not None:
            headers.append((int(ds.InstanceNumber), f))
        else:
            missing += 1
            iop = [float(v) for v in ds.ImageOrientationPatient]
            normal = np.cross(iop[0:3], iop[3:6])
            pos = float(np.dot([float(v) for v in ds.ImagePositionPatient], normal))
            headers.append((pos, f))

    if missing:
        if missing != len(files):
            raise ValueError(
                f'{directory}: {missing}/{len(files)} slices lack InstanceNumber — '
                'mixed ordering keys, refusing to guess')
        print(f'  [order] InstanceNumber absent on all {missing} slices; '
              'sorted by ImagePositionPatient along the slice normal')

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


# ── Oblique cylinder geometry ───────────────────────────────────────────────

class OrientedCylinder:
    """Core cylinder + reference ring about an arbitrary (oblique) axis.

    All coordinates are (Z, row, col). Voxel indices are converted to mm via
    `spacing` so the geometry stays correct under anisotropic voxels.

    Masks are generated lazily per slice inside the crop window — materialising
    the full 1200^3 volume would cost ~1.7 GB per series.
    """

    def __init__(self, center_mm, axis, spacing,
                 half_h=CYL_HEIGHT_MM / 2.0,
                 r_core=CYLINDER_MM, r_in=RING_INNER_MM, r_out=RING_OUTER_MM):
        self.c      = np.asarray(center_mm, dtype=np.float64)
        a           = np.asarray(axis, dtype=np.float64)
        self.a      = a / np.linalg.norm(a)
        self.sp     = np.asarray(spacing, dtype=np.float64)   # (sz, srow, scol)
        self.half_h = float(half_h)
        self.r_core = float(r_core)
        self.r_in   = float(r_in)
        self.r_out  = float(r_out)

    def bbox_half_mm(self) -> np.ndarray:
        """Half-extent of the cylinder's axis-aligned bounding box, per axis (mm).

        For a cylinder of half-height H and radius R with unit axis a, the
        extent along coordinate axis i is  H*|a_i| + R*sqrt(1 - a_i^2).
        """
        a2 = np.clip(self.a ** 2, 0.0, 1.0)
        return self.half_h * np.abs(self.a) + self.r_out * np.sqrt(1.0 - a2)

    def slice_mask(self, z: int, kind: str,
                   row0: int, nrow: int, col0: int, ncol: int) -> np.ndarray:
        """Binary mask of `kind` for DICOM slice `z`, over the crop window."""
        dz = z * self.sp[0] - self.c[0]
        dr = np.arange(row0, row0 + nrow, dtype=np.float64) * self.sp[1] - self.c[1]
        dc = np.arange(col0, col0 + ncol, dtype=np.float64) * self.sp[2] - self.c[2]
        R, C = np.meshgrid(dr, dc, indexing='ij')

        t  = dz * self.a[0] + R * self.a[1] + C * self.a[2]      # along axis
        d2 = dz ** 2 + R ** 2 + C ** 2 - t ** 2                  # radial^2
        d  = np.sqrt(np.maximum(d2, 0.0))

        within = np.abs(t) <= self.half_h
        core   = d <= self.r_core
        ring   = (d >= self.r_in) & (d <= self.r_out)

        if kind == 'cylinder':
            return within & core
        if kind == 'ring':
            return within & ring
        if kind == 'union':
            return within & (core | ring)
        raise ValueError(f'unknown kind: {kind}')


def fit_axis(coords_mm: np.ndarray):
    """PCA on the raw predicted mask → (center, axis, eigenvalues).

    The defect annotation is a flat 8 mm slab, so the smallest-variance
    eigenvector is the cylinder axis. For a perfect template the eigenvalues
    are (h^2/12, R^2/4, R^2/4) = (5.33, 20.25, 20.25) mm^2.
    """
    ctr = coords_mm.mean(axis=0)
    X   = coords_mm - ctr
    cov = X.T @ X / len(X)
    w, V = np.linalg.eigh(cov)          # ascending eigenvalues
    axis = V[:, 0]
    # Sign convention: make the dominant component positive (cosmetic only —
    # the cylinder is symmetric about its axis).
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return ctr, axis, w


def write_series(slices, geom: 'OrientedCylinder', kind: str, output_dir: Path,
                 row_min, crop_h, col_min, crop_w,
                 row_cos, col_cos, row_sp, col_sp,
                 z_lo: int, z_hi: int,
                 apply_otsu: bool = False):
    """Write one DICOM series (cropped, masked) to output_dir.

    Slices outside [z_lo, z_hi] are written as all-zero, matching the GT format
    where every instance is present but only the defect block carries data.

    When apply_otsu=True the geometric mask is further AND-ed with a per-slice
    Otsu bone map so only calcified/bone voxels survive within the ROI.  This
    is useful for bone-volume-fraction (BV/TV) quantification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_active = 0
    for i, (inst, path) in enumerate(slices):
        ds      = pydicom.dcmread(str(path))
        cropped = ds.pixel_array[row_min:row_min + crop_h,
                                 col_min:col_min + crop_w].astype(np.int32).copy()

        if z_lo <= i <= z_hi:
            m = geom.slice_mask(i, kind, row_min, crop_h, col_min, crop_w)
            if apply_otsu and m.any():
                hu   = pixel_to_hu(ds.pixel_array, ds)
                otsu = compute_otsu_map(normalize_hu(hu)).astype(bool)
                m &= otsu[row_min:row_min + crop_h, col_min:col_min + crop_w]
            cropped[~m] = 0
            if m.any():
                n_active += 1
        else:
            cropped[:] = 0

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

    out_files = [f for f in output_dir.glob('*.dcm') if not f.name.startswith('._')]
    total_mb  = sum(f.stat().st_size for f in out_files) / 1e6
    print(f'  → {len(out_files)} files, {n_active} non-empty  '
          f'({total_mb:.1f} MB)  in {output_dir.name}')


def write_axial_preview(slices, geom: 'OrientedCylinder', out_png: Path,
                        z_lo: int, z_hi: int, row_min: int, crop_h: int,
                        col_min: int, crop_w: int):
    """Reslice the CT PERPENDICULAR to the fitted axis and save a PNG.

    This is the true top-down axial view: the defect appears as a circle
    surrounded by the reference bone ring.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load only the subvolume covering the cylinder (~200^3, not 1200^3)
    sub = np.stack([
        pydicom.dcmread(str(slices[i][1])).pixel_array[
            row_min:row_min + crop_h, col_min:col_min + crop_w]
        for i in range(z_lo, z_hi + 1)
    ], axis=0).astype(np.float32)
    origin = np.array([z_lo, row_min, col_min], dtype=np.float64)

    # Orthonormal basis spanning the plane perpendicular to the axis
    a   = geom.a
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1  = np.cross(a, tmp);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(a, e1);   e2 /= np.linalg.norm(e2)

    fov  = geom.r_out + 2.0
    step = float(geom.sp.min())
    uv   = np.arange(-fov, fov + step, step)
    U, V = np.meshgrid(uv, uv, indexing='ij')

    # Maximum-intensity projection through the 8 mm slab, along the axis
    ts  = np.linspace(-geom.half_h, geom.half_h, 41)
    acc = np.full(U.shape, -np.inf, dtype=np.float32)
    for t in ts:
        p   = (geom.c[None, None, :]
               + t * a[None, None, :]
               + U[..., None] * e1[None, None, :]
               + V[..., None] * e2[None, None, :])
        idx = (p / geom.sp[None, None, :]) - origin[None, None, :]
        samp = map_coordinates(sub, [idx[..., 0], idx[..., 1], idx[..., 2]],
                               order=1, mode='constant', cval=0.0)
        acc = np.maximum(acc, samp)
    acc[~np.isfinite(acc)] = 0.0

    nz = acc[acc > 0]
    vmin, vmax = (np.percentile(nz, [1, 99]) if nz.size else (0.0, 1.0))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(acc.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
              extent=[uv[0], uv[-1], uv[0], uv[-1]])
    for r, col, lab in [(geom.r_core, 'cyan',   f'{2*geom.r_core:.0f} mm core'),
                        (geom.r_in,   'yellow', f'{2*geom.r_in:.0f} mm ring ID'),
                        (geom.r_out,  'lime',   f'{2*geom.r_out:.0f} mm ring OD')]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls='--', lw=1.6,
                                color=col, label=lab))
    ax.plot(0, 0, 'r+', ms=10)
    ax.set_xlabel('mm'); ax.set_ylabel('mm')
    ax.set_title(f'Axial view perp. to fitted axis  ({CYL_HEIGHT_MM:.0f} mm slab MIP)\n'
                 f'axis (Z,row,col) = {np.round(a, 3)}', fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_png.name}')


def load_model(model_path: Path, device: torch.device):
    """Load checkpoint and auto-detect in_channels from the saved weights."""
    from model import UNet
    ckpt  = torch.load(str(model_path), map_location=device)
    state = ckpt['model_state']
    # First conv layer weight shape: (out_ch, in_ch, kH, kW)
    in_ch = state['inc.net.0.weight'].shape[1]
    model = UNet(in_channels=in_ch, out_channels=1, features=(32, 64, 128, 256))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f'Loaded model from {model_path.name}  (epoch {epoch}, in_channels={in_ch})')
    return model, in_ch


@torch.no_grad()
def predict_slice(model, img_norm_curr: np.ndarray, device: torch.device, threshold: float,
                  in_ch: int = 2,
                  img_norm_prev: np.ndarray = None,
                  img_norm_next: np.ndarray = None,
                  z_pos: float = 0.5) -> np.ndarray:
    """Returns binary mask at TARGET_SIZE×TARGET_SIZE.

    When in_ch == 7 (2.5D + Z-position mode), pass img_norm_prev, img_norm_next and z_pos.
    Neighbouring slices default to the current slice when not provided (boundary handling).
    """
    curr_t  = resize2d(img_norm_curr, TARGET_SIZE, order=1)
    curr_ot = compute_otsu_map(curr_t)

    if in_ch == 7:
        prev_raw = img_norm_prev if img_norm_prev is not None else img_norm_curr
        next_raw = img_norm_next if img_norm_next is not None else img_norm_curr
        prev_t   = resize2d(prev_raw, TARGET_SIZE, order=1)
        next_t   = resize2d(next_raw, TARGET_SIZE, order=1)
        prev_ot  = compute_otsu_map(prev_t)
        next_ot  = compute_otsu_map(next_t)
        z_map    = np.full_like(curr_t, fill_value=z_pos, dtype=np.float32)
        # 7 channels: [prev_HU, prev_Otsu, curr_HU, curr_Otsu, next_HU, next_Otsu, Z_pos]
        inp = np.stack([prev_t, prev_ot, curr_t, curr_ot, next_t, next_ot, z_map], axis=0)
    else:
        inp = np.stack([curr_t, curr_ot], axis=0)  # (2, H, W) — original mode

    t     = torch.from_numpy(inp).unsqueeze(0).to(device)
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
    parser.add_argument('--margin-mm', type=float, default=BBOX_MARGIN_MM,
                        help='Extra padding around the crop box (default 0, matching GT)')
    parser.add_argument('--otsu-refine', action='store_true',
                        help='Also write *_cylinder_bone and *_ring_bone series '
                             'where the geometric mask is AND-ed with a per-slice '
                             'Otsu bone map (useful for BV/TV quantification)')
    parser.add_argument('--no-axial-preview', action='store_true',
                        help='Skip the perpendicular-to-axis axial view PNG')
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

    model, in_ch = load_model(model_path, device)

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

    # Slice spacing from the IPP step (falls back to SliceThickness)
    if n > 1:
        p0 = np.array([float(v) for v in ds_ref.ImagePositionPatient])
        p1 = np.array([float(v) for v in pydicom.dcmread(
            str(slices[1][1]), stop_before_pixels=True).ImagePositionPatient])
        slice_sp = float(np.linalg.norm(p1 - p0))
    else:
        slice_sp = float(getattr(ds_ref, 'SliceThickness', row_sp))
    if slice_sp <= 0:
        slice_sp = float(getattr(ds_ref, 'SliceThickness', row_sp))
    spacing = np.array([slice_sp, row_sp, col_sp])
    print(f'Voxel spacing (Z,row,col): {spacing} mm')

    # ── Pass 1: predict, keeping only voxel COORDINATES (not the 1.7 GB volume) ──
    print(f'\nPass 1 — predicting masks  (mode: {"2.5D + Z-pos" if in_ch == 7 else "2D"})...')
    per_slice_coords = [None] * n     # index → (rows, cols) arrays

    def _read_hu(path):
        ds = pydicom.dcmread(str(path))
        return normalize_hu(pixel_to_hu(ds.pixel_array, ds))

    for i, (inst, path) in enumerate(slices):
        img_n = _read_hu(path)

        if in_ch == 7:
            # Read neighbouring slices (re-use current at boundaries to avoid extra I/O)
            img_prev = _read_hu(slices[max(0, i - 1)][1])
            img_next = _read_hu(slices[min(n - 1, i + 1)][1])
            z_pos    = i / max(n - 1, 1)
            mask_small = predict_slice(model, img_n, device, args.threshold,
                                       in_ch=in_ch, img_norm_prev=img_prev,
                                       img_norm_next=img_next, z_pos=z_pos)
        else:
            mask_small = predict_slice(model, img_n, device, args.threshold, in_ch=in_ch)

        full_mask  = upsample_mask(mask_small, orig_h, orig_w)
        # Keep only the largest connected component; drop masks below min area
        if full_mask.any():
            labeled, n_comps = cc_label(full_mask)
            if n_comps > 0:
                sizes = np.bincount(labeled.ravel())[1:]
                if sizes.max() >= args.min_blob_area:
                    full_mask = (labeled == (sizes.argmax() + 1)).astype(np.uint8)
                else:
                    full_mask[:] = 0
        if full_mask.any():
            rr, cc = np.where(full_mask)
            per_slice_coords[i] = (rr.astype(np.int32), cc.astype(np.int32))

        if (i + 1) % 100 == 0 or i + 1 == n:
            n_act = sum(1 for k in range(i + 1) if per_slice_coords[k] is not None)
            print(f'  {i+1}/{n}  active so far: {n_act}')

    active_z = np.array([i for i in range(n) if per_slice_coords[i] is not None])

    # Keep only the largest contiguous Z block (gap tolerance = 20 slices).
    # This discards isolated false-positive predictions far from the main region.
    if len(active_z) > 1:
        MAX_Z_GAP = 20
        gaps   = np.diff(active_z)
        splits = np.where(gaps > MAX_Z_GAP)[0] + 1
        blocks = np.split(active_z, splits)
        kept   = max(blocks, key=len)
        if len(blocks) > 1:
            print(f'  [CC filter] kept largest Z block ({len(kept)} slices), '
                  f'dropped {len(active_z) - len(kept)} isolated slices')
        active_z = kept

    if len(active_z) == 0:
        print('No defect region predicted. Aborting — no output written.')
        return
    if len(active_z) < args.min_active_slices:
        print(f'\nWARNING: only {len(active_z)} active slices predicted '
              f'(threshold={args.threshold}). Output will be written but may be unreliable.')

    # ── Fit the oblique cylinder axis by PCA on the raw predicted mask ───────
    pts = []
    for i in active_z:
        rr, cc = per_slice_coords[i]
        pts.append(np.stack([np.full(rr.shape, i, dtype=np.float64),
                             rr.astype(np.float64), cc.astype(np.float64)], axis=1))
    coords_vox = np.concatenate(pts, axis=0)
    coords_mm  = coords_vox * spacing[None, :]

    center_mm, axis, eigvals = fit_axis(coords_mm)
    center_vox = center_mm / spacing

    print(f'\n── Fitted cylinder ──────────────────────────────────────────')
    print(f'  raw predicted voxels : {len(coords_mm):,}')
    print(f'  center (Z,row,col)   : {np.round(center_vox, 1)} px   '
          f'{np.round(center_mm, 2)} mm')
    print(f'  axis   (Z,row,col)   : {np.round(axis, 4)}')
    print(f'  eigenvalues (mm²)    : {np.round(eigvals, 2)}   '
          f'[GT template ≈ (5.33, 20.25, 20.25)]')
    tilt = np.degrees(np.arccos(min(1.0, abs(axis[0]))))
    print(f'  tilt from slice axis : {tilt:.1f}°  '
          f'(a Z-aligned stamp would be {tilt:.0f}° wrong)')
    print(f'  template             : h={CYL_HEIGHT_MM} mm, core r≤{CYLINDER_MM} mm, '
          f'ring {RING_INNER_MM}–{RING_OUTER_MM} mm')

    geom = OrientedCylinder(center_mm, axis, spacing)

    # ── Crop box + active Z range from the cylinder's own bounding box ───────
    half = geom.bbox_half_mm() + args.margin_mm
    lo   = (center_mm - half) / spacing
    hi   = (center_mm + half) / spacing

    z_lo = max(0,      int(np.floor(lo[0])))
    z_hi = min(n - 1,  int(np.ceil(hi[0])))
    row_min = max(0,      int(np.round(lo[1])))
    row_max = min(orig_h, int(np.round(hi[1])) + 1)
    col_min = max(0,      int(np.round(lo[2])))
    col_max = min(orig_w, int(np.round(hi[2])) + 1)
    crop_h  = row_max - row_min
    crop_w  = col_max - col_min

    print(f'\nActive Z    : {z_lo} → {z_hi}  ({z_hi - z_lo + 1} slices)')
    print(f'Bounding box: rows {row_min}→{row_max} ({crop_h} px)  '
          f'cols {col_min}→{col_max} ({crop_w} px)')

    union_dir = output_dir
    cyl_dir   = output_dir.parent / (output_dir.name + '_cylinder')
    ring_dir  = output_dir.parent / (output_dir.name + '_ring')

    write_kw = dict(row_min=row_min, crop_h=crop_h, col_min=col_min, crop_w=crop_w,
                    row_cos=row_cos, col_cos=col_cos, row_sp=row_sp, col_sp=col_sp,
                    z_lo=z_lo, z_hi=z_hi)

    # ── Pass 2: write all three DICOM series ─────────────────────────────────
    print('\nPass 2 — writing union (core ∪ ring) series...')
    write_series(slices, geom, 'union', union_dir, **write_kw)

    print(f'\nPass 3 — writing {2*CYLINDER_MM:.0f} mm core cylinder series...')
    write_series(slices, geom, 'cylinder', cyl_dir, **write_kw)

    print(f'\nPass 4 — writing {2*RING_OUTER_MM:.0f} mm reference ring series...')
    write_series(slices, geom, 'ring', ring_dir, **write_kw)

    if args.otsu_refine:
        cyl_bone_dir  = output_dir.parent / (output_dir.name + '_cylinder_bone')
        ring_bone_dir = output_dir.parent / (output_dir.name + '_ring_bone')
        print('\nPass 5 — writing Otsu-refined core cylinder (bone only)...')
        write_series(slices, geom, 'cylinder', cyl_bone_dir, apply_otsu=True, **write_kw)
        print('\nPass 6 — writing Otsu-refined reference ring (bone only)...')
        write_series(slices, geom, 'ring', ring_bone_dir, apply_otsu=True, **write_kw)

    if not args.no_axial_preview:
        print('\nRendering axial view (⟂ to fitted axis)...')
        write_axial_preview(slices, geom,
                            output_dir.parent / (output_dir.name + '_axial_view.png'),
                            z_lo, z_hi, row_min, crop_h, col_min, crop_w)

    print('\nDone.')


if __name__ == '__main__':
    main()
