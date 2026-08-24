"""
axial_view.py
Reslice an inference (or ground-truth) defect series PERPENDICULAR to the
cylinder axis — the true top-down axial view, in which the 10 mm defect reads as
a circle sitting inside the 18 mm reference ring, fully surrounded by bone.

The DICOM slice plane is NOT the anatomical axial plane here: the defect
cylinder axis is oblique (74-88 deg off the slice-stacking axis, mean 81) and differs per
subject. The axis is recovered by PCA on the output mask itself, the same fit
3_inference.py uses.

Typical notebook use:

    from axial_view import AxialView

    av = AxialView(INPUT_DIR, OUTPUT_BASE)   # OUTPUT_BASE = the union series
    av.summary()
    av.show()                 # 8 mm slab MIP through the defect
    av.show(t_mm=0.0)         # single plane through the defect centre
    av.interactive()          # slider through the slab
"""

import importlib.util
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates

_SPEC = importlib.util.spec_from_file_location(
    '_inference', Path(__file__).parent / '3_inference.py')
_INF = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INF)

OrientedCylinder = _INF.OrientedCylinder
fit_axis         = _INF.fit_axis
dcm_files_sorted = _INF.dcm_files_sorted


def _crop_offset(ds_orig, ds_crop):
    """Recover (row_min, col_min) of a cropped series inside the original frame."""
    iop = [float(v) for v in ds_orig.ImageOrientationPatient]
    row_cos, col_cos = np.array(iop[0:3]), np.array(iop[3:6])
    ps = [float(v) for v in ds_orig.PixelSpacing]
    shift = (np.array([float(v) for v in ds_crop.ImagePositionPatient])
             - np.array([float(v) for v in ds_orig.ImagePositionPatient]))
    return (int(round(np.dot(shift, col_cos) / ps[0])),
            int(round(np.dot(shift, row_cos) / ps[1])))


class AxialView:
    """Loads a masked output series, fits its cylinder axis, and reslices the CT
    perpendicular to that axis."""

    def __init__(self, input_dir, output_dir, fov_mm=11.0, load_margin_mm=3.0,
                 stride=None):
        """`stride` decimates the load in all three axes. Default: chosen so the
        effective voxel is ~0.1 mm — 1 (unchanged) for the in vivo scans, ~7 for
        15 µm ex vivo µCT, whose native subvolume would be ~10 GB."""
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.fov_mm     = float(fov_mm)

        hdr_out = dcm_files_sorted(self.output_dir)
        hdr_in  = dcm_files_sorted(self.input_dir)
        self._in_by_inst = {inst: p for inst, p in hdr_in}

        ds_in0 = pydicom.dcmread(str(hdr_in[0][1]), stop_before_pixels=True)
        ps     = [float(v) for v in ds_in0.PixelSpacing]
        if len(hdr_in) > 1:
            p0 = np.array([float(v) for v in ds_in0.ImagePositionPatient])
            p1 = np.array([float(v) for v in pydicom.dcmread(
                str(hdr_in[1][1]), stop_before_pixels=True).ImagePositionPatient])
            sz = float(np.linalg.norm(p1 - p0))
        else:
            sz = float(getattr(ds_in0, 'SliceThickness', ps[0]))
        self.native_spacing = np.array([sz, ps[0], ps[1]])
        if stride is None:
            stride = max(1, int(round(0.1 / float(self.native_spacing.min()))))
        self.stride = k = int(stride)
        self.spacing = self.native_spacing * k    # sampling-grid spacing

        # ── Collect mask voxels in FULL-frame NATIVE coordinates ─────────────
        r0 = c0 = None
        pts, self.active_k = [], []
        for zi in range(0, len(hdr_out), k):
            inst, p = hdr_out[zi]
            arr = pydicom.dcmread(str(p)).pixel_array[::k, ::k]
            if not arr.any():
                continue
            if r0 is None:
                r0, c0 = _crop_offset(ds_in0, pydicom.dcmread(str(p), stop_before_pixels=True))
            rr, cc = np.where(arr != 0)
            pts.append(np.stack([np.full(rr.shape, zi), rr * k + r0, cc * k + c0], axis=1))
            self.active_k.append(zi)
        if not pts:
            raise ValueError(f'No non-empty slices in {self.output_dir}')
        coords = np.concatenate(pts, axis=0).astype(np.float64)
        self.crop_origin = (r0, c0)

        self.center_mm, self.axis, self.eigvals = fit_axis(
            coords * self.native_spacing)
        self.geom = OrientedCylinder(self.center_mm, self.axis, self.spacing)

        # ── Load only the CT subvolume the reslice can reach ─────────────────
        half = self.geom.bbox_half_mm() + load_margin_mm + max(0.0, fov_mm - self.geom.r_out)
        lo = np.floor((self.center_mm - half) / self.native_spacing).astype(int)
        hi = np.ceil((self.center_mm + half) / self.native_spacing).astype(int)
        ds_h, ds_w = ds_in0.Rows, ds_in0.Columns
        self.z0 = max(0, lo[0]); self.z1 = min(len(hdr_in) - 1, hi[0])
        self.r0 = max(0, lo[1]); self.r1 = min(ds_h, hi[1] + 1)
        self.c0 = max(0, lo[2]); self.c1 = min(ds_w, hi[2] + 1)

        self.ct = np.stack([
            pydicom.dcmread(str(hdr_in[z][1])).pixel_array[self.r0:self.r1:k,
                                                           self.c0:self.c1:k]
            for z in range(self.z0, self.z1 + 1, k)
        ], axis=0).astype(np.float32)
        # Origin in sampling-grid units: native index / stride, so that
        # plane()'s  idx = p / self.spacing - self.origin  hits ct[] cells.
        self.origin = np.array([self.z0, self.r0, self.c0],
                               dtype=np.float64) / k

        a = self.axis
        tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.e1 = np.cross(a, tmp); self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(a, self.e1); self.e2 /= np.linalg.norm(self.e2)

    # ── geometry info ────────────────────────────────────────────────────────

    @property
    def tilt_deg(self):
        """Angle between the cylinder axis and the DICOM slice-stacking axis."""
        return float(np.degrees(np.arccos(min(1.0, abs(self.axis[0])))))

    def summary(self):
        g = self.geom
        print(f'Series      : {self.output_dir.name}')
        print(f'Voxel size  : {self.spacing} mm')
        print(f'Centre      : {np.round(self.center_mm / self.spacing, 1)} px  '
              f'{np.round(self.center_mm, 2)} mm')
        print(f'Axis        : {np.round(self.axis, 4)}  (Z,row,col)')
        print(f'Tilt vs Z   : {self.tilt_deg:.1f}°  — the DICOM slice plane is NOT axial here')
        print(f'Eigenvalues : {np.round(self.eigvals, 2)} mm²  '
              f'[GT template = (5.33, 20.98, 20.99)]')
        print(f'Template    : h={2*g.half_h:.0f} mm, core r≤{g.r_core:.0f} mm, '
              f'ring {g.r_in:.0f}–{g.r_out:.0f} mm')
        print(f'CT subvol   : {self.ct.shape}  (z {self.z0}–{self.z1})')

    # ── resampling ───────────────────────────────────────────────────────────

    def _grid(self, fov_mm=None):
        fov  = self.fov_mm if fov_mm is None else fov_mm
        step = float(self.spacing.min())
        uv   = np.arange(-fov, fov + step, step)
        return uv, np.meshgrid(uv, uv, indexing='ij')

    def plane(self, t_mm=0.0, fov_mm=None):
        """Single resampled plane at offset `t_mm` along the axis."""
        uv, (U, V) = self._grid(fov_mm)
        p = (self.center_mm[None, None, :]
             + t_mm * self.axis[None, None, :]
             + U[..., None] * self.e1[None, None, :]
             + V[..., None] * self.e2[None, None, :])
        idx = (p / self.spacing[None, None, :]) - self.origin[None, None, :]
        img = map_coordinates(self.ct, [idx[..., 0], idx[..., 1], idx[..., 2]],
                              order=1, mode='constant', cval=0.0)
        return img, uv

    def slab(self, slab_mm=None, fov_mm=None, n=41, mode='max'):
        """Projection through a slab centred on the defect (`max` or `mean`)."""
        h   = self.geom.half_h if slab_mm is None else slab_mm / 2.0
        acc = None
        for t in np.linspace(-h, h, n):
            img, uv = self.plane(t, fov_mm)
            acc = img if acc is None else (np.maximum(acc, img) if mode == 'max' else acc + img)
        if mode != 'max':
            acc = acc / n
        return acc, uv

    # ── display ──────────────────────────────────────────────────────────────

    def show(self, t_mm=None, slab_mm=None, fov_mm=None, ax=None, title=None):
        """Render the axial view with the ROI circles overlaid.

        Pass `t_mm` for a single plane; omit it for a slab projection.
        """
        import matplotlib.pyplot as plt
        if t_mm is None:
            img, uv = self.slab(slab_mm, fov_mm)
            what = f'{2*self.geom.half_h if slab_mm is None else slab_mm:.0f} mm slab MIP'
        else:
            img, uv = self.plane(t_mm, fov_mm)
            what = f'single plane, t = {t_mm:+.1f} mm'

        nz = img[img > 0]
        vmin, vmax = (np.percentile(nz, [1, 99]) if nz.size else (0.0, 1.0))

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img.T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
                  extent=[uv[0], uv[-1], uv[0], uv[-1]])
        g = self.geom
        for r, col, lab in [(g.r_core, 'cyan',   f'{2*g.r_core:.0f} mm core'),
                            (g.r_in,   'yellow', f'{2*g.r_in:.0f} mm ring ID'),
                            (g.r_out,  'lime',   f'{2*g.r_out:.0f} mm ring OD')]:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ls='--', lw=1.6,
                                    color=col, label=lab))
        ax.plot(0, 0, 'r+', ms=10)
        ax.set_xlabel('mm'); ax.set_ylabel('mm')
        ax.set_title(title or f'Axial view perp. to fitted axis  ({what})\n'
                              f'axis (Z,row,col) = {np.round(self.axis, 3)}', fontsize=10)
        ax.legend(loc='lower right', fontsize=8)
        return ax

    def interactive(self, fov_mm=None):
        """ipywidgets slider stepping through the slab, plane by plane."""
        import matplotlib.pyplot as plt
        from ipywidgets import interact, FloatSlider

        h = self.geom.half_h

        def _draw(t_mm):
            fig, ax = plt.subplots(figsize=(7, 7))
            self.show(t_mm=t_mm, fov_mm=fov_mm, ax=ax)
            plt.show()

        return interact(_draw, t_mm=FloatSlider(
            min=-h, max=h, step=float(self.spacing.min()), value=0.0,
            description='t (mm)', continuous_update=False,
            style={'description_width': 'initial'}))
