# Rabbit Craniofacial Defect Segmentation

Localizes the surgical calvarial defect in rabbit in-vivo micro-CT scans and stamps
the standard ROI template used for bone-healing quantification: a 10 mm core
cylinder over the defect plus an 18 mm reference-bone ring around it.

A 2D U-Net finds the defect region slice by slice. The raw prediction is then
replaced by an exact geometric template fitted to it, so the ROI you measure has
the same rigid dimensions for every subject and only its position and orientation
vary. Output is written as DICOM series matching the ground-truth
`_output_dicom` format, so it drops straight into an existing analysis workflow.

## The defect axis is oblique — this is the thing to know

The DICOM slice plane is **not** the anatomical axial plane for these scans. The
defect cylinder axis is tilted 20–25° off the slice-stacking axis, and the tilt
differs per subject. Scrolling the raw slices gives you a *coronal* cut through
the defect, not a top-down view.

`3_inference.py` recovers the true axis by PCA on the raw predicted mask: because
the network was trained on these oblique cylinders, the smallest-variance
eigenvector of its output is the cylinder axis. Validated against ground truth,
the smallest eigenvalue is 5.47 mm² predicted vs 5.33 mm² measured, where the
expected value for an 8 mm slab is h²/12 = 5.33.

Stamping a Z-aligned cylinder instead — as any pre-August-2026 output did — tilts
the ROI by the full 20–25° and makes the resulting volume and BV/TV numbers
invalid. If you have outputs generated before then, regenerate them.

## Geometry of the ROI template

Reverse-engineered from the 12 labeled ground-truth volumes, which all share an
identical rigid template of ~1,432,500 voxels:

| Element | Extent |
|---|---|
| Height along axis | 8.0 mm |
| Core cylinder | r ≤ 5.0 mm (10 mm defect zone) |
| Gap (excluded) | 5.0 < r < 7.0 mm |
| Reference ring | 7.0 ≤ r ≤ 9.0 mm (14 mm ID / 18 mm OD) |

## Install

```bash
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA, or CPU — `3_inference.py` picks the fastest
available device automatically.

## Running inference on a scan

```bash
python 3_inference.py --input /path/to/original_dicom_dir --output /path/to/SUBJECTID_output_dicom --otsu-refine
```

`--input` is a directory of `.dcm` slices for one subject. Slices are ordered by
`InstanceNumber`, falling back to `ImagePositionPatient` projected onto the slice
normal for exports that omit it (ORS Dragonfly, for example).

This writes several sibling directories next to `--output`:

| Path | Contents |
|---|---|
| `<output>/` | union of core and ring, same format as the ground-truth series |
| `<output>_cylinder/` | 10 mm core cylinder only |
| `<output>_ring/` | 18 mm reference ring only |
| `<output>_cylinder_bone/` | core ∩ Otsu bone mask (`--otsu-refine` only) |
| `<output>_ring_bone/` | ring ∩ Otsu bone mask (`--otsu-refine` only) |
| `<output>_axial_view.png` | reslice perpendicular to the fitted axis |

The `_bone` series are what you want for bone-volume-fraction work: the geometric
ROI is intersected with a per-slice Otsu bone map, so only calcified voxels
survive and BV/TV falls out as a voxel ratio against the matching all-tissue
series.

Useful flags: `--threshold` (sigmoid cutoff, default 0.5), `--min-blob-area`
(drops small false-positive blobs, default 200 px), `--min-active-slices` (warns
below 10), `--margin-mm` (crop padding, default 0 to match ground truth), and
`--no-axial-preview` to skip the PNG.

Sanity-check the console output before trusting a run. The reported eigenvalues
should land near the ground-truth template's (5.33, 20.25, 20.25) mm², and the
tilt should be in the 20–25° range. A tilt near 0° or wildly different
eigenvalues means the mask fit failed and the ROI is misplaced.

## Visualization

Open `visualize_output.ipynb`, set `INPUT_DIR` and `OUTPUT_BASE` in Cell 2, and
run all cells. It reports ROI volumes and BV/TV, gives you a slice-by-slice
input-vs-prediction viewer, a grid of active slices, and the axial view.

For the axial view outside the notebook:

```python
from axial_view import AxialView

av = AxialView(INPUT_DIR, OUTPUT_BASE)   # OUTPUT_BASE = the union series
av.summary()                             # centre, axis, tilt, eigenvalues
av.show()                                # 8 mm slab MIP — defect ringed by bone
av.show(t_mm=0.0)                        # single plane through the centre
av.interactive()                         # slider through the slab
```

`AxialView` re-fits the axis from a written output series, so it works on
ground-truth `_output_dicom` volumes too, not just predictions.

## The model

`models/best_model.pth` (89 MB) is included, so inference needs no training step.

- 2D U-Net, features (32, 64, 128, 256), ~7.8M parameters
- **2 input channels**: normalized HU + Otsu bone map, at 256×256
- Trained on all 12 labeled 3-month subjects across the Defect, Defect+PDLLA, MC,
  and MC+PDLLA groups, with positive slices oversampled 5×
- Checkpoint is epoch 95 of 100, selected on training Dice 0.9807

**That 0.9807 is a training-set Dice with no held-out validation split**, so it
measures fit, not generalization. Treat it as a sanity check that optimization
converged, not as an accuracy estimate for new scans. The geometric template is
what makes the output robust — the network only has to find roughly the right
region for the PCA fit to land correctly.

`3_inference.py` reads the input-channel count from the checkpoint itself, so
2-channel and 7-channel weights both load without a flag.

## Retraining

The raw DICOM tree and the derived `data/` directory are **not** in this repo —
they are hundreds of GB of animal-subject scans. You need your own copy of the
study data to retrain, and the manifest in `0_build_manifest.py` expects the
original group/timepoint/subject folder layout.

```bash
python 0_build_manifest.py --base "/path/to/PDLLA RAW DICOM VIVO DATA"
python 1_prepare_dataset.py
python 2_train.py
```

Step 0 writes `data/subjects.csv` and verifies each subject has 1200 slices.
Step 1 reconstructs full-frame 1200×1200 masks from the cropped ground-truth
series using `ImagePositionPatient` math, then saves 512×512 arrays to
`data/prepared/<subject_id>.npz`. Step 2 trains and writes
`models/best_model.pth` plus `logs/train_log.csv`.

Note that `2_train.py` currently defaults to `IN_CHANNELS = 7`, an experimental
2.5D mode that stacks the previous, current, and next slice (HU + Otsu each) plus
a normalized Z-position channel. **The shipped checkpoint is the 2-channel
model.** Set `IN_CHANNELS = 2` to reproduce it; leave it at 7 only if you intend
to train the 2.5D variant from scratch.

## Repository layout

```
0_build_manifest.py      build data/subjects.csv from the raw DICOM tree
1_prepare_dataset.py     pair scans with ground truth → data/prepared/*.npz
2_train.py               train the U-Net
3_inference.py           predict + stamp the ROI template → DICOM series
model.py                 U-Net, Dice / focal / combined losses
axial_view.py            reslice perpendicular to the fitted defect axis
visualize_output.ipynb   inspect outputs, ROI volumes, BV/TV
models/best_model.pth    trained 2-channel checkpoint
```

`data/`, `logs/`, and `models/last_model.pth` are gitignored — the first two are
regenerated by steps 0–2, and the extra checkpoint is redundant.

## Testing this without study data

No sample scan ships with the repo; a single series is 1200 slices. To try the
code end to end you need a rabbit calvarial CT directory of your own. Without
one you can still confirm the install is sound:

```bash
python model.py                  # prints parameter count, runs a forward pass
python 3_inference.py --help
python -c "import axial_view; print(axial_view.AxialView)"
```
