# Rabbit Craniofacial Defect Segmentation

Locates the surgical calvarial defect in rabbit micro-CT scans and stamps the
study's standard ROI template — an 8 mm-tall rigid cylinder set: a 10 mm core
over the defect, plus an 18 mm OD / 14 mm ID reference-bone ring around it.
Outputs are DICOM series in the same format as the hand-labeled ground truth,
plus core and ring BV/TV at a fixed HU threshold and the core-to-ring ratio.

It handles two kinds of scans:

- **In vivo scans** (0.1 mm voxels, whole-head): a U-Net finds the defect
  region and the template is fitted to it. 3-month scans use the network
  directly; any later timepoint places the ROI by registration from the same
  animal's 3-month ROI.
- **Ex vivo specimen scans** (SCANCO µCT, ~15 µm, excised calvaria): the
  network is not used at all — the defect is located geometrically from the
  specimen itself and the same template is stamped.

## Install

```bash
pip install -r requirements.txt
```

Runs on Apple Silicon (MPS), CUDA, or CPU. The trained model
(`models/best_model.pth`) is included — no training step needed.

## The web app — the recommended way to run everything

> New lab members: read **`LAB_GUIDE.md`** — a step-by-step walkthrough.

Double-click **`Launch Defect Segmenter.command`** (or run
`python3 webapp.py --open`) and a browser page opens at
`http://127.0.0.1:8765`. Drop the scan folder in; everything else is
automatic:

- **The scan is recognised from its own DICOM headers.** Scans already in
  the study archive skip the upload entirely. A scan the app does NOT
  recognise brings up a short form first — sample name/ID, treatment group,
  timepoint, scan type, notes — which names the results folder and fills the
  scan's spreadsheet row.
- **The placement mode is chosen for you:**
  - *3-month in vivo* → direct network placement (the training timepoint).
  - *Any later in vivo timepoint* → registration from the animal's 3-month
    ROI, with the reference located automatically.
  - *Ex vivo specimen* (recognised by voxel size, scanner, or location in
    `Ex Vivo CT Data`) → geometric placement, no network.
  - *Unrecognised scan* → network placement, flagged with a caveat.
- **Every run is sanity-checked** with mode-appropriate checks (axis tilt,
  template enclosure, registration dice, defect visibility, ring coverage).
  A run shows exactly one of four statuses: **queued**, **running**,
  **passed**, or **error** — a run whose checks include a failure shows as
  error. The full check details (warnings included) stay on the run page.
- **Results on screen**: ROI volumes, core/ring BV/TV at the chosen
  threshold, the core-to-ring ratio (the number to report), Otsu + radiomic
  features, the axial preview, and a **Download results** zip.
- **The study spreadsheet updates itself**: every run that passes is added
  to the study-wide radiomics database
  (`radiomics_database/radiomics_database.xlsx`) the moment it finishes —
  one row per scan, re-runs and nudges replace the row. Runs that error or
  fail a check are never added.
- **All outputs land in one place**: `outputs/<sample>_<timepoint>_<treatment>/`
  next to this repo (e.g. `outputs/37952_3m_Defect/`), holding the five DICOM
  series, the axial preview and the feature files for that scan.
- **Batches run unattended**: drop several scan folders at once (or a folder
  containing many scans) and each becomes a queued job — jobs run one after
  another, the Mac is kept from idle-sleeping while they run, and still-queued
  jobs survive an app restart. Drop a batch in the evening, read the checks in
  the morning.
- **A re-run replaces its own outputs folder** (and its spreadsheet row) —
  no `_v2` clutter. The hand-labeled ground-truth series in the archive can
  never be overwritten, and nothing under `outputs/` is ever mistaken for a
  raw scan.
- **Adjust placement** overlays draggable rings on a finished run for small
  manual nudges (≤ 6 mm). Applying a nudge updates THAT run in place — the
  series is re-stamped at the new position, features re-extracted, and the
  scan's spreadsheet row refreshed — and the run is permanently flagged as
  readjusted (the replaced pose is kept under `logs/webapp/adjust/`, and
  re-running the scan restores the automatic placement). Avoid nudging
  3-month runs — the automatic fit is validated to ~0.2 mm there.

The **Advanced** panel keeps the fully manual run (explicit mode, reference,
threshold, overwrite). Run history and logs persist in `logs/webapp/`. To
share one instance on the lab network: `python3 webapp.py --host 0.0.0.0`.

## Command line

### In vivo, 3-month scan

```bash
python 3_inference.py --input /path/to/original_dicom_dir \
    --output /path/to/SUBJECTID_output_dicom --otsu-refine
```

`--input` is a directory of `.dcm` slices for one subject. This writes the
output series next to `--output`:

| Path | Contents |
|---|---|
| `<output>/` | union of core and ring (ground-truth format) |
| `<output>_cylinder/` | 10 mm core cylinder only |
| `<output>_ring/` | 18 mm reference ring only |
| `<output>_cylinder_bone/` | core ∩ bone (`HU > --bone-threshold`) |
| `<output>_ring_bone/` | ring ∩ bone |
| `<output>_axial_view.png` | reslice perpendicular to the fitted axis |

Useful flags: `--bone-threshold` (default 226 HU), `--threshold` (sigmoid
cutoff, default 0.5), `--fit-only JSON` (write the fitted pose only, no
series), `--no-axial-preview`.

### In vivo, any later timepoint

**Do not use raw network placement at 6 or 9 months** — the network has a
measured 2–3 mm bias there. Run detection, then place by registration from
the animal's 3-month ROI:

```bash
python 3_inference.py --input 6m_dicom_dir --output SUBJ_6m_output_dicom   # detection hint
python 4_propagate_roi.py \
    --ref-input    3m_dicom_dir  --ref-roi    SUBJ_output_dicom \
    --target-input 6m_dicom_dir  --target-roi SUBJ_6m_output_dicom \
    --output SUBJ_6m_output_dicom --bone-refine
```

The reference ROI is the 3-month ground-truth series when one exists, or a
3-month prediction otherwise. The script refuses to write anything if the
registration dice is below `--dice-min` (default 0.5) — if a scan fails the
gate, prefer re-exporting it at original resolution over lowering the gate.
`--wide-search` adds tilt perturbations to the rotation search;
`--target-fit` accepts the JSON from `3_inference.py --fit-only`.

### Ex vivo specimen scan

```bash
python 5_exvivo_roi.py --input EXVIVO_DICOM_DIR \
    --output 261_5776_exvivo_output_dicom --bone-refine
```

No network involved: the plate normal is fitted by PCA on the bone mask and
the trephine site is found by a matched filter on the plate's HU deficit.
Output series are identical in format to the in vivo ones, written at native
resolution (budget ~10 GB and 10–20 min per scan). Expect the axis tilt near
0° — ex vivo specimens lie flat, unlike in vivo scans where the defect axis
is 74–88° off the slice axis.

Sanity signals printed per run: `defect HU deficit` (below ~200 HU the
defect may be fully bridged — verify placement on the preview), `ring bone
coverage`, and the tilt. `--fit-only JSON` and `--place-threshold` (geometry
only, default 500 HU) are available.

### Extracting Otsu + radiomic features

Runs automatically after every web-app job; for a series produced on the
command line:

```bash
python 6_extract_features.py --input DICOM_DIR --roi OUT_BASE
```

Writes `<roi>_features.csv` / `.json` with ~38 features per region (core and
ring) plus core-to-ring ratios: first-order intensity statistics, Otsu
features (per-region Otsu and 3-class multi-Otsu thresholds, BV/TV at Otsu
and at the fixed study threshold, mean HU of supra-threshold bone — a
tissue-mineral-density proxy), and 3D GLCM texture (contrast, homogeneity,
correlation, entropy, …). Implemented on numpy/scipy/scikit-image with
IBSI-style definitions — pyradiomics does not install on current
Python/NumPy.

Extraction runs at native voxels for in vivo scans and block-averaged
~0.06 mm voxels for ex vivo µCT (recorded in the JSON metadata). Two rules:
compare features only between runs with the same voxel size and placement
method, and treat the per-region Otsu threshold as a reported diagnostic —
in an in vivo core containing air it separates air from tissue (it can land
near −300 HU), not bone from soft tissue, which is why the fixed-threshold
BV/TV remains the headline number.

### Building the study-wide radiomics database

```bash
python 7_build_database.py            # add --no-extract to re-assemble without extracting
```

The spreadsheet is ONE sheet, one row per scan
(`../radiomics_database/radiomics_database.xlsx`, sheets `README`,
`database`, `feature_dictionary`, plus `radiomics_database.csv`). Each row
carries treatment, subject, timepoint, scan type, placement method (with a
`manually_adjusted` flag for `_adj` nudges), the QC outcome (web-app checks
or the 2026-08-11 batch QC), the template pose, and `core_*`, `ring_*`,
`core_to_ring_*` columns for all 113 features. Rows are keyed by scanner
StudyID, so re-adding a scan — from any export folder — replaces its row
instead of duplicating it. The normal path is incremental and automatic: the
web app adds every run that finishes without a failed check the moment it
completes (equivalent to `python 7_build_database.py --add-job JOB_ID`);
errored or check-failed runs are never added, and a manual readjustment
re-adds its run automatically with `manually_adjusted` set. The database
directory can be overridden with the `DEFECT_DB_DIR` environment variable
(default `../radiomics_database`). The full rebuild above instead picks the
best series on disk per scan (ground truth first; a manual `_adj` series
only when nothing else exists) and writes stub rows for scans with no usable
series.

A full rebuild re-scans the whole archive (skipping the central `outputs/`
folder, which holds ROI series, never raw scans); it does not include scans
that were uploaded through the app rather than archived — those enter the
spreadsheet through their automatic web-app runs.

### Stamping at an explicit pose

```bash
python stamp_roi.py --input DICOM_DIR --output OUT_BASE --pose pose.json --bone-refine
```

Writes the same five series at a given `{"center_mm": ..., "axis": ...}` —
this is what the web app's manual adjustment uses.

## Reading the numbers

- **Report the core-to-ring BV/TV ratio**, not absolute BV/TV: the 8 mm
  template is taller than the calvarial plate, so absolute BV/TV is diluted
  by geometry and not comparable to published values. The ratio cancels the
  dilution.
- **Fix one bone threshold for the whole study and state it with every
  number** (default 226 HU, the conventional mineralised-bone threshold).
  The ratio is strongly threshold-dependent — never compare ratios computed
  at different thresholds.
- **Never mix placement methods in one comparison**: network-placed,
  registration-placed, and manually adjusted ROIs are different
  measurements.
- **Never compare ex vivo numbers with in vivo numbers**, even at the same
  threshold — partial-volume behaviour at 15 µm vs 100 µm makes them
  different measurements. Compare ex vivo only with ex vivo.
- Trust a run only if its sanity checks pass (the web app runs them for
  you). For in vivo fits, the axis tilt should land in the 74–88° range; a
  tilt near 0° on an in vivo scan means the mask was a diffuse blob and the
  ROI is misplaced. For ex vivo fits the opposite holds: tilt near 0° is
  expected.

## Visualization

Open `visualize_output.ipynb`, set `INPUT_DIR` and `OUTPUT_BASE` in Cell 2,
and run all cells — ROI volumes, BV/TV, a slice viewer, and the axial view.
Or, outside the notebook:

```python
from axial_view import AxialView

av = AxialView(INPUT_DIR, OUTPUT_BASE)   # OUTPUT_BASE = the union series
av.summary()                             # centre, axis, tilt, eigenvalues
av.show()                                # top-down slab view of the defect
```

`AxialView` works on any written output series, including ground truth, and
automatically decimates ex vivo scans so memory stays manageable.

## Repository layout

```
3_inference.py           in vivo: predict + stamp the ROI template
4_propagate_roi.py       in vivo: place later-timepoint ROIs by registration
5_exvivo_roi.py          ex vivo: place the ROI geometrically (no network)
stamp_roi.py             stamp the template at an explicit pose
7_build_database.py      study-wide radiomics spreadsheet (all scans, core/ring features)
webapp.py + webapp.html  local web app wrapping all of the above
Launch Defect Segmenter.command   double-click launcher for the web app
axial_view.py            reslice perpendicular to the fitted defect axis
visualize_output.ipynb   inspect outputs, ROI volumes, BV/TV
models/best_model.pth    trained 2-channel U-Net checkpoint
0_build_manifest.py, 1_prepare_dataset.py, 2_train.py   retraining pipeline
model.py                 U-Net and losses
```
