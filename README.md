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

The DICOM slice plane is **not** the anatomical axial plane for these scans.
Measured across all 12 ground-truth subjects, the defect cylinder axis sits
**74.5–87.8° off the slice-stacking axis** (mean 81.3° ± 4.2°), so it lies close
to the slice plane itself and varies by ~13° between subjects. Scrolling the raw
slices gives you a *coronal* cut through the defect, not a top-down view.

`3_inference.py` recovers the true axis by PCA on the raw predicted mask: because
the network was trained on these oblique cylinders, the smallest-variance
eigenvector of its output is the cylinder axis. Validated against ground truth,
the smallest eigenvalue is 5.47 mm² predicted vs 5.33 mm² measured, where the
expected value for an 8 mm slab is h²/12 = 5.33.

Stamping a Z-aligned cylinder instead — as any pre-August-2026 output did — tilts
the ROI by that full ~81° and makes the resulting volume and BV/TV numbers
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

## Lab web app — the easy way to run this

> New lab members: read **`LAB_GUIDE.md`** — a step-by-step walkthrough of the app.

For routine use, nobody needs the command line. Double-click
**`Launch Defect Segmenter.command`** (or run `python3 webapp.py --open`) and a
browser page opens at `http://127.0.0.1:8765` with one control: **drop the
scan folder in**. Everything else is automatic:

- **The scan is recognised from its own DICOM headers.** At startup the app
  indexes every raw scan in the archive by scanner StudyID (`t57860` …). One
  uploaded slice is enough to identify the animal, group, and timepoint — and
  a scan already in the archive skips the multi-GB upload entirely and reuses
  the copy on the volume. Unknown scans upload fully, then run.
- **Placement mode is chosen for you.** 3-month scans run direct network
  placement; any other recognised timepoint runs registration from the
  animal's 3-month ROI (`4_propagate_roi.py`), with the reference scan and
  ground-truth ROI located automatically. Unrecognised scans fall back to
  network placement and the results are flagged with the placement caveat.
- **Nothing is destroyed by a re-run**: outputs auto-version (`…_v2`, `…_v3`),
  and hand-labeled ground-truth series (`<id>_output_dicom`) can never be
  overwritten — the app refuses that output name outright.
- **Every run is sanity-checked automatically** against the ground-truth
  envelope from this README: axis tilt 74–88°, eigenvalues near
  (5.33, 20.98, 20.99) mm², full template enclosure, registration dice. Green
  means trust it; red means discard it.
- **Results on screen**: ROI volumes, core/ring BV/TV at the chosen threshold,
  the core-to-ring ratio (the number to report), the axial preview, and a
  **Download results** zip of every written series.
- An **Advanced** panel keeps the fully manual run (explicit placement mode,
  reference, threshold, overwrite) for the pipeline maintainer.

Run history and logs persist in `logs/webapp/`. To let other lab machines use
one shared instance, start it with `python3 webapp.py --host 0.0.0.0` and open
`http://<that-machine>:8765` — keep that to the lab network.

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

The `_bone` series intersect the geometric ROI with `HU > --bone-threshold`
(default **226 HU**), giving BV/TV as a voxel ratio against the matching
all-tissue series.

**Read the two caveats below before quoting any BV/TV number.**

#### Why a fixed HU threshold

226 HU is the conventional mineralised-bone threshold in CT bone morphometry (the
226 mg HA/cm³ value in the ASBMR / Bouxsein et al. 2010 µCT guidelines). There is
no universal "bone threshold" — what those guidelines actually require is that you
fix one for the study and report it. Override with `--bone-threshold`.

A fixed value is valid here because HU is calibrated consistently: air measures
**−1002 HU** in every scan checked across the 3-, 6- and 9-month timepoints,
despite rescale slopes varying from 0.117 to 0.161.

Earlier versions ran Otsu over the whole 1200×1200 slice instead. That slice is
overwhelmingly air, so the threshold landed near **−150 HU** and separated
specimen from air rather than bone from soft tissue — it reported ~81% "bone"
inside an unbridged defect and made core and ring look identical (1.03×). Otsu is
still fine as a *network input channel*, where it is just a feature; it was only
the BV/TV use that was wrong.

#### Absolute BV/TV is geometrically diluted — use the ratio

The ROI is 8 mm tall along its axis, but the rabbit calvarial plate is much
thinner. Profiling bone fraction along the axis, the plate occupies only the
middle **3.5–4 mm**; the rest of the cylinder extends into soft tissue on both
sides. Within the plate the reference ring is nearly solid bone (peaking at 84–93%),
but averaged over the full 8 mm, dilution alone caps ring BV/TV near 44–50%.

So absolute BV/TV over this ROI is dominated by the height of the template, not by
bone quality, and **is not comparable to published BV/TV values**. The
**core-to-ring ratio** is the meaningful measure, because the dilution cancels.
Report that, and state the threshold and ROI height alongside it.

Measured at 226 HU on the four registration-placed 6-month outputs, the reference
ring lands at 39.9–43.4% — at the ceiling dilution predicts. So the ring is
essentially fully mineralised within the plate, which is what makes it a usable
reference.

#### The threshold changes the biological conclusion

The core-to-ring ratio is strongly threshold-dependent, because the defect contains
low-density woven bone that a high threshold excludes and a low one counts.
Registration-placed 6-month outputs (three untreated Defect animals + one
Defect+PDLLA):

| threshold | 37951 | 37952 | 41122 | 38743 (PDLLA) |
|---|---|---|---|---|
| 226 HU | 0.79× | 0.74× | 0.83× | 0.89× |
| 500 HU | 0.31× | 0.33× | 0.42× | 0.49× |

At 226 HU the defects read as 74–89% as mineralised as intact bone; at 500 HU as
under half. Both are defensible and they answer different questions — 226 HU asks
"how much mineralised tissue is there," a higher threshold asks "how much *mature*
bone is there." Pick deliberately with your supervisor, fix it for the whole
study, and state it with every number. Do not compare ratios computed at different
thresholds — and do not compare numbers from network-placed and
registration-placed ROIs, either.

Useful flags: `--threshold` (sigmoid cutoff, default 0.5), `--min-blob-area`
(drops small false-positive blobs, default 200 px), `--min-active-slices` (warns
below 10), `--margin-mm` (crop padding, default 0 to match ground truth), and
`--no-axial-preview` to skip the PNG.

Sanity-check the console output before trusting a run. All 12 ground-truth volumes
fit eigenvalues of (5.33, 20.98, 20.99) mm² and a tilt of 74–88°, so a run should
land close to that. A tilt near 0°, or eigenvalues an order of magnitude larger,
means the mask was a diffuse blob rather than a defect and the ROI is misplaced —
discard it rather than measuring it.

### Accuracy against ground truth — depends on the timepoint

**On 3-month scans** (the training timepoint), measured on all 12 labeled
subjects, comparing the PCA fit of the *prediction* against the PCA fit of the
*ground-truth mask* in the same full-frame coordinates:

| | mean | median | worst |
|---|---|---|---|
| axis error | **1.42°** | 1.32° | 3.28° |
| centre error | **0.21 mm** | 0.17 mm | 0.46 mm |

These are training-set numbers — there is no validation split — so treat them as
an upper bound on accuracy, not a held-out estimate.

**On later timepoints the network's placement is measurably biased.** Rigidly
registering the known 3-month trephine sites onto the same animals' 6-month scans
(skull-local registration, bone-mask dice 0.72–0.75) puts the network's 6-month
ROI **2.2–3.2 mm off the true site** for untreated defects, displaced toward the
ingrown bone edge — part-healed defects are outside its training distribution.
Corrected for that placement error, the residual void in untreated defects sits
0.7–1.2 mm from the trephine site: untreated holes do not migrate; the apparent
"void offset" was mostly the network's error. (A PDLLA-treated animal placed to
1.31 mm, with a genuinely off-centre residual void of 3.2 mm — asymmetric
healing does exist; the point is you cannot tell which case you have from one
image alone.)

**Consequence: do not use raw network placement for any timepoint except
3 months.** Run `3_inference.py` for detection, then fix the placement with
`4_propagate_roi.py` (next section).

One counterintuitive detail worth recording, because it misleads: **the raw
predicted mask is always strongly elongated.** Its transverse eigenvalue ratio is
0.35–0.41 across all 12 subjects, against 1.0 for the radially symmetric template,
and the network under-segments by ~23%. Yet the centroid is still accurate to
0.21 mm. Elongation is a second moment; the centroid is a first moment. A lopsided
mask is *not* evidence of a displaced centre, and the rigid template only needs the
network to find roughly the right region for the fit to land.

### An approach that was tried and rejected

An image-driven re-centring pass — slide the template to maximise `mean HU in the
reference ring − mean HU in the core`, a matched filter for "dark core inside
bright bone" — looked convincing on proxies: it improved contrast by 40–65 HU and
landed within ~1 mm of a manual annotation read off screenshots.

Ground truth rejected it. It moved the centre error from **0.21 mm to 0.73 mm** and
improved centring on only **1 of 12** subjects. The same contrast metric also asks
for a 5–11° axis rotation that GT shows would be wrong, so it is biased as an
objective for both centre and orientation — plausibly because tilting or shifting
seats the 8 mm ring better against a domed skull, raising ring HU without being
anatomically right.

Do not reintroduce it without held-out ground truth to validate against — and
note the postscript below, because this story had one more turn.

**Postscript — the visual observation was right after all.** The re-centring was
rejected on 3-month ground truth, where the network's placement is already
accurate and shifting onto the residual void degrades it. But the observation
that prompted it — the core visibly off the defect on *6-month* scans — was later
confirmed by cross-timepoint registration to be a real ~2–3 mm network placement
error at that timepoint (see the accuracy section above). On the untreated
6-month scans the contrast shift was pointing approximately at the true trephine
site. It stays removed anyway: it corrects the wrong thing on 3-month scans,
and on treated animals with genuinely asymmetric healing (38743) it would drag a
correctly-placed ROI onto the void. Registration (`4_propagate_roi.py`) fixes
placement using anatomy instead of a heuristic, and works in both cases.

## Cross-timepoint placement — `4_propagate_roi.py`

For any scan that is not a 3-month scan, place the ROI by registration from the
animal's earlier trusted ROI instead of trusting the network:

```bash
python 3_inference.py --input 6m_dicom_dir --output SUBJ_6m_output_dicom   # detection + hint
python 4_propagate_roi.py \
    --ref-input    3m_dicom_dir  --ref-roi    SUBJ_output_dicom \
    --target-input 6m_dicom_dir  --target-roi SUBJ_6m_output_dicom \
    --output SUBJ_6m_output_dicom --bone-refine
```

The reference ROI is the 3-month ground-truth series when one exists, or a
3-month *prediction* otherwise (that timepoint is in-distribution for the
network). Registration is skull-local — a 48 mm box around the defect, largest
connected bone component, signed distance maps — because the full field of view
contains paws, a positioning tube and a dish that stay put while the head moves,
and whole-volume registration locks onto the hardware. Rotation is initialised
from the two defect axes with a 12-way spin multi-start.

It refuses to write anything if the bone-mask dice after registration is below
`--dice-min` (default 0.5), and prints how far the network's placement was from
the registered site. Output series are identical in format to `3_inference.py`'s.

`--wide-search` extends the rotation multi-start with tilt perturbations (±12°,
±24°) for cases where the network hint's axis is unreliable. `--target-fit`
accepts the JSON from `3_inference.py --fit-only` so the network's throwaway
series never have to be written at all.

Measured on the full 42-scan study: 41 of 42 scans placed automatically (dice
0.59–0.75). The one failure — a scan re-exported at 0.2 mm instead of 0.1 mm,
with no InstanceNumber, a large head-pose change, and a healed defect — was
correctly *refused* by the dice gate rather than silently mis-measured, across
four escalating configurations. If a scan fails the gate, prefer re-exporting it
at original resolution over lowering `--dice-min`.

### Throughput

One scan takes ~4–5 minutes unattended (Apple M-series): ~1–2 min for the
prediction pass (`--fit-only --fast-fit`, batched GPU inference with threaded
DICOM reads) plus ~2–3 min for registration and writing all five series in a
single pass over the source slices. The naive per-slice, per-series pipeline was
~3× slower; the network itself was never the bottleneck — DICOM I/O and full-
resolution mask post-processing were.

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
4_propagate_roi.py       place later-timepoint ROIs by registration (use it!)
webapp.py + webapp.html  local web app wrapping 3 & 4 for lab use
Launch Defect Segmenter.command   double-click launcher for the web app
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
