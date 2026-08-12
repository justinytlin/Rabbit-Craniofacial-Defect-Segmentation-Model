# Defect Segmenter — Lab User Guide

Segment a rabbit calvarial defect scan and get its bone-healing numbers in
three steps. No programming needed.

---

## 1. Start the app

Double-click **`Launch Defect Segmenter.command`** in the `defect_segmentation`
folder. A terminal window opens, then your browser opens the app at
`http://127.0.0.1:8765`. Leave the terminal window open while you work.

(If the app runs on a shared lab machine, just open
`http://<that-machine's-address>:8765` in your own browser instead.)

## 2. Drop the scan in

Drag the folder of `.dcm` slices onto the big **"Drop a scan folder here"**
box (or click the box and choose the folder). That's it — the app does the
rest on its own:

- It reads the first slice and **recognises the scan** (animal, group,
  timepoint) from the scanner's study ID. Scans already in the study archive
  don't even need to upload — segmentation starts within seconds.
- It **picks the right placement method automatically**:
  - *3-month scan* → direct AI placement (the model's training timepoint).
  - *6- or 9-month scan* → the ROI is placed by registration from that same
    animal's 3-month ROI, which the app locates by itself. This is the
    validated method — the AI alone is 2–3 mm off at later timepoints.
  - *Scan it doesn't recognise* → the whole scan uploads (a few GB — give it
    a few minutes), then AI placement runs and the result is clearly flagged
    so you know the timepoint rules above couldn't be checked.
- Re-running a scan never destroys anything: results get a `_v2`, `_v3`…
  name, and the hand-labeled ground-truth folders can't be overwritten at all.

You can drop the whole subject folder if that's easier — the app finds the
raw scan inside it. You can queue several scans; they run one after another.

## 3. Read the results

Click the run in the **Runs** list (it updates live; a 3-month scan takes a
few minutes, a registration run longer).

**a. Sanity checks — every line should be a green ✓.**
- ⚠ yellow: finished, but read the warning and look at the preview picture
  before using the numbers.
- ✕ red: do **not** use the numbers. Usually the model latched onto the wrong
  thing — re-run or ask for help.

**b. The number to record** is the big blue **Core : ring BV/TV ratio** —
how mineralised the defect is relative to the intact bone around it
(1.0× ≈ healed to normal density). Always note the **threshold (226 HU)**
next to it. Ignore the absolute BV/TV percentages for publications — the 8 mm
ROI height dilutes them.

**c. The preview picture.** The dark defect should sit inside the cyan
**10 mm core** circle, with the yellow/green **reference ring** on solid
bright bone. Circles obviously off the defect → don't trust the run.

**d. Where the files went.** For archive scans, the DICOM series are written
next to the scan (e.g. `37951_6m_output_dicom…`) ready for the existing
analysis workflow. For uploaded scans — or to take results elsewhere — click
**Download results** for a zip.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Browser says "can't connect" | The app isn't running — double-click the launcher again. |
| "No .dcm files found in that folder" | You dropped the wrong folder — use the one full of `.dcm` slices (or the subject folder containing it). |
| Scan not recognised but it *is* a study scan | The scan index may still be building (first minute after launch). Wait a moment and drop it again. |
| Registration says **dice** failed / too low | The two timepoints couldn't be aligned confidently, so nothing was written. Open **Advanced**, re-run with **Wide rotation search** ticked. Still failing → ask for help. |
| Axis tilt check red / circles miss the defect | The model mis-detected. Discard the run and ask for help — do not record its numbers. |
| A run errored (red "error") | Open **Console log**, scroll to the bottom, send the last lines to whoever maintains the pipeline. |

The **Advanced** panel on the front page still allows a fully manual run
(choose the placement mode, reference scan, bone threshold) — normally only
the pipeline maintainer needs it.

**Two rules worth repeating:** never compare ratios computed at different
thresholds, and never mix numbers from AI-placed and registration-placed ROIs
in the same comparison.
