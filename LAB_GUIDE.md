# Defect Segmenter — Lab User Guide

Segment a rabbit calvarial defect scan and get its bone-healing numbers in
three steps. No programming needed. The whole workflow is:

> **Drop the scan in → (fill in its details if the app doesn't know it) →
> results appear and the study spreadsheet updates itself → nudge the
> placement if needed (the run updates in place) → done.**

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
  - *Scan it doesn't recognise* → a short form pops up first: give the scan
    a **sample name/ID** (required), its treatment group, timepoint and scan
    type, plus optional notes. The app pre-fills what it can read from the
    scan itself. Then the whole scan uploads (a few GB — give it a few
    minutes) and placement runs; what you entered names the results folder
    and fills the scan's spreadsheet row.
  - *Ex vivo specimen scan* (excised skull piece on the SCANCO µCT) → the AI
    isn't used at all: the defect is already the centre of the specimen, so
    the app finds it geometrically from the bone itself and stamps the same
    10/18 mm template. Ex vivo numbers are measured at ~15 µm — compare them
    only with other ex vivo numbers, never with in vivo ones.
- Every run's files land in one place: the **`outputs`** folder next to the
  group folders, one subfolder per scan named
  `outputs/<sample>_<timepoint>_<treatment>/` (e.g. `outputs/37952_3m_Defect/`).
  Re-running a scan simply replaces that folder's contents — no `_v2` copies
  pile up — and the hand-labeled ground-truth folders can't be overwritten
  at all.

You can drop the whole subject folder if that's easier — the app finds the
raw scan inside it. You can also drop **several scan folders at once** (or a
folder containing many scans): each becomes its own queued job.

## Running a batch overnight

Drop everything you want processed — the jobs queue up and run **one after
another automatically**. Once the upload bar finishes you can close the
browser; the jobs run on the server. On a Mac the app keeps the machine from
idle-sleeping while a job is running, and if the app is restarted, jobs that
were still waiting in the queue pick up where they left off (a job that was
interrupted *mid-run* shows as **error** instead — re-run that one yourself). Come
back in the morning and read each run's checks as usual. Two practical notes:
don't let the machine run out of disk (an ex vivo run writes ~10 GB), and
keep laptops plugged in — macOS only honours the keep-awake on AC power.

## 3. Read the results

Click the run in the **Runs** list (it updates live; a 3-month scan takes a
few minutes, a registration run longer). Every run shows one of four badges:

- **queued** / **running** — waiting its turn, or working.
- **passed** — finished and usable (already added to the spreadsheet).
- **error** — do **not** use the numbers: the run crashed, was cancelled, or
  a sanity check failed. Open the run to see why.

**a. Sanity checks — every line should be a green ✓.** (They're inside the
run page; the badge only summarises them.)
- ⚠ yellow: the run still shows **passed**, but read the warning and look at
  the preview picture before using the numbers.
- ✕ red: the run shows **error** and is NOT added to the spreadsheet.
  Usually the model latched onto the wrong thing — re-run or ask for help.

**b. The number to record** is the big blue **Core : ring BV/TV ratio** —
how mineralised the defect is relative to the intact bone around it
(1.0× ≈ healed to normal density). Always note the **threshold (226 HU)**
next to it. Ignore the absolute BV/TV percentages for publications — the 8 mm
ROI height dilutes them.

**c. The preview picture.** The dark defect should sit inside the cyan
**10 mm core** circle, with the yellow/green **reference ring** on solid
bright bone. Circles obviously off the defect → don't trust the run.

**c2. The feature table.** Below the BV/TV numbers each run shows Otsu &
radiomic features for the core and ring (mean HU, bone density, texture…),
with the full ~38-feature set saved as `…_features.csv` next to the series
and inside the results zip. Only compare features between runs of the same
scan type (in vivo with in vivo, ex vivo with ex vivo).

**c3. The study spreadsheet updates itself.** Every run that finishes as
**passed** is added automatically to
`radiomics_database/radiomics_database.xlsx` (next to the group folders):
treatment, subject, timepoint, placement method, checks, and every core /
ring / core-to-ring feature. The run page shows **"✓ in spreadsheet"** with
the row it became. The spreadsheet has **one `database` sheet with one row
per scan** — a re-run or a manual nudge replaces that scan's row, so the
sheet never collects duplicates. Runs that end in **error** are never added.

**d. Where the files went.** Everything is in the central **`outputs`**
folder, one subfolder per scan
(`outputs/<sample>_<timepoint>_<treatment>/`), holding the five DICOM series,
the preview picture and the feature files. To take results elsewhere, click
**Download results** for a zip.

## Nudging the placement by hand (use sparingly)

If a finished run's circles look slightly off the defect, open the run and
click **Adjust placement**. After the reslice loads, drag the rings so the
core sits on the defect — the offset shows live in mm — and click
**Apply — write adjusted series**.

Three things to know before you use it:

- The nudge **updates that run in place**: the same files in the same
  `outputs` folder are re-stamped at the new position, the features are
  re-extracted, and the scan's spreadsheet row is refreshed. The run gets a
  permanent **"readjusted"** badge and its row is flagged
  `manually_adjusted` — never mix manually placed and automatically placed
  ROIs in the same comparison. (The position it replaced is kept by the app,
  and simply re-running the scan restores the automatic placement.)
- **On 3-month scans, don't.** The automatic fit matches the annotation
  protocol to ~0.21 mm; in a validated test, placements that "looked more
  centred" were actually *worse* on 11 of 12 subjects. The app will remind
  you.
- A nudge is for **small** corrections (a millimetre or two). If the rings are
  far from the defect, the detection failed — re-run instead, and the app
  blocks nudges beyond 6 mm for exactly that reason.

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
