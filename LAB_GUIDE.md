# Defect Segmenter — Lab User Guide

A step-by-step guide for segmenting a rabbit calvarial defect scan and getting
its bone-healing numbers. No programming needed.

---

## 1. Start the app

Double-click **`Launch Defect Segmenter.command`** in the `defect_segmentation`
folder. A terminal window opens, then your browser opens the app at
`http://127.0.0.1:8765`.

Leave the terminal window open while you work — closing it stops the app.
(If someone runs the app on a shared machine, you can instead just open
`http://<that-machine's-address>:8765` in your own browser.)

## 2. Pick the scan you want to segment

1. Under **Scan to segment**, click **Browse**.
2. Navigate to your animal: *group* → *timepoint* → *subject*, e.g.
   `Defect → 6 MONTH → 37951`.
3. Select the folder holding the raw scan — it's the one named like
   **`dicom_t58524`** with a blue **1200 .dcm** badge. Double-click it, or open
   it and press **Select this folder**.

The app then fills in everything it can figure out on its own. Check the little
grey chips under the box: they should show the right **subject**, **group**,
**timepoint**, and about **1200 DICOM slices**.

## 3. Check the placement mode (usually already correct)

The app picks this for you from the timepoint — you normally don't touch it:

| Your scan is… | Mode the app picks | What you need to do |
|---|---|---|
| **3-month** | *3-month scan — direct network placement* | Nothing. |
| **6- or 9-month** | *Later timepoint — registration from the 3-month ROI* | Check the two **Reference** boxes were auto-filled (they point at the same animal's 3-month scan and its ROI). |
| Later timepoint, but **no 3-month scan exists** for this animal | You must select *raw network placement* yourself | The app will warn you: this placement can be 2–3 mm off. Talk to your supervisor before using these numbers. |

Why this matters: the AI model was trained on 3-month scans only. On later
scans it finds the defect but places the measurement cylinder a few mm off, so
the app instead lines up the later scan with the animal's 3-month scan and
carries the trusted ROI position across.

## 4. Leave the settings alone (mostly)

- **Bone threshold** stays at the study value (**226 HU**) unless your
  supervisor tells you otherwise. Numbers computed at different thresholds
  cannot be compared with each other.
- **Output series name** is suggested automatically and follows the existing
  naming convention — leave it.
- Tick **Overwrite existing** only if you are deliberately redoing a previous
  run of the same scan.
- The app will refuse to touch the hand-labeled ground-truth folders
  (`37951_output_dicom` etc.) — you can't break those.

## 5. Run it

Click **Run segmentation**. The run appears in the list on the right with a
live console log.

- A 3-month scan takes a few minutes.
- A registration run (6/9-month) takes longer — let it finish.
- You can queue more runs while one is going, and **Cancel** a running one.
- You can close the browser tab and come back later; the run keeps going and
  the history is saved.

## 6. Read the results

When the run finishes, click it in the list. Read top to bottom:

**a. Sanity checks.** Every line should have a green ✓.
- **⚠ yellow warning** — the run finished, but look at the axial preview
  carefully before using the numbers (and note what the warning says).
- **✕ red fail** — do **not** use the numbers. The most common cause is the
  model latching onto the wrong thing; re-run, or ask for help.

**b. The numbers.** The big blue box, **Core : ring BV/TV ratio**, is the
number the study reports — how mineralised the defect is compared to the intact
bone around it (1.0× ≈ healed as dense as normal bone). When you record it,
always note the **threshold (226 HU)** alongside. Ignore the absolute BV/TV
percentages for publications — the 8 mm ROI height dilutes them, and they are
not comparable to published values.

**c. The axial preview picture.** One glance tells you the run is sane: you
should see the dark defect sitting inside the cyan **10 mm core** circle, with
the yellow/green **reference ring** lying on solid bright bone around it. If
the circles are obviously not centred on the defect, don't trust the run.

## 7. Where the output files went

Everything is written next to the scan you selected, e.g. in
`Defect/6 MONTH/37951/`:

| Folder / file | What it is |
|---|---|
| `37951_6m_output_dicom` | the full ROI (core + ring) as a DICOM series |
| `..._cylinder`, `..._ring` | core-only and ring-only series |
| `..._cylinder_bone`, `..._ring_bone` | the same, restricted to bone above the threshold |
| `..._axial_view.png` | the preview picture |

These drop straight into the existing analysis workflow (Dragonfly, the
notebook, etc.).

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Browser says "can't connect" | The app isn't running — double-click the launcher again. |
| "output series already exists" | A previous run used this name. Tick **Overwrite existing** if you mean to redo it. |
| "No 3-month reference found automatically" | Use the **Browse** buttons to point at the animal's 3-month `dicom_t…` folder and its `<subject>_output_dicom` folder yourself. If the animal truly has no 3-month scan, see the raw-placement row in step 3. |
| Registration run says **dice** failed / too low | The two scans couldn't be aligned confidently, so nothing was written. Re-run with **Wide rotation search** ticked. Still failing → ask for help. |
| Axis tilt check is red / preview circles miss the defect | The model mis-detected. Discard the run and ask for help — do not record its numbers. |
| The run just errored (red "error") | Open **Console log**, scroll to the bottom, and send the last lines to whoever maintains the pipeline. |

**Two rules worth repeating:** never compare ratios computed at different
thresholds, and never mix numbers from raw-network-placed and
registration-placed ROIs in the same comparison.
