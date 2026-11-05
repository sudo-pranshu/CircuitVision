# Experiments

## Status

Every result carries one of three labels: **REAL CGHD**, **SYNTHETIC/CONTROLLED** or **UNTESTED**.

| Result | Label | Evidence |
|---|---|---|
| Real CGHD annotation format (rotation nested in `<bndbox>`, 1-degree values) | REAL CGHD (one file) | `tests/data/cghd_real/` (from the Hugging Face copy, not the MD5-checked Zenodo zip) |
| Wire tracing on synthetic photos with known graphs: 0.994 exact over 11 conditions | SYNTHETIC/CONTROLLED | `results/synthetic_topology.json` |
| GT-box pipeline on CGHD-shaped synthetic data: 18/18 topology, exact and operating-point match | SYNTHETIC/CONTROLLED | Run log below; reproducible with `synth --write-cghd` |
| EasyOCR on synthetic value labels: 0.914 exact, 0.950 character accuracy | SYNTHETIC/CONTROLLED | Same run |
| Text → netlist: 30-sentence set 0.80 → 1.00 exact (optimistic); fresh 20-sentence set 0.90; Pawan's 20-sentence set (T-7) 0.75. Re-run after the 2026-11-04 fixes: fresh 1.00 (optimistic), others unchanged | CONTROLLED (hand-written) | `results/nlp_eval*.json`, `results/nlp_rerun_2026-11-04/` |
| `train.py`, `detector_eval.py`, `evaluate.py --weights --ocr`, `orientation.py`, `report_tables.py` | Executed end to end on CPU (torch 2.14, ultralytics 8.4.184) | Synthetic data only; numbers carry no meaning about real photos |
| CGHD audit, rotation convention (`PLUS_AT_0=left`, `CGHD_CLOCKWISE=False`), label overlays | REAL CGHD (dataset checks, no model metric) | Run locally on the MD5-checked v14 zip, 2026-11-02; Gates 1 and 2 approved by Pawan. See the findings below |
| YOLOv8s detection on test drafter 12: mAP50 0.904, mAP50-95 0.611 | REAL CGHD | `results/baseline_v1/test/metrics.json`; see Results below |
| Downstream on real photos: 37.5% (GT boxes) and 10.4% (detector) of 48 analog photos match the reference topology | REAL CGHD | `results/baseline_v1/downstream/`; see Results below |

`baseline_v1` ran on Colab on 2026-11-04; its REAL CGHD results are in [Results: baseline_v1](#results-baseline_v1). Training needs a GPU: at the fixed settings, a 16 GB Apple M5 took about 132 s per batch (about 29 days for 100 epochs).

## Findings from real CGHD annotations (before any training)

These were checked against real annotation files (`drafter_13/C145_D1_P1.xml`, `drafter_1/C1_D1_P1.xml`) and the CGHD paper. They changed the code.

- **Rotation is nested inside `<bndbox>`**, with 1-degree resolution (for example 2, 8, 90, 269). The parser read it from `<object>`, so every rotation would have been missing. Fixed in `parse_voc`; regression test `test_real_xml_rotation_is_nested_in_bndbox`.
- **What 0° means can be partly inferred.** A wide resistor box has `rotation=2` and a tall one `90`, so 0° is horizontal for resistors. The audit now prints, for every class, how box shape is distributed in each rotation bin. Clockwise vs counter-clockwise, and which side is + at 0°, can't be read from boxes. Those remain Gate 1 decisions.
- **Mirror annotations exist** (1,437 in the latest release). Their effect on polarity depends on undocumented symbol templates, so mirrored polarized symbols are excluded from orientation training and scoring, and counted.
- **CGHD has many out-of-scope symbols** (transistors, ICs, lamps, switches). They're now masked during wire tracing so their ink can't short two wires together, and every downstream metric is also reported on the in-scope-only subset.
- **The CGHD paper uses a different split:** train on drafters 1–20 and 25, validate on 21–22, test on 23–24. This baseline keeps the split fixed earlier (validation 11, test 12), so its numbers aren't directly comparable with the paper's.
- **A newer release exists** (Zenodo v16, 2025, 4.9 GB). The notebook pins v14 by MD5 for reproducibility.

### Full dataset checks (2026-11-02, MD5-checked v14)

- **Rotation convention (Gate 1).** Diode, LED and Zener sheets show the anode on the left at 0° and the symbol turning counter-clockwise; the dataset's `classes_ports.json` and `utils.transform` agree for diodes. Battery sheets fit the same convention. The ports file doesn't match the sheets for voltage sources, so the sheets were used.
- **`voltage.dc` is an exception.** All 182 rotated crops are labelled 0° with + on top. `parse_voc` now adds 270° for this label; the raw value stays available to the audit. None are in test drafter 12, so this changes only the training crops. Orientation accuracy on drafter 12 voltage sources (32 real crops): 0.875 before, 0.906 with `voltage.dc` excluded, 0.938 corrected. Diodes: 0.950 in all three. **These test crops were seen while choosing the fix, and this was a local run, not baseline_v1.**
- **Placeholder XML sizes (Gate 2).** In 336 training images from drafters -1, 29 and 30, `<size>` says 1000x1000 while the boxes are in displayed-image pixels. The first conversion dropped 5,321 boxes and misplaced others. `annotation_size` now uses the real size: 1 box dropped, 5 clipped, 210,782 training objects (was 205,462). Validation and test labels are unchanged. Regression tests: `test_converter_ignores_placeholder_xml_size`, `test_placeholder_xml_size_uses_the_displayed_frame`.
- **The test split has no `gnd` symbols** and only 32 voltage sources and 40 diodes, so ground-symbol and polarity results on drafter 12 rest on few or no examples.

## Environment

Colab preinstalls several OpenCV distributions (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`), and they all write to the same `site-packages/cv2/` folder. Installing ultralytics and easyocr makes pip replace or remove some of them. That can leave `cv2/` hollow: `import cv2` still works, but its attributes are gone. The first run of baseline_v1 failed this way (`module 'cv2' has no attribute 'HOGDescriptor'`).

`scripts/pin_opencv.sh` fixes it:
1. It uninstalls every OpenCV variant.
2. It deletes any leftover `cv2/` folder.
3. It installs exactly `opencv-python-headless==4.10.0.84` without dependencies.
4. It verifies that `HOGDescriptor` and `ml.SVM` work.

This must be the last package step. `pip check` will then report that ultralytics wants `opencv-python`; that's expected, because both provide the same `cv2` module. `scripts/check_opencv.py` re-verifies before training and before each evaluation step, and `pip_freeze.txt` records the final environment.

### Dataset location and audit speed

The CGHD zip is downloaded once and cached on Drive (`MyDrive/CircuitVision/data/cghd-zenodo-14.zip`). Each run:
1. Copies the zip to local disk.
2. Checks it against Zenodo's MD5.
3. Extracts it to `/content/cghd`.

Every dataset step reads from local disk; only results are written to Drive.

The audit's image check used to decode every full-size photo twice. It now works like this:
- **Dimensions:** for JPEGs, the raw and EXIF-applied sizes come from the file header. `tests/test_audit_images.py` shows these match two full `cv2.imread` decodes for all 8 EXIF orientations, for truncated files and for unreadable files.
- **Readability:** one real decode at 1/8 scale.
- **Everything else:** other formats, and any image where the header path fails, fall back to the two full decodes.

The image pass runs on all CPUs and prints `Audited N/total images` every 100 images. Every image is still checked. On a 2-vCPU machine with 6 MB phone photos, it went from 370 to 36 ms per image.

## Human verification gates

`notebooks/train_colab.ipynb` contains two gate cells whose `assert` stops execution, including under "Run all", until you approve them.

- **Gate 1, after the audit and rotation sheets:** set `GATE1_APPROVED`, plus `PLUS_AT_0` and `CGHD_CLOCKWISE` as read from the real sheets. The cell exports them as `CIRCUITVISION_PLUS_AT_0` and `CIRCUITVISION_CGHD_CLOCKWISE`, and every later step records which convention it used and its source.
  - Decided on 2026-11-02 from the real sheets (`PLUS_AT_0='left'`, `CGHD_CLOCKWISE=False`) and reviewed on 2026-11-04 (issue #7). These are now the code defaults, and the gate cell is prefilled with them; a person still has to check the sheets and set `GATE1_APPROVED`.
  - If the sheets are inconclusive, set the convention to `None`. Orientation is then skipped and reported as UNTESTED.
- **Gate 2, after the label overlays:** set `GATE2_APPROVED` only if the boxes visibly sit on their symbols.

Both approvals are saved, with your notes, to `gates/gate1.json` and `gates/gate2.json`. After a runtime restart the gate cells load these files instead of asking again; delete a file to re-decide that gate. If the label conversion differs in any way from the one approved at Gate 2, the conversion cell stops.

The Gate 2 sample (24 overlays) always includes: the photos whose EXIF rotation was baked in during conversion (up to a quarter of the sample), at least two images from each split, and every class, rare classes first.

## Restarts

The notebook is meant to be re-run with **Run all** after any disconnect. The first cell prints which stages exist on Drive; finished stages (audit, rotation sheets, detector test, orientation, downstream) are skipped unless named in `REDO`. Training uses `train.py --auto`, which reads `runs/train/weights/last.pt`:
- an interrupted run resumes;
- a finished run (ultralytics stores `epoch = -1` with no optimizer) is only recorded;
- no checkpoint means a fresh start.

The kill-and-resume path was tested with real ultralytics 8.4.184 on CPU (3-epoch quick run, killed after epoch 1, resumed at epoch 2), and both notebook sessions (fresh, then restarted) were run end to end on the fake CGHD fixture.

## baseline_v1: how to reproduce

The whole experiment is `notebooks/train_colab.ipynb`, run on a Colab T4 GPU. The same steps work from a shell:

```bash
pip install -e ".[detect]"            # plus: apt install ngspice  (or brew install ngspice)
bash scripts/pin_opencv.sh            # LAST package step: exactly one OpenCV (headless 4.10.0.84)
python -m pytest -q tests
EXP=experiments/baseline_v1

# 0. Data: Zenodo 14042961, cghd-zenodo-14.zip, md5 ab1cde6feb5edaafbde711cd2059a2f4
unzip cghd-zenodo-14.zip -d data/cghd_raw      # CGHD=<folder that contains drafter_1>

# 1. Audit (assumes nothing): drafters, pairs, classes, boxes, sizes/EXIF, rotation and text fields
python -m circuitvision.audit --cghd $CGHD --out $EXP/audit

# 2. Rotation sheets -> GATE 1: inspect, then export the convention you read off them
python -m circuitvision.visualize rotation --cghd $CGHD --out $EXP/rotation_check
export CIRCUITVISION_PLUS_AT_0=<left|right|top|bottom> CIRCUITVISION_CGHD_CLOCKWISE=<0|1>

# 3. Convert (split by drafter: val 11, test 12) and inspect the overlays
python -m circuitvision.cghd_to_yolo --cghd $CGHD --out data/yolo --val-drafters 11 --test-drafters 12
python -m circuitvision.visualize labels --yolo data/yolo --out $EXP/label_check -n 24
# GATE 2: continue only if the overlays are correct

# 4. Train (fixed config: train.BASELINE, seed 0, deterministic)
python -m circuitvision.train --data data/yolo/dataset.yaml --exp $EXP

# 5. Test-set metrics and qualitative failures
python -m circuitvision.detector_eval --weights $EXP/runs/train/weights/best.pt \
    --data data/yolo/dataset.yaml --exp $EXP

# 6. Downstream stages: GT boxes, then detector boxes
python -m circuitvision.orientation --cghd $CGHD --out $EXP/models --test-drafters 12 --exp $EXP
python -m circuitvision.evaluate --cghd $CGHD --drafters 12 --exp $EXP \
    --orient $EXP/models --ocr --weights $EXP/runs/train/weights/best.pt
```

### Fixed baseline configuration (`train.BASELINE`)

| Setting | Value |
|---|---|
| Model | YOLOv8s, COCO-pretrained `yolov8s.pt` |
| Image size | 1024 |
| Batch | 16 |
| Epochs | 100 (early-stop patience 30) |
| Optimizer | SGD, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 5e-4, 3 warm-up epochs |
| Seed | 0, deterministic |
| Augmentation | fliplr 0.5, flipud 0.5, degrees 5, scale 0.5, translate 0.1, hsv_s 0.3, hsv_v 0.4, hsv_h 0, mosaic 1.0 (off for the last 10 epochs) |
| Classes | 9: resistor, capacitor, inductor, voltage_source, diode, gnd, junction, crossover, text |
| Split | By drafter: val = drafter_11, test = drafter_12, train = all others |

`train_config.json` records this table together with the library and GPU versions and the dataset sizes. Ultralytics' own `args.yaml` is copied next to it.

### Output layout (`experiments/baseline_v1/`)

```
EVIDENCE_README.md, commit.txt, dataset_md5.txt, gpu.txt
gates/            gate1.json, gate2.json   (human approvals + notes)
logs/             stdout/stderr of every step
audit/            audit.md, audit.json, raw_xml_samples.txt, raw_xml_full_example.xml
rotation_check/   rotation_<label>.png                    (STOP: read before step 6)
conversion_stats.json
label_check/      24 overlays + index.txt                 (STOP: inspect before step 4)
train_config.json, args.yaml, results.csv, results.png
runs/train/weights/best.pt                                (Drive only, never git)
test/             metrics.json, confusion_matrix.csv, *.png, per_image.csv,
                  qualitative/{correct,missed,false_positive,text_confusion,misclassified,crowded}/
orientation.json, models/
downstream/       gt_boxes.csv, detector.csv, ocr.csv, summary.json, netlists/{gt_boxes,detector}/
tables.md         results tables, generated from the files above (report_tables.py)
```

Never committed: the CGHD data, `data/`, `runs/`, `*.pt`, `experiments/` and the evidence zip. The evidence zip leaves out `runs/`, `models/` and checkpoints.

## What each downstream number means

| Stage | Metric | Caveat |
|---|---|---|
| Orientation | Accuracy vs CGHD `<rotation>` (binned to 90°), mirrored parts excluded | Only as valid as the convention set at Gate 1; UNTESTED if Gate 1 left it `None` |
| OCR | Exact match and character accuracy (1 − CER) vs CGHD `<text>` on GT text boxes, for all labels and for value-like labels | All labels include names like R1 or S2, which the value allowlist can't read |
| Topology | Share of elements with exactly 2 terminals | **Proxy:** plausibility, not correctness |
| Topology | Cross-photo consistency: the 4 photos of one drawing should give isomorphic graphs | **Proxy:** consistent isn't the same as correct |
| Topology | `topology_match` / `exact_match` / `op_match` against a reference netlist | Only where references exist (synthetic data). CGHD has no ground-truth graphs in v14, but the audit reports any netlist-like files it finds |
| Netlist | Passes grammar + semantic checks | A valid netlist isn't necessarily the right circuit |
| Simulation | ngspice returns an operating point | |
| Detector vs GT | Detector netlist isomorphic to the GT-box netlist | Isolates detection errors from downstream errors |

Every metric is reported on all images and on the in-scope-only subset (no out-of-scope symbols).

Drawings without a ground symbol get an implicit ground: the first source's second terminal, or else the most connected net. The rule used is written into each netlist and counted in `summary.json`.

Two confidence thresholds are in play:
- The detector feeding the downstream pipeline uses `conf = 0.35`, recorded in `downstream/summary.json`.
- The qualitative analysis in `detector_eval` uses `conf = 0.25`.

mAP is computed by ultralytics over all thresholds.

In ground-truth-box mode, the netlist uses the ground-truth rotation and text as oracle inputs, so its netlist numbers isolate topology.

## Synthetic verification (no real-data claims)

```bash
python -m circuitvision.synth --out results/synthetic_topology.json        # topology benchmark
python -m circuitvision.synth --write-cghd /tmp/synth_cghd                  # CGHD-shaped data + references
python -m circuitvision.evaluate --cghd /tmp/synth_cghd --drafters 10,11,12 --exp /tmp/synth_exp --ocr
python -m circuitvision.nlp.evaluate --out results/nlp_eval                 # frozen 30-sentence set
python -m circuitvision.nlp.evaluate --set src/circuitvision/nlp/data/handwritten_fresh.jsonl
python -m circuitvision.report_tables                                       # -> results/tables.md
```

### Detector-mode end to end (SYNTHETIC)

A YOLOv8s model trained for 40 CPU epochs on synthetic CGHD-shaped drawings was run through the full
downstream pipeline on synthetic drafter 12 (6 images). `results/synthetic_detector_e2e.json`:

| Mode | Valid netlist | Simulates | Topology match | Exact match | Op-point match |
|---|---|---|---|---|---|
| GT boxes | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Detector boxes | 1.00 | 0.83 | 0.83 | 0.67 | 0.67 |

Orientation on held-out drafter 12: 5/5. OCR: 0.958 exact, 0.990 character accuracy (24 labels).
This only shows that detector mode runs end to end and that detection errors propagate as expected.
It says nothing about real photos.

### Re-check after the teammate's changes (2026-11-04, SYNTHETIC)

The synthetic GT-box run above was repeated on Linux (Python 3.13, ngspice 42) after PRs #5–#11, with orientation models trained on the synthetic tree: 18/18 valid, simulated, and matching their references on topology, values and operating point; OCR 0.914 exact and 0.950 character accuracy over 70 labels; orientation 18/18. Identical to the October run.

### NLP fixes and re-run (2026-11-04, HAND-WRITTEN)

"330R" (330 ohm) is now read as a value in both the text path and the image path, and "a pair of" counts as two. Both gaps were found by the fresh 20-sentence set, so its re-run score (1.000) is optimistic; the once-evaluated 0.900 remains the reported number. The teammate's set is unchanged at 0.750. See `results/nlp_rerun_2026-11-04/README.md`.

## Results: baseline_v1

**REAL CGHD.** Run on Colab, 2026-11-04. Every number below comes from the evidence bundle; the files that matter are copied to [`results/baseline_v1/`](results/baseline_v1/), and the full tables are in [`results/baseline_v1/tables.md`](results/baseline_v1/tables.md). Test drafter 12 was not looked at to make any training or gate decision.

### Run record

| Item | Value | Source |
|---|---|---|
| Dataset | CGHD v14, MD5 `ab1cde6feb5edaafbde711cd2059a2f4` | `dataset_md5.txt` |
| Split | train 2,981 images and 210,782 objects; validation drafter 11, 96 images and 4,032 objects; test drafter 12, 96 images and 1,628 objects | `conversion_stats.json` |
| Gates | Gate 1 `left` / `False` (Pawan, reviewed by Pranshu, issue #7); Gate 2 approved | `gates/` |
| Config | `train.BASELINE`: YOLOv8s from COCO, 100 epochs planned, imgsz 1024, batch 16, SGD, patience 30, seed 0 | `train_config.json`, `args.yaml` |
| **Training length** | **Stopped early at epoch 73**: no validation improvement for 30 epochs. Best epoch **43** (validation mAP50 0.940, mAP50-95 0.627); `best.pt` from that epoch is evaluated | `logs/train.log`, `results.csv` |
| **Hardware** | **Epochs 1–33 on a Tesla T4** (five sessions, resumed from `last.pt`), **epochs 34–73 on an A100-SXM4-40GB** (one session, 0.45 h). The best epoch was trained on the A100. Same config throughout; only the GPU changed | `logs/setup.log`, `logs/train.log`, `gpu.txt` (last session only) |
| Code | Sessions cloned `aa1df19`, `5a0d825` and finally `31ea206`, all on `main`. The only training-related difference is a guard for malformed `voltage.dc` rotations, and CGHD v14 has none; every session confirmed the conversion was identical to the one approved at Gate 2 | `commit.txt`, `logs/setup.log` |
| Packages | torch 2.11.0 (CUDA 13), ultralytics 8.4.184, EasyOCR 1.7.2, OpenCV headless 4.10.0.84 | `pip_freeze.txt` |

### Detection on test drafter 12 (REAL CGHD)

96 photos, 1,628 symbols. Source: `test/metrics.json`.

| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all classes** | **0.886** | **0.875** | **0.904** | **0.611** |
| resistor | 0.963 | 1.000 | 0.995 | 0.766 |
| capacitor | 0.739 | 1.000 | 0.972 | 0.682 |
| inductor | 0.980 | 0.833 | 0.953 | 0.678 |
| voltage source | 0.608 | 0.594 | **0.482** | 0.257 |
| diode | 0.966 | 0.714 | 0.939 | 0.706 |
| junction | 0.943 | 0.874 | 0.906 | 0.411 |
| crossover | 0.947 | 1.000 | 0.994 | 0.655 |
| text | 0.941 | 0.982 | 0.989 | 0.736 |

At confidence 0.25: 1,526 true positives, 181 false positives, 102 misses. The commonest confusion is **voltage source → capacitor (9)**. Drafter 12 has no `gnd` symbols, so ground detection is not measured here.

### Orientation (REAL CGHD)

| What | Result | Source |
|---|---|---|
| Classifier on real test crops | voltage source 0.906 (32 crops), diode 0.775 (40) | `orientation.json` |
| In the pipeline, ground-truth boxes | 0.833 (72 polarised symbols) | `downstream/summary.json` |
| In the pipeline, detector boxes (matched to ground truth) | 0.639 | same |

The voltage-source test crops were also seen on 2026-11-02 while choosing the `voltage.dc` fix (Full dataset checks, above), so the 0.906 is slightly optimistic. The test counts are small.

### OCR (REAL CGHD)

EasyOCR on the 656 ground-truth text boxes: **0.021 exact, 0.148 character accuracy**. The labels are mostly handwritten words ("Resistor", "DC Source", "R1"); EasyOCR is trained on printed text. Only **4 labels are values**, so value reading is effectively **not measured** on this test drafter. Source: `downstream/summary.json`, `downstream/ocr.csv`.

### Downstream: netlists on real photos (REAL CGHD)

Read these with two facts in mind:
- **Half the test photos have nothing to reconstruct.** Circuits C139–C144 (48 photos) are logic-gate diagrams with no in-scope parts. They always give empty netlists ("no ground node"). The meaningful base is the **48 analog photos** (C133–C138).
- **"Valid" is not "correct".** A netlist can pass the grammar and semantic checks and still be the wrong circuit. **Topology match against the reference netlists** (T-3, checked against the photos) is the measure of correctness.

| 48 analog photos | Ground-truth boxes | Detector boxes |
|---|---|---|
| Valid netlist | 12 (25%) | 20 (42%) |
| **Topology match vs reference** | **18 (37.5%)** | **5 (10.4%)** |
| Exact match (values too) | 18 | 3 |

C137, the only circuit with nothing out of scope (8 photos): ground-truth boxes give 1 of 8 correct, including the simulated operating point. The detector gives 0 of 8, though 5 of its 8 netlists are "valid". The detector adds false parts, such as a capacitor on top of the battery, and they happen to close the loose ends.

Computed from `downstream/gt_boxes.csv` and `downstream/detector.csv`, keeping rows whose image name matches `C13[3-8]_`. The full-set figures (all 96 photos, proxies, consistency) are in `tables.md`.

### Where downstream fails (error analysis, C-3)

- **Battery terminals are the main bottleneck.** With ground-truth boxes, a part that ends up with other than two terminals is dropped from the netlist. Across the 48 analog photos there are 32 such drops: **voltage source 16**, capacitor 5, resistor 4, diode 4, inductor 3. A typical case is `C137_D1_P2`: the battery is found with 3 terminals instead of 2, probably because its long plates stick out of the box and are traced as wires. Source: `downstream/netlists/gt_boxes/`.
- **Wire gaps on real photos** split a net that should be one, for example the bottom wire in C136 (`R1 6 9` against `V1 5 0`).
- **Label text attached as a value.** A label such as "Resistor 2" gives a "2" that is attached as 2 Ω, so exact match fails even when the topology is right.
- **Detector false positives** make wrong netlists look valid (above), and voltage sources are the weakest class.

### Post-hoc fix (2026-11-04, measured after seeing the test results)

A user photo exposed a faint stretch of pen that split one straight wire. Gaps are now also closed along straight horizontal and vertical lines (`topology.extract_nets`), never sideways. Re-running the GT-box downstream evaluation on the same 48 analog photos gives **21 correct circuits instead of 18**, with no photo worse; the synthetic topology benchmark is identical (0.994). Because it was measured after the test results were known, the baseline figures above remain the reported result. Detector-mode downstream was not re-run.

### Summary statement

> Detection baseline established on the real test drafter: mAP50 0.904 and mAP50-95 0.611, with the voltage source the weakest class. Downstream, with perfect boxes, 37.5% of the analog test photos reconstruct to the correct circuit and 10% with detected boxes. Wire tracing at battery symbols, and OCR on handwritten labels, are the main limitations.
