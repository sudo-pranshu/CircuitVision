# Experiments

## Status

Every result carries one of three labels: **REAL CGHD**, **SYNTHETIC/CONTROLLED** or **UNTESTED**.

| Result | Label | Evidence |
|---|---|---|
| Real CGHD annotation format (rotation nested in `<bndbox>`, 1-degree values) | REAL CGHD (one file) | `tests/data/cghd_real/` (from the Hugging Face copy, not the MD5-checked Zenodo zip) |
| Wire tracing on synthetic photos with known graphs: 0.994 exact over 11 conditions | SYNTHETIC/CONTROLLED | `results/synthetic_topology.json` |
| GT-box pipeline on CGHD-shaped synthetic data: 18/18 topology, exact and operating-point match | SYNTHETIC/CONTROLLED | Run log below; reproducible with `synth --write-cghd` |
| EasyOCR on synthetic value labels: 0.914 exact, 0.950 character accuracy | SYNTHETIC/CONTROLLED | Same run |
| Text → netlist: 30-sentence set 0.80 → 1.00 exact (optimistic); fresh 20-sentence set 0.90 | CONTROLLED (hand-written) | `results/nlp_eval*.json` |
| `train.py`, `detector_eval.py`, `evaluate.py --weights --ocr`, `orientation.py`, `report_tables.py` | Executed end to end on CPU (torch 2.14, ultralytics 8.4.184) | Synthetic data only; numbers carry no meaning about real photos |
| CGHD audit, rotation convention | UNTESTED | Awaiting the Colab run and Gate 1 |
| YOLOv8s detection on test drafter 12 | UNTESTED | Awaiting the Colab run |
| Downstream stages on real photos (GT boxes / detector) | UNTESTED | Awaiting the Colab run |

No REAL CGHD metric exists yet, and nothing here may be reported as a real-photo result until `baseline_v1` has run.

Why it hasn't run: the build environment can't reach Zenodo, GitLab, Kaggle or Hugging Face, and has no GPU. The experiment runs on Colab.

## Findings from real CGHD annotations (before any training)

These were checked against real annotation files (`drafter_13/C145_D1_P1.xml`, `drafter_1/C1_D1_P1.xml`) and the CGHD paper. They changed the code.

- **Rotation is nested inside `<bndbox>`**, with 1-degree resolution (for example 2, 8, 90, 269). The parser read it from `<object>`, so every rotation would have been missing. Fixed in `parse_voc`; regression test `test_real_xml_rotation_is_nested_in_bndbox`.
- **What 0° means can be partly inferred.** A wide resistor box has `rotation=2` and a tall one `90`, so 0° is horizontal for resistors. The audit now prints, for every class, how box shape is distributed in each rotation bin. Clockwise vs counter-clockwise, and which side is + at 0°, can't be read from boxes. Those remain Gate 1 decisions.
- **Mirror annotations exist** (1,437 in the latest release). Their effect on polarity depends on undocumented symbol templates, so mirrored polarized symbols are excluded from orientation training and scoring, and counted.
- **CGHD has many out-of-scope symbols** (transistors, ICs, lamps, switches). They're now masked during wire tracing so their ink can't short two wires together, and every downstream metric is also reported on the in-scope-only subset.
- **The CGHD paper uses a different split:** train on drafters 1–20 and 25, validate on 21–22, test on 23–24. This baseline keeps the split fixed earlier (validation 11, test 12), so its numbers aren't directly comparable with the paper's.
- **A newer release exists** (Zenodo v16, 2025, 4.9 GB). The notebook pins v14 by MD5 for reproducibility.

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
  - The code defaults are **unverified** assumptions.
  - If the sheets are inconclusive, leave the convention as `None`. Orientation is then skipped and reported as UNTESTED.
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

## Results: baseline_v1

*Pending. To be filled only from files in the evidence bundle.*

The required statement at this stage:

> Detection baseline established; downstream real-photo performance is still being evaluated.

This statement applies only once step 5 has actually run.
