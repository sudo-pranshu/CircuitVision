# Experiments

## Status

Every result carries one of three labels: **REAL CGHD**, **SYNTHETIC/CONTROLLED** or **UNTESTED**.

| Result | Label | Evidence |
|---|---|---|
| Voltage divider end to end (V(2) = 0.4545 V) | SYNTHETIC/CONTROLLED | `tests/test_pipeline.py` |
| Wire tracing, crossovers, grammar, simulation | SYNTHETIC/CONTROLLED | Unit tests |
| Orientation SVM on drawn diode crops (≥ 95%) | SYNTHETIC/CONTROLLED | `tests/test_orientation.py` |
| Text → netlist: 8/8 handwritten sentences | SYNTHETIC/CONTROLLED | `tests/test_text2netlist.py` |
| Notebook logic: gates, logging, failure stop, evidence bundle | SYNTHETIC/CONTROLLED | Executed locally on a fake CGHD tree |
| CGHD audit, rotation convention | UNTESTED | Awaiting the Colab run and Gate 1 |
| YOLOv8s detection on test drafter 12 | UNTESTED | Awaiting the Colab run |
| Downstream stages on real photos (GT boxes / detector) | UNTESTED | Awaiting the Colab run |
| `train.py`, `detector_eval.py`, `ocr.py` on a GPU | UNTESTED | PyTorch is unavailable in the build environment |

No REAL CGHD number exists yet. Nothing here may be reported as a real-photo result until `baseline_v1` has run.

Why it hasn't run: the build environment can't reach Zenodo, GitLab or Kaggle, and has no GPU. The experiment runs on Colab.

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

Both approvals are saved, with your notes, to `gates/gate1.json` and `gates/gate2.json`.

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
downstream/       gt_boxes.csv, detector.csv, summary.json
```

Never committed: the CGHD data, `data/`, `runs/`, `*.pt`, `experiments/` and the evidence zip. The evidence zip leaves out `runs/`, `models/` and checkpoints.

## What each downstream number means

| Stage | Metric | Caveat |
|---|---|---|
| Orientation | Accuracy vs CGHD `<rotation>` | Only as valid as the convention set at Gate 1; UNTESTED if Gate 1 left it `None` |
| OCR | Exact match vs CGHD `<text>` on GT text boxes | Strict string match |
| Topology | Share of elements with exactly 2 terminals | **Proxy.** CGHD has no ground-truth netlists, so this measures plausibility, not correctness |
| Netlist | Passes grammar + semantic checks | |
| Simulation | ngspice returns an operating point | |
| Detector vs GT | `struct_match`: same element counts per class and same node count | Not full graph isomorphism |

Two confidence thresholds are in play:
- The detector feeding the downstream pipeline uses `conf = 0.35`, recorded in `downstream/summary.json`.
- The qualitative analysis in `detector_eval` uses `conf = 0.25`.

mAP is computed by ultralytics over all thresholds.

In ground-truth-box mode, the netlist uses the ground-truth rotation and text as oracle inputs, so its netlist numbers isolate topology.

## Results: baseline_v1

*Pending. To be filled only from files in the evidence bundle.*

The required statement at this stage:

> Detection baseline established; downstream real-photo performance is still being evaluated.

This statement applies only once step 5 has actually run.
