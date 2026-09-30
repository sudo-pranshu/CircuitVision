# Experiments

## Status

| Result | Kind | State |
|---|---|---|
| Voltage divider end to end (V(2) = 0.4545 V) | Synthetic | Verified (`tests/test_pipeline.py`) |
| Wire tracing, crossovers, grammar, simulation | Synthetic | Verified (unit tests) |
| Orientation SVM on drawn diode crops (≥ 95%) | Synthetic | Verified (`tests/test_orientation.py`) |
| Text → netlist: 8/8 handwritten sentences | Controlled | Verified (`tests/test_text2netlist.py`) |
| CGHD dataset audit, rotation convention | **Real** | **Not run yet** |
| YOLOv8s detection on test drafter 12 | **Real** | **Not run yet** |
| Downstream stages on real photos (GT boxes / detector) | **Real** | **Not run yet** |

No real-data number exists yet. Nothing in this repository should be reported as a real-photo result until `baseline_v1` has been run.

Why it hasn't run: the build environment can't reach Zenodo, GitLab or Kaggle to fetch CGHD, and it has no GPU. Running the experiment is on Colab.

## baseline_v1: how to reproduce

The whole experiment is `notebooks/train_colab.ipynb`, run on a Colab T4 GPU. The same steps work from a shell:

```bash
pip install -e ".[detect]"            # plus: apt install ngspice  (or brew install ngspice)
EXP=experiments/baseline_v1

# 0. Data: Zenodo 14042961, cghd-zenodo-14.zip, md5 ab1cde6feb5edaafbde711cd2059a2f4
unzip cghd-zenodo-14.zip -d data/cghd_raw      # CGHD=<folder that contains drafter_1>

# 1. Audit (assumes nothing): drafters, pairs, classes, boxes, sizes/EXIF, rotation and text fields
python -m circuitvision.audit --cghd $CGHD --out $EXP/audit

# 2. Rotation convention: inspect the sheets, then set PLUS_AT_0 / CGHD_CLOCKWISE in orientation.py
python -m circuitvision.visualize rotation --cghd $CGHD --out $EXP/rotation_check

# 3. Convert (split by drafter: val 11, test 12) and inspect the overlays
python -m circuitvision.cghd_to_yolo --cghd $CGHD --out data/yolo --val-drafters 11 --test-drafters 12
python -m circuitvision.visualize labels --yolo data/yolo --out $EXP/label_check -n 24

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
commit.txt, dataset_md5.txt
audit/            audit.md, audit.json, raw_xml_samples.txt
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

Never committed: the CGHD data, `data/`, `runs/`, `*.pt`, `experiments/` and the evidence zip.

## What each downstream number means

| Stage | Metric | Caveat |
|---|---|---|
| Orientation | Accuracy vs CGHD `<rotation>` | Only as valid as the convention verified in step 2 |
| OCR | Exact match vs CGHD `<text>` on GT text boxes | Strict string match |
| Topology | Share of elements with exactly 2 terminals | **Proxy.** CGHD has no ground-truth netlists, so this measures plausibility, not correctness |
| Netlist | Passes grammar + semantic checks | |
| Simulation | ngspice returns an operating point | |
| Detector vs GT | `struct_match`: same element counts per class and same node count | Not full graph isomorphism |

In ground-truth-box mode, the netlist uses the ground-truth rotation and text as oracle inputs, so its netlist numbers isolate topology.

## Results: baseline_v1

*Pending. To be filled only from files in the evidence bundle.*

The required statement at this stage:

> Detection baseline established; downstream real-photo performance is still being evaluated.

This statement applies only once step 5 has actually run.
