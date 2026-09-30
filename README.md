# CircuitVision

Automated conversion of hand-drawn circuit diagrams into simulatable SPICE netlists using deep learning and image processing.

```
photo ─► preprocess ─► YOLO detection ─► wire tracing ─► value OCR ─► netlist (.cir) ─► ngspice
          (binarize)    (components)      (nets)          (labels)
```

## Setup (macOS, Apple Silicon)

```bash
cd ~/CircuitVision
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,dev]"
brew install ngspice        # optional, to simulate the output
pytest -q                   # 3 tests should pass
```

## Try it without a trained model

The example ships with ground-truth boxes, so it exercises wire tracing and netlist generation:

```bash
python -m circuitvision.pipeline examples/divider.png --xml examples/divider.xml
ngspice -b out/divider.cir
```

The netlist goes to `out/divider.cir`, and `out/divider_overlay.png` colours each detected net.

## Dataset: CGHD

- Zenodo: https://zenodo.org/records/14042961 (about 4.4 GB, CC BY 3.0)
- It holds about 3k phone photos of hand-drawn circuits in Pascal VOC XML, organised as `drafter_<k>/images` and `drafter_<k>/annotations`.

Convert it to YOLO format with the reduced class set. The split is by drafter to avoid leakage:

```bash
python -m circuitvision.cghd_to_yolo --cghd ~/data/cghd --out data/yolo
```

## Train the detector

```bash
yolo detect train data=data/yolo/dataset.yaml model=yolov8s.pt imgsz=1024 epochs=100 device=mps
python -m circuitvision.pipeline my_circuit.jpg --weights runs/detect/train/weights/best.pt
```

On a MacBook Air, run short experiments locally with `device=mps` and do full training on Colab.

## Scope (mini project)

- **Classes:** resistor, capacitor, inductor, voltage source, diode, ground, junction, crossover, text (see `classes.py`).
- **Two-terminal elements only.** Anything else is flagged as a `* WARNING` in the netlist.

## Module map

| File | Role |
|---|---|
| `preprocess.py` | Adaptive thresholding for phone photos, speckle removal |
| `cghd_to_yolo.py` | CGHD XML → YOLO labels, split by drafter |
| `detect.py` | YOLO inference wrapper |
| `topology.py` | Masks boxes, labels wires, reads terminal rings, handles crossovers |
| `netlist.py` | Assigns ground to node 0, attaches values to nearest text, emits SPICE |
| `pipeline.py` | CLI and debug overlay |

## Evaluation plan

1. **Detection:** mAP@0.5 on the test drafter.
2. **Wire tracing with ground-truth boxes (`--xml`):** net accuracy, which isolates topology errors from detection errors.
3. **End to end:** percentage of netlists that are correct, and percentage that simulate in ngspice.

## Next steps

- [ ] Download CGHD, convert it, and train the baseline YOLOv8s.
- [ ] Value OCR on `text` boxes, either with an off-the-shelf engine or with a CNN+CTC model trained on CGHD's text labels.
- [ ] Measure wire-tracing accuracy on CGHD test images using ground-truth boxes.
- [ ] NLP extension: a CFG netlist validator and text → netlist generation.
