# CircuitVision

Automated conversion of hand-drawn circuit diagrams into simulatable SPICE netlists, using deep learning and image processing.

```
photo ─► preprocess ─► YOLO detection ─► wire tracing ─► value OCR ─► netlist ─► grammar check ─► ngspice
          (binarize)    (components)      (nets)          (EasyOCR)    (.cir)     (CFG + Earley)    (.op)
```

## Setup (macOS, Apple Silicon)

```bash
cd ~/CircuitVision
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,app,dev]"
brew install ngspice
pytest -q                   # 16 tests (2 skip if ngspice is missing)
```

## Try it without a trained model

The example ships with ground-truth boxes, so it runs every stage except detection:

```bash
python -m circuitvision.pipeline examples/divider.png --xml examples/divider.xml --simulate
```

Expected output:

```
V1 1 0 DC 5
R1 1 2 10k
R2 2 0 1k
validation: OK
  v(1)  5
  v(2)  0.4545
```

The pipeline writes `out/divider.cir` and `out/divider_overlay.png`, which colours each detected net.

## Dataset: CGHD

- Zenodo: https://zenodo.org/records/14042961 (about 4.4 GB, CC BY 3.0, J. Bayer et al.)
- About 3k phone photos of hand-drawn circuits with Pascal VOC XML annotations, organised as `drafter_<k>/{images,annotations}`.

Convert it to YOLO format. The split is by drafter, so the same drawing never appears in both training and test data:

```bash
python -m circuitvision.cghd_to_yolo --cghd ~/data/cghd --out data/yolo
```

## Train the detector

The recommended route is Colab: open `notebooks/train_colab.ipynb`, set the runtime to T4 GPU, and run all cells. The notebook downloads CGHD, converts it, trains, saves `best.pt` to your Drive, and evaluates on the test drafter.

To train locally on your Mac's GPU (MPS), start with a smoke test:

```bash
python -m circuitvision.train --data data/yolo/dataset.yaml --quick
python -m circuitvision.train --data data/yolo/dataset.yaml --epochs 100
```

## Run on your own photo

```bash
python -m circuitvision.pipeline my_circuit.jpg --weights best.pt --simulate
python -m circuitvision.app --weights best.pt          # web demo at localhost:7860
```

## Polarity (diodes, sources)

Rotation is stored in degrees CCW, with `+`/anode on the left at 0°. The orientation classifier is HOG + linear SVM per class, trained on CGHD crops with 4× rotation augmentation:

```bash
python -m circuitvision.orientation --cghd ~/data/cghd --out models
python -m circuitvision.pipeline my_circuit.jpg --weights best.pt --orient models --simulate
```

Before trusting polarity, check `CGHD_CLOCKWISE` and `PLUS_AT_0` in `orientation.py` against 2–3 CGHD images.

## Text → netlist (NLP)

```bash
python -m circuitvision.nlp.text2netlist train
python -m circuitvision.nlp.text2netlist "a 5V battery driving a 10k resistor in series with a 1k resistor"
```

The pipeline runs in four steps:

1. Tokenize.
2. Tag with an HMM (Viterbi, add-k smoothing, 5% word dropout for unseen words).
3. Chunk each component noun with its value.
4. Parse the series/parallel structure, using right attachment for "which is in parallel with".

Semantic constraints then correct the tagger: the value's unit decides the class, a source must be in volts, and kilo/mega values imply a resistor. The source is placed across the load, and the result is validated by `grammar.py`.

| Test set | Token accuracy | Exact circuit match |
|---|---|---|
| Same templates as training | 1.000 | 1.000 (not informative) |
| Held-out phrasings and nouns | 0.879 | 0.998* |
| 8 handwritten sentences (`tests/test_text2netlist.py`) | — | 8/8 |

\* The unit constraints were designed after inspecting held-out errors, so this row is optimistic. The handwritten set is the clean check.

## Evaluate

```bash
# Wire tracing alone, using ground-truth boxes
python -m circuitvision.evaluate --images D/images --annotations D/annotations

# End to end, comparing the detector's output against the ground-truth-box run
python -m circuitvision.evaluate --images D/images --annotations D/annotations --weights best.pt
```

| Metric | Meaning |
|---|---|
| mAP@0.5 | Detection quality on the test drafter (printed by `train`) |
| `terminals_ok` | Share of elements with exactly 2 terminals |
| `valid` | Netlist passes the grammar and semantic checks |
| `simulates` | ngspice produced an operating point |
| `struct_match` | Detector netlist matches the ground-truth-box netlist in element counts per class and node count |

## Module map

| File | Role | Syllabus link |
|---|---|---|
| `preprocess.py` | Adaptive thresholding, speckle removal | Image processing |
| `cghd_to_yolo.py` | CGHD XML → YOLO labels, split by drafter | |
| `detect.py`, `train.py` | YOLOv8 detection; training with auto device selection | CV / deep learning |
| `topology.py` | Masks boxes, labels wire regions, reads terminal rings, handles crossovers | Image processing |
| `ocr.py` | Reads component values from text boxes | CV + text recognition |
| `netlist.py` | Ground → node 0, nearest-label values, SPICE output | |
| `grammar.py` | CFG for the SPICE subset, Earley recognizer, semantic constraints | NLP: parsing (M3), constraints (M4) |
| `simulate.py` | ngspice batch run, operating point | |
| `evaluate.py` | Batch metrics → CSV | Precision / accuracy (M2) |
| `orientation.py` | HOG + SVM rotation classifier, polarity convention | Image processing |
| `nlp/hmm.py` | HMM tagger, Viterbi, per-tag P/R/F1 | NLP: stochastic tagging (M2) |
| `nlp/parser.py`, `nlp/tree.py` | Chunking, series/parallel semantics, netlist | NLP: parsing, compositional semantics (M3–M4) |
| `nlp/corpus.py` | Synthetic tagged corpus with a held-out phrasing split | |
| `app.py` | Gradio demo | |

## Scope and limitations

- **Classes:** resistor, capacitor, inductor, voltage source, diode, ground, junction, crossover, text.
- **Two-terminal elements only.** Anything else is flagged as `* WARNING` in the netlist and by the validator.
- **Polarity needs orientation models.** Without them, terminal order falls back to geometry (left→right or top→bottom).

## Roadmap

- [ ] Train the baseline YOLOv8s and report test mAP.
- [ ] Measure wire-tracing accuracy on CGHD test images.
- [x] Polarity for diodes and sources from the symbol's orientation.
- [x] Text description → netlist (HMM tagger + series/parallel parser).
- [ ] Verify the CGHD rotation convention and train the orientation models.
- [ ] Text → netlist: nested groupings ("A and B in parallel, in series with C"), a larger handwritten test set.

## Credits

Dataset: J. Bayer et al., *A Public Ground-Truth Dataset for Handwritten Circuit Diagram Images* (CGHD), CC BY 3.0.
