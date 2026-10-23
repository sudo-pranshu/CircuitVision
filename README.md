# CircuitVision

Automated conversion of hand-drawn circuit diagrams into validated, simulatable SPICE netlists, using computer vision, image processing and NLP.

```
photo ─► binarize ─► YOLOv8 ─► orientation ─► wire tracing ─► OCR ─► netlist ─► CFG/Earley ─► ngspice
         (paper-     (9        (HOG + SVM,     (nets,          (Easy- (.cir)    + semantic     (.op)
          relative    classes)  polarity)       crossovers)     OCR)              checks
          contrast)
text ─► tokenize ─► HMM tagger (Viterbi) ─► chunking + series/parallel parser ─► netlist ─► same checks
```

## Status

**No real-photo results exist yet.** The pipeline is implemented and tested, and every stage has been executed end to end on CPU against synthetic data in the real CGHD format. The real CGHD experiment (`baseline_v1`) is ready to run on Colab. See [EXPERIMENTS.md](EXPERIMENTS.md); every result there is labelled **REAL CGHD**, **SYNTHETIC/CONTROLLED** or **UNTESTED**.

| Result (all synthetic or controlled) | Value | Source |
|---|---|---|
| Wire tracing on synthetic photos with known graphs (11 conditions) | 0.994 exact topology | `results/synthetic_topology.json` |
| Text → netlist, fresh 20-sentence test set (evaluated once) | 0.90 exact circuit, 1.00 token accuracy | `results/nlp_eval_fresh.json` |
| Text → netlist, 30-sentence set (before → after development) | 0.80 → 1.00 exact (after is optimistic) | `results/nlp_eval*.json` |

All tables: `results/tables.md`, regenerated with `python -m circuitvision.report_tables`.

## Setup

```bash
git clone https://github.com/sudo-pranshu/CircuitVision.git && cd CircuitVision
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[detect,app,dev]"
bash scripts/pin_opencv.sh        # exactly one OpenCV build; run after every pip install
brew install ngspice              # Linux: apt install ngspice
pytest -q                         # ~105 tests, a few skip without ngspice
```

Only the base install is needed for wire tracing, netlists, grammar, simulation, NLP and the tests (`pip install -e ".[dev]"`). `detect` adds YOLO and EasyOCR (PyTorch); `app` adds Gradio.

## Usage

```bash
# Image -> netlist with ground-truth boxes (no model needed)
python -m circuitvision.pipeline examples/divider.png --xml examples/divider.xml --simulate

# Image -> netlist with a trained detector (and optional orientation models)
python -m circuitvision.pipeline photo.jpg --weights best.pt --orient models --simulate
python -m circuitvision.app --weights best.pt                 # web demo, localhost:7860

# Text -> netlist
python -m circuitvision.nlp.text2netlist "a 9V battery driving a 1k resistor in series with (a 2k resistor in parallel with a 3k resistor)"
```

The pipeline writes `out/<name>.cir` and `out/<name>_overlay.png`, with each net coloured. Every netlist states which ground rule was used: the ground symbol, or an implicit ground when the drawing has none.

## Real-data experiment

Open `notebooks/train_colab.ipynb` in Colab on a T4 and choose Run all. It runs these steps:

1. Caches CGHD v14 on Drive and checks its MD5.
2. Audits the full dataset, with progress output.
3. Draws rotation sheets, then stops at **Gate 1**: you confirm the rotation convention.
4. Converts the dataset to YOLO format and draws label overlays, then stops at **Gate 2**: you confirm the labels.
5. Trains YOLOv8s with a fixed config and seed.
6. Evaluates on test drafter 12.
7. Evaluates every downstream stage twice, with ground-truth boxes and with detector boxes.
8. Writes the results tables and an evidence zip.

The shell equivalents and the full configuration are in EXPERIMENTS.md.

## Module map

| File | Role | Course link |
|---|---|---|
| `preprocess.py` | Binarisation: grey closing estimates the paper, contrast relative to the paper, Otsu (handles grid paper and shadows) | Image processing |
| `cghd_to_yolo.py` | CGHD VOC parsing (rotation in `<bndbox>`, mirror), explicit class map, EXIF-aware loading, YOLO export split by drafter | |
| `audit.py`, `visualize.py` | Dataset audit with convention evidence, label overlays, rotation sheets | |
| `detect.py`, `train.py`, `detector_eval.py` | YOLOv8 inference, fixed-config training, test metrics and failure categories | CV / deep learning; P/R (M2) |
| `orientation.py` | HOG + linear SVM rotation classifier, polarity convention | Image processing |
| `topology.py` | Box masking, wire components, terminal rings, crossovers; all sizes scale with symbol size | Image processing |
| `ocr.py`, `netlist.py` | EasyOCR values; RKM codes (4k7), unit/class checks, one label per element, implicit ground | |
| `grammar.py` | CFG for the SPICE subset, Earley recogniser, semantic checks | NLP: parsing (M3), constraints (M4) |
| `simulate.py`, `compare.py` | ngspice operating point; graph isomorphism and operating-point equivalence against a reference | |
| `evaluate.py`, `report_tables.py` | Stage-wise metrics (GT vs detector, in-scope subset, proxies labelled), tables from artifacts | |
| `synth.py` | Synthetic photos with known graphs: topology benchmark and CGHD-shaped test data | |
| `nlp/` | HMM tagger (Viterbi), chunking, series/parallel/nested parser, synthetic corpus, frozen evaluation sets | NLP: tagging (M2), parsing and semantics (M3–M4) |

## Limitations

- **Two-terminal elements only.** Transistors, ICs, switches, lamps and similar symbols are out of scope: they're masked so they can't short wires, and metrics are also reported on images without them.
- **Topology accuracy on real photos can't be measured exactly.** CGHD v14 has no ground-truth circuit graphs. On real data, topology is reported through proxies (terminal counts, and consistency across photos of the same drawing). Exact topology is measured only on synthetic data.
- **The rotation convention must be confirmed at Gate 1.** Until then, polarity results are unverified, and mirrored polarized symbols are excluded.
- **Text → netlist covers series/parallel descriptions.** It doesn't handle bridges or circuits that need named nodes, and it misses some phrasings: "330R", "a pair of".
- **The split differs from the CGHD paper's** (theirs tests on drafters 23–24), so results aren't directly comparable.

## Credits

Dataset: F. Thoma, J. Bayer, Y. Li, A. Dengel, *A Public Ground-Truth Dataset for Handwritten Circuit Diagram Images*, ICDAR 2021 (CGHD, CC BY 3.0). One annotation file is included as a test fixture, with attribution in `tests/data/cghd_real/README.md`.
