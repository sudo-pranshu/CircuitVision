# CircuitVision: Project Status

Updated 2026-11-04 · `main` · Full reference: [PROJECT_BIBLE.md](PROJECT_BIBLE.md)

## Summary

The pipeline (image → netlist → validation → simulation, plus text → netlist) is **implemented, tested and verified on synthetic data**.

**The real CGHD experiment (baseline_v1) has run** (Colab, 2026-11-04). Results are in [EXPERIMENTS.md](../EXPERIMENTS.md#results-baseline_v1), and the key files are in [`results/baseline_v1/`](../results/baseline_v1/).

| REAL CGHD, test drafter 12 | Result |
|---|---|
| Detection, 96 photos, 1,628 symbols | **mAP50 0.904, mAP50-95 0.611** (P 0.886, R 0.875); weakest class voltage source (AP50 0.48) |
| Orientation | classifier: voltage source 0.906, diode 0.775; in the pipeline 0.833 (GT boxes) and 0.639 (detector) |
| OCR on handwritten labels | 0.021 exact; only 4 labels are values, so value reading is effectively not measured |
| **Correct circuit**, 48 analog photos, against the T-3 references | **37.5% with ground-truth boxes, 10.4% with detector boxes** |

Training stopped early at **epoch 73 of 100** (patience 30). The best epoch, **43**, is the one evaluated. It ran on a T4 for epochs 1–33 and an A100 for 34–73, with the same config. Half the test photos (C139–C144) are logic-gate diagrams with nothing to reconstruct. The main downstream failure is battery terminals in wire tracing, followed by OCR on handwriting (C-3, in EXPERIMENTS).

**What's left:** T-5 (report numbers against `results/baseline_v1/`), T-6 (re-run the tables on a clean machine), the hand-filled report, and demo prep with the real weights.

## Sync audit by Pranshu (2026-11-04)

Pranshu reviewed PRs #5–#11 and kept them as they are. His fixes (`5137acd`) and docs (`93a7bea`) were first merged and reverted (`ba0ee7e`), then brought onto `main` from `feature/Pawan` by cherry-picking both commits, after the status update above. Each fix was checked against the running training:

| Fix | Effect on the running training |
|---|---|
| A malformed `voltage.dc` rotation no longer crashes `parse_voc` | None. CGHD v14 has no malformed values: 310 `voltage.dc` objects, of which 128 have no rotation, which is already handled. The label conversion is identical, so Gate 2 stays valid. |
| "330R" is read as 330 Ω in the image and text paths | None on training. In the image path it only affects OCR value attachment, and the drafter-12 drawings have no written values. |
| "a pair of" counts as two | None (text path only) |
| NLP failures no longer mislabelled "tokenization" | Labels only; Pawan's set stays 0.75, now 2 tagging, 1 structure and 2 components/values |
| Gate 1 values prefilled in the notebook | None: Gate 1 is already saved on Drive |
| Convention labelled as the Gate 1 decision; Pawan's set added to `results/tables.md` | None |

The NLP fixes target the fresh set's two failures. **The fresh set's recorded 0.90 therefore stays the headline**, and its 1.00 re-run is optimistic (`results/nlp_rerun_2026-11-04/`).

## What works

- Every stage runs end to end on CPU: binarize → detect → orientation → wire tracing → OCR → netlist → CFG/Earley + semantic checks → ngspice.
- Text → netlist with an HMM tagger and a series/parallel parser, including nesting, counts and RKM values.
- The Colab notebook is restart-safe:
  - stage status is printed at the start;
  - finished stages are skipped;
  - training resumes automatically;
  - gate approvals are saved;
  - Gate 2 overlays cover every split, every class and the EXIF re-encoded photos.
- The downstream evaluation scores drafter-12 netlists against the 12 reference netlists in [`references/`](../references/README.md) (C-4, PR #10).

## Final report

The submission report (Word, institute template, 48 pages) is built from [`report/`](../report/README.md). Pranshu updated its source on 2026-11-04 with the Gate 1 and 2 decisions, the full-dataset findings, the new defects, the NLP re-run and 129 tests. The team fills in the real-CGHD values by hand from the evidence files (decided 2026-11-04). Every such value is highlighted `PENDING: real CGHD run` (§6.3 and a few sentences), and the cover's date of submission is a placeholder. There is no slide deck.

## Tested

| What | Result | Label |
|---|---|---|
| Unit and regression tests | 129 pass (`pytest -q`; macOS and Linux, ngspice present, 2026-11-04) | — |
| GT-box end to end, synthetic drafters 10–12, re-run after PRs #5–#11 | 18/18 topology, exact and op match; OCR 0.914 / 0.950; orientation 18/18 | SYNTHETIC |
| Wire tracing, 11 photo conditions | 0.994 exact topology | SYNTHETIC |
| Detector-mode end to end (synthetic YOLO, 6 images) | 0.83 topology, 0.67 exact/op match | SYNTHETIC |
| Text → netlist, fresh 20 sentences (evaluated once) | 0.90 exact circuit | HAND-WRITTEN |
| Text → netlist, Pawan's 20 sentences (T-7, evaluated once) | 0.75 exact circuit, component F1 0.986 | HAND-WRITTEN |
| Text → netlist re-run after the 2026-11-04 fixes | Fresh 1.00 (optimistic), 30-sentence 1.00, Pawan's 0.75 unchanged | HAND-WRITTEN |
| Training kill + resume (`train.py --auto`) | Resumed at epoch 2, then detected as finished | Real ultralytics, CPU |
| Training resume on Colab | Resumed across six sessions and two GPUs (T4, then A100) with the saved gates | Real |
| Demo app (`app.py`) | Starts, serves on localhost, processes a photo end to end (throwaway weights) | — |

## Reproduced by the teammate (T-1, 2026-11-02)

macOS on Apple Silicon, Python 3.13.18, full install with OpenCV headless 4.10.0.84, ngspice 47. Regenerated outside the repository and compared with the committed files.

| What | Result |
|---|---|
| `pytest -q` | 107 pass, 0 skipped (before the three label-fix regression tests) |
| Divider example with `--simulate` | Valid; v(2) = 0.4545 V |
| `results/tables.md` | Identical |
| `synthetic_topology.json` and `.txt` | Identical |
| `nlp_eval.json` and `.md`, `nlp_eval_fresh.json` and `.md`, `nlp_synthetic.txt` | Identical |
| GT-box pipeline, synthetic drafters 10–12 | 18/18: topology, exact and operating-point match 1.00 |
| EasyOCR, same run (70 labels) | 0.914 exact, 0.950 character accuracy, as recorded |

The EasyOCR model download from `release-assets.githubusercontent.com` was reset on one network and worked on another.

Not reproduced: detector-mode end to end, which needs a 40-epoch CPU training run.

## Not tested

- The Gradio demo with the real weights (`best.pt` and `models/` are on Drive, not in the zip).
- Value reading on real handwritten values: drafter 12 has only 4 value labels.

## Tasks

| ID | Task | State |
|---|---|---|
| T-1 | Reproduce the tests and synthetic results | Done 2026-11-02 |
| T-2 | Independent review of Gates 1 and 2 | Done: Pranshu agreed ([issue #7](https://github.com/sudo-pranshu/CircuitVision/issues/7)) |
| T-3 | Reference netlists for drafter 12 | Done: 12 files, drafted by Claude from the photos, checked by Pawan (PR #9) |
| T-4 | Literature review | Done: report Chapter 2, sources reviewed by Pawan |
| T-7 | New frozen NLP test set | Done: 20 sentences, evaluated once, 0.75 (PR #8) |
| C-4 | Score against the reference netlists | Done (PR #10); runs in the Colab downstream step |
| P-4, P-5 | Training, test, downstream, evidence zip | Done 2026-11-04 (early stop at epoch 73, best 43) |
| C-2 | `EXPERIMENTS.md` and the docs from the zip | Done: results section; key files in `results/baseline_v1/` |
| C-3 | Error analysis of detector and downstream failures | Done in EXPERIMENTS (battery terminals, wire gaps, label text as values, detector false positives) |
| T-5 | Check the report's numbers against `results/baseline_v1/` | Next: Pawan |
| T-6 | Re-run `report_tables` from the zip on a clean machine | Next: someone other than the person who ran the last Colab session |

## Next actions

| # | Who | Action |
|---|---|---|
| 1 | Team | Fill the report's PENDING values by hand from [`results/baseline_v1/tables.md`](../results/baseline_v1/tables.md) and EXPERIMENTS (Results: baseline_v1), including the early stop, the hardware note and the analog-photo base. |
| 2 | Pawan | T-5: check every report number against `results/baseline_v1/`. |
| 3 | Pawan or Pranshu | T-6: `python -m circuitvision.report_tables --exp <unzipped evidence> --out tables.md` on a clean machine; it must match `results/baseline_v1/tables.md`. |
| 4 | Pawan | Demo prep per [`DEMO.md`](DEMO.md): download `best.pt` and `models/` from `MyDrive/CircuitVision/experiments/baseline_v1/`. |
| 5 | Pranshu | Submission date for the report cover. |

## Commands

```bash
pip install -e ".[detect,app,dev]" && bash scripts/pin_opencv.sh && pytest -q
python -m circuitvision.pipeline examples/divider.png --xml examples/divider.xml --simulate
python -m circuitvision.nlp.text2netlist "a 9V battery driving a 1k resistor in series with a 2k resistor"
python -m circuitvision.report_tables --exp <evidence folder> --out tables.md
```

## Links

- [README](../README.md) · [EXPERIMENTS](../EXPERIMENTS.md) · [Project Bible](PROJECT_BIBLE.md) · [Colab notebook](../notebooks/train_colab.ipynb) · [Synthetic results tables](../results/tables.md)
- Report: Word file built from [`report/`](../report/README.md) (submission format). Earlier draft: https://claude.ai/artifact/J3GRtWxXebwtmNMUsKddFn (private)
- Slides: none will be presented (decided 2026-11-04). An earlier draft deck exists at https://claude.ai/artifact/1mx24Viff93aLYdbiYvkCD.
- Demo script: [`DEMO.md`](DEMO.md)

*Update this file whenever a significant task finishes. Update the Bible only when architecture, methodology, assumptions or project facts change.*
