# CircuitVision: Project Status

Updated 2026-11-04 (sync audit) · `main` · Full reference: [PROJECT_BIBLE.md](PROJECT_BIBLE.md)

## Summary

The pipeline (image → netlist → validation → simulation, plus text → netlist) is **implemented, tested and verified on synthetic data**.

**The real CGHD experiment (baseline_v1) has not run yet**, so no baseline metric exists. The project is **not complete** until it runs. The real dataset has been audited, Gates 1 and 2 are decided, and the label fixes are on `main` (PR #5, merged 2026-11-02), so the run is ready to start on Colab; see [EXPERIMENTS.md](../EXPERIMENTS.md#full-dataset-checks-2026-11-02-md5-checked-v14).

## What works

- Every stage runs end to end on CPU: binarize → detect → orientation → wire tracing → OCR → netlist → CFG/Earley + semantic checks → ngspice.
- Text → netlist with an HMM tagger and a series/parallel parser, including nesting, counts and RKM values.
- The Colab notebook is restart-safe:
  - stage status is printed at the start;
  - finished stages are skipped;
  - training resumes automatically;
  - gate approvals are saved;
  - Gate 2 overlays cover every split, every class and the EXIF re-encoded photos.

## Final report

The submission report (Word, institute template with 0.75 in margins, 35 pages) is built from [`report/`](../report/README.md) with `bash report/build.sh`. Updated 2026-11-04 with the Gate 1 and 2 decisions, the full-dataset findings, the new defects, the NLP re-run and 129 tests. Every real-CGHD value is highlighted `PENDING: real CGHD run` (§6.3 and a few sentences); the cover's date of submission is also a highlighted placeholder. After the Colab run, the team fills in the pending values by hand from the evidence files (decided 2026-11-04). There is no slide deck.

## Tested

| What | Result | Label |
|---|---|---|
| Unit and regression tests | 129 pass (`pytest -q`, Linux, Python 3.13, ngspice 42, 2026-11-04) | — |
| GT-box end to end, synthetic drafters 10–12, re-run after PRs #5–#11 | 18/18 topology, exact and op match; OCR 0.914 / 0.950; orientation 18/18 | SYNTHETIC |
| Wire tracing, 11 photo conditions | 0.994 exact topology | SYNTHETIC |
| Detector-mode end to end (synthetic YOLO, 6 images) | 0.83 topology, 0.67 exact/op match | SYNTHETIC |
| Text → netlist, fresh 20 sentences (evaluated once) | 0.90 exact circuit | HAND-WRITTEN |
| Text → netlist, Pawan's 20 sentences (T-7, evaluated once) | 0.75 exact circuit, component F1 0.986 | HAND-WRITTEN |
| Text → netlist re-run after the 2026-11-04 fixes | Fresh 1.00 (optimistic), 30-sentence 1.00, Pawan's 0.75 unchanged | HAND-WRITTEN |
| Training kill + resume (`train.py --auto`) | Resumed at epoch 2, then detected as finished | Real ultralytics, CPU |

## Sync audit (2026-11-04)

All of Pawan's work (PRs #5–#11) is on `main`, and no branch has unmerged commits. It was reviewed and kept as is. Fixes found during the review are in `5137acd`:
- a malformed `voltage.dc` rotation no longer crashes `parse_voc`;
- "330R" values are read in the image path as well as the text path;
- "a pair of" counts as two;
- NLP failures are no longer mislabelled "tokenization" because of a final full stop;
- Gate 1 is prefilled in the notebook;
- the convention is labelled as the Gate 1 decision;
- Pawan's set is in `results/tables.md`.

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

- YOLO detection, orientation, OCR and topology on real photos.
- The Gradio demo with real weights.

## Next actions

| # | Who | Action |
|---|---|---|
| 1 | Whoever runs Colab | Colab → open `notebooks/train_colab.ipynb` from GitHub `main` → T4 GPU → **Run all**. At Gate 1 the decided values (`left`, `False`) are prefilled: check the sheets, then set `GATE1_APPROVED = True`. |
| 2 | Whoever runs Colab | Run all → confirm the 24 overlays look right → set Gate 2 → Run all (training takes about 3–5 h; after a disconnect, Run all again). |
| 3 | Whoever runs Colab → Claude | Send `baseline_v1_evidence.zip` (also saved to `MyDrive/CircuitVision/experiments/`). The team fills the report by hand; Claude updates EXPERIMENTS.md and the docs from it. Demo: [`DEMO.md`](DEMO.md). |
| 4 | Done | Independent review of the Gate 1 and Gate 2 evidence (T-2): Pranshu agreed on 2026-11-04 ([issue #7](https://github.com/sudo-pranshu/CircuitVision/issues/7)). |
| 5 | Teammate | After step 3, check the report's numbers against `results/` and the evidence zip (T-5). T-3 is done: 12 reference netlists in [`references/`](../references/README.md), drafted by Claude from the photos and checked by Pawan; C-4 scores against them in the Colab run. T-4 is done: the related-work section is report Chapter 2, with sources reviewed by Pawan. |

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
