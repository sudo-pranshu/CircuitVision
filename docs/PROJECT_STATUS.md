# CircuitVision: Project Status

Updated 2026-11-02 · `main` · Full reference: [PROJECT_BIBLE.md](PROJECT_BIBLE.md)

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

The submission report (Word, institute template, 47 pages) is built from [`report/`](../report/README.md) with `bash report/build.sh`. Every real-CGHD value is highlighted `PENDING: real CGHD run` in Section 6.3 and a few sentences; the cover's date of submission is also a highlighted placeholder. After the Colab run, Claude replaces the pending values from the evidence files and rebuilds. The report still gives the earlier count of 107 tests, in the text and in the per-file test table; update these at the same time.

## Tested

| What | Result | Label |
|---|---|---|
| Unit and regression tests | 110 pass (`pytest -q`, macOS, ngspice present) | — |
| Wire tracing, 11 photo conditions | 0.994 exact topology | SYNTHETIC |
| Detector-mode end to end (synthetic YOLO, 6 images) | 0.83 topology, 0.67 exact/op match | SYNTHETIC |
| Text → netlist, fresh 20 sentences (evaluated once) | 0.90 exact circuit | HAND-WRITTEN |
| Training kill + resume (`train.py --auto`) | Resumed at epoch 2, then detected as finished | Real ultralytics, CPU |

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
| 1 | Whoever runs Colab | Colab → open `notebooks/train_colab.ipynb` from GitHub `main` → T4 GPU → **Run all**. At Gate 1, set `PLUS_AT_0 = 'left'` and `CGHD_CLOCKWISE = False`. |
| 2 | Whoever runs Colab | Run all → confirm the 24 overlays look right → set Gate 2 → Run all (training takes about 3–5 h; after a disconnect, Run all again). |
| 3 | Whoever runs Colab → Claude | Send `baseline_v1_evidence.zip` (also saved to `MyDrive/CircuitVision/experiments/`). Claude fills the report, slides and EXPERIMENTS.md from it. |
| 4 | Done | Independent review of the Gate 1 and Gate 2 evidence (T-2): Pranshu agreed on 2026-11-04 ([issue #7](https://github.com/sudo-pranshu/CircuitVision/issues/7)). |
| 5 | Teammate | Write the drafter-12 reference netlists (T-3). After step 3, check the report's numbers against `results/` and the evidence zip (T-5). |

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
- Slides: https://claude.ai/artifact/1mx24Viff93aLYdbiYvkCD (private until Pranshu shares it)

*Update this file whenever a significant task finishes. Update the Bible only when architecture, methodology, assumptions or project facts change.*
