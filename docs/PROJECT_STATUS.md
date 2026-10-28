# CircuitVision: Project Status

Updated 2026-10-28 · `main` · Full reference: [PROJECT_BIBLE.md](PROJECT_BIBLE.md)

## Summary

The pipeline (image → netlist → validation → simulation, plus text → netlist) is **implemented, tested and verified on synthetic data**.

**The real CGHD experiment (baseline_v1) has not run yet**, so no real-photo metric exists. The project is **not complete** until it runs.

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

The submission report (Word, institute template, 47 pages) is built from [`report/`](../report/README.md) with `bash report/build.sh`. Every real-CGHD value is highlighted `PENDING: real CGHD run` in Section 6.3 and a few sentences; the cover's date of submission is also a highlighted placeholder. After the Colab run, Claude replaces the pending values from the evidence files and rebuilds.

## Tested

| What | Result | Label |
|---|---|---|
| Unit and regression tests | 107 pass (`pytest -q`) | — |
| Wire tracing, 11 photo conditions | 0.994 exact topology | SYNTHETIC |
| Detector-mode end to end (synthetic YOLO, 6 images) | 0.83 topology, 0.67 exact/op match | SYNTHETIC |
| Text → netlist, fresh 20 sentences (evaluated once) | 0.90 exact circuit | HAND-WRITTEN |
| Training kill + resume (`train.py --auto`) | Resumed at epoch 2, then detected as finished | Real ultralytics, CPU |

## Not tested

- The full CGHD audit and the rotation convention (Gate 1).
- Label conversion on real data (Gate 2).
- YOLO detection, orientation, OCR and topology on real photos.
- The Gradio demo with real weights.

## Next actions

| # | Who | Action |
|---|---|---|
| 1 | Pranshu | Colab → open `notebooks/train_colab.ipynb` from GitHub `main` → T4 GPU → **Run all**. It stops at Gate 1. |
| 2 | Pranshu → Claude | Send `audit/audit.md`, `audit/dataset_docs/` and `rotation_check/*.png` from Drive. Claude recommends the convention; Pranshu sets Gate 1. |
| 3 | Pranshu | Run all → inspect the 24 overlays → set Gate 2 → Run all (training takes about 3–5 h; after a disconnect, Run all again). |
| 4 | Pranshu → Claude | Send `baseline_v1_evidence.zip` (also saved to `MyDrive/CircuitVision/experiments/`). Claude fills the report, slides and EXPERIMENTS.md from it. |
| 5 | Teammate | Onboarding (Bible §15), then tasks T-1, T-2 and T-3 (Bible §11.3). Proof-read `report/content.js` against `results/` (T-5). |

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
