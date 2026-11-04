# CircuitVision: Project Status

Updated 2026-11-04 · `main` · Full reference: [PROJECT_BIBLE.md](PROJECT_BIBLE.md)

## Summary

The pipeline (image → netlist → validation → simulation, plus text → netlist) is **implemented, tested and verified on synthetic data**.

**The real CGHD experiment (baseline_v1) is training on Colab now.** No test metric exists yet, and the project is **not complete** until the test and downstream evaluations finish and the evidence zip is in. Everything that doesn't need those results is done: both gates were decided and independently reviewed, the reference netlists exist, scoring against them is merged, and the demo is ready except for the weights.

## baseline_v1 on Colab

| Stage | State |
|---|---|
| Dataset | CGHD v14 cached on Drive, MD5-checked |
| Gate 1 (rotation convention) | Approved: `PLUS_AT_0='left'`, `CGHD_CLOCKWISE=False`; `voltage.dc` corrected in `parse_voc` |
| Gate 2 (label overlays) | Approved; the conversion matches the approved one on every resume |
| Training (YOLOv8s, 100 epochs, T4) | **In progress.** Last epoch seen: 20/100, at about 4.5 min per epoch, so about 6 h of training left. It may finish in a second Colab account through the shared Drive folder; that resumes the same run. |
| Test (drafter 12), downstream, tables, evidence zip | Run automatically after training |

At epoch 20, the validation check on drafter 11 gave mAP50 0.925 and mAP50-95 0.600. This is **training monitoring only, not a result**: reportable numbers come only from the drafter-12 test in the evidence zip.

When the zip arrives, confirm that `commit.txt` names a commit on `main`. Between 11:45 and 11:55 IST on 2026-11-04, `main` briefly held the sync-audit commits (see below). A session started in that window would have recorded them, though they don't change training or the label conversion.

## Sync audit by Pranshu (2026-11-04): not on `main` yet

Pranshu reviewed PRs #5–#11 and kept them as they are. His fixes are on `feature/sync-audit-nov4` (`5137acd`, plus docs in `93a7bea`); he pushed them to `main` and then reverted them (`ba0ee7e`), so `main` is unchanged. The plan is to cherry-pick `5137acd` after this status update.

| Fix | Effect on the running training |
|---|---|
| A malformed `voltage.dc` rotation no longer crashes `parse_voc` | None. CGHD v14 has no malformed values: 310 `voltage.dc` objects, of which 128 have no rotation, which is already handled. The label conversion is identical, so Gate 2 stays valid. |
| "330R" is read as 330 Ω in the image and text paths | None on training. In the image path it only affects OCR value attachment, and the drafter-12 drawings have no written values. |
| "a pair of" counts as two | None (text path only) |
| NLP failures no longer mislabelled "tokenization" | Labels only; Pawan's set stays 0.75, now 2 tagging, 1 structure and 2 components/values |
| Gate 1 values prefilled in the notebook | None: Gate 1 is already saved on Drive |
| Convention labelled as the Gate 1 decision; Pawan's set added to `results/tables.md` | None |

The NLP fixes target the fresh set's two failures. **The fresh set's recorded 0.90 therefore stays the headline**, and its 1.00 re-run is optimistic (`results/nlp_rerun_2026-11-04/` on that branch).

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

The submission report (Word, institute template) is built from [`report/`](../report/README.md). The team fills in the real-CGHD values by hand from the evidence files (decided 2026-11-04). Every such value is highlighted `PENDING: real CGHD run` in Section 6.3 and a few sentences, and the cover's date of submission is a placeholder. PR #10 updated the source for 125 tests, the T-7 NLP result and the T-3 references; the `.docx` hasn't been rebuilt since. There is no slide deck.

## Tested

| What | Result | Label |
|---|---|---|
| Unit and regression tests | 125 pass (`pytest -q`, macOS, ngspice present, `main`, 2026-11-04) | — |
| Wire tracing, 11 photo conditions | 0.994 exact topology | SYNTHETIC |
| Detector-mode end to end (synthetic YOLO, 6 images) | 0.83 topology, 0.67 exact/op match | SYNTHETIC |
| Text → netlist, fresh 20 sentences (evaluated once) | 0.90 exact circuit | HAND-WRITTEN |
| Text → netlist, Pawan's 20 sentences (T-7, evaluated once) | 0.75 exact circuit, component F1 0.986 | HAND-WRITTEN |
| Training kill + resume (`train.py --auto`) | Resumed at epoch 2, then detected as finished | Real ultralytics, CPU |
| Training resume on Colab | Resumed across sessions with the saved gates | Real, T4 |
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

- YOLO detection, orientation, OCR and topology on real photos (waiting on the run).
- The Gradio demo with real weights.

## Tasks

| ID | Task | State |
|---|---|---|
| T-1 | Reproduce the tests and synthetic results | Done 2026-11-02 |
| T-2 | Independent review of Gates 1 and 2 | Done: Pranshu agreed ([issue #7](https://github.com/sudo-pranshu/CircuitVision/issues/7)) |
| T-3 | Reference netlists for drafter 12 | Done: 12 files, drafted by Claude from the photos, checked by Pawan (PR #9) |
| T-4 | Literature review | Done: report Chapter 2, sources reviewed by Pawan |
| T-7 | New frozen NLP test set | Done: 20 sentences, evaluated once, 0.75 (PR #8) |
| C-4 | Score against the reference netlists | Done (PR #10); runs in the Colab downstream step |
| P-4, P-5 | Finish training, then test, downstream and evidence zip | In progress |
| C-2 | Update `EXPERIMENTS.md` and the docs from the zip | Waits on the zip |
| T-5 | Check the report's numbers against the zip | Waits on the zip |
| T-6 | Re-run `report_tables` from the zip on a clean machine | Waits on the zip; someone other than the Colab runner |

## Next actions

| # | Who | Action |
|---|---|---|
| 1 | Whoever runs Colab | Keep training running. After any disconnect: open the notebook from GitHub `main`, pick a T4 and **Run all**. Only one session at a time on the shared folder. |
| 2 | Pawan | Cherry-pick Pranshu's `5137acd` onto `feature/Pawan` and open a PR (safe for the running training; see above). |
| 3 | Whoever runs Colab → Claude | Send `baseline_v1_evidence.zip` (a copy is saved to `MyDrive/CircuitVision/experiments/`). Claude updates `EXPERIMENTS.md` and the docs; the team fills the report by hand. |
| 4 | Pawan | T-5 against the zip; demo prep per [`DEMO.md`](DEMO.md) (download `best.pt` and `models/` from Drive). |
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
