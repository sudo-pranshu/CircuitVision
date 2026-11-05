# CircuitVision: Project Status

Updated 2026-11-05 · `main` at `f241a05` · Full reference: [PROJECT_BIBLE.md](PROJECT_BIBLE.md)

## Summary

**The project is complete and its final report is ready for submission.** Every planned stage is built and tested, the real CGHD experiment has run, and the application works with the trained model.

- **Pipeline:** photo or English description → SPICE netlist → grammar and semantic validation → ngspice simulation.
- **Real experiment (baseline_v1):** trained and evaluated on CGHD release 14 on Colab, 2026-11-04. Results are in [EXPERIMENTS.md](../EXPERIMENTS.md#results-baseline_v1), and the files are in [`results/baseline_v1/`](../results/baseline_v1/).
- **Final report:** [`report/CircuitVision_Mini_Project_Report.pdf`](../report/CircuitVision_Mini_Project_Report.pdf) (46 pages, with the `.docx` beside it), built 2026-11-04 from the evidence files, with no pending values.
- **Application:** Gradio app with Photo, Text and Send-from-phone tabs, running the baseline_v1 model, tested with real user input ([USER_INPUT_TESTING.md](USER_INPUT_TESTING.md)).
- **Open:** two checks by people (T-5, T-6), the exact submission date on the cover, and optionally more user photos. See *Next actions*.

## Headline results

| REAL CGHD, test drafter 12 (unseen handwriting) | Result |
|---|---|
| Detection, 96 photos, 1,628 symbols | **mAP@0.5 0.904, mAP@0.5:0.95 0.611** (P 0.886, R 0.875); weakest class voltage source (AP@0.5 0.48) |
| Orientation (polarity) | classifier: voltage source 0.906 (32 crops), diode 0.775 (40); in the pipeline 0.833 (GT boxes), 0.639 (detector) |
| OCR on handwritten labels | 0.021 exact, but almost all labels are words that the value-only reader can't spell; only 4 are values, so value reading is effectively unmeasured |
| **Correct circuit**, 48 analog photos, against the T-3 references | **37.5% with ground-truth boxes, 10.4% with detector boxes** |

- **Training:** stopped early at **epoch 73 of 100** (patience 30). The best epoch, **43**, is the one evaluated. It ran on a T4 for epochs 1–33 and an A100 for 34–73, with the same config.
- **Scope of the test set:** half the test photos (C139–C144) are logic-gate diagrams with nothing to reconstruct.
- **Main failures:** battery terminals in wire tracing (16 of 32 dropped parts), faint wires, label text read as values, and the battery also detected as a capacitor.
- **Post-hoc fix:** a faint-wire fix made after these results raised the GT-box figure to 21 of 48 with no photo worse. It's reported separately, because it was measured after seeing the test results.

| Other results | Result | Label |
|---|---|---|
| Wire tracing, 11 photo conditions | 0.994 exact topology | SYNTHETIC |
| GT-box end to end, synthetic drafters 10–12 | 18/18 topology, exact and operating-point match | SYNTHETIC |
| Text → netlist, fresh 20 sentences (evaluated once) | 0.90 exact circuit | HAND-WRITTEN |
| Text → netlist, Pawan's 20 sentences (T-7, evaluated once) | 0.75 exact circuit, component F1 0.986 | HAND-WRITTEN |
| Text input, 20 typed descriptions | 12 of 15 with a definite answer electrically correct; invalid and ambiguous input reported | USER INPUT |
| User photos (2) | plain paper: correct topology (the "+5V" label is misread by OCR); ruled paper: fails | USER INPUT |
| Unit and regression tests | **139 pass** (`pytest -q`, macOS, 2026-11-05) | — |

## Deliverables

| Deliverable | State | Where |
|---|---|---|
| Final report | **Ready.** 46 pages, institute template. Contents: real results (§6.3), user testing (§6.4), Gate 1 sheets and seven Gate 2 overlays, training curves, confusion matrix. Generated from `report/content.js`, which includes Pranshu's layout and writing revision; numbers taken only from the evidence files. | `report/CircuitVision_Mini_Project_Report.pdf` / `.docx` |
| Code and tests | Complete; 139 tests | `src/`, `tests/` |
| Real-run evidence | Complete. Key files are in the repo; the full zip and weights are on Drive | `results/baseline_v1/`; Drive `MyDrive/CircuitVision/experiments/` |
| Live demo | Ready; weights and models are on Pawan's Mac | [`DEMO.md`](DEMO.md) |
| Slides | Not needed (decided 2026-11-04) | — |

## Application

- **Start:** `python -m circuitvision.app --weights runs/baseline_v1/best.pt --orient experiments/baseline_v1/models --lan`
- **Photo tab:** upload, webcam or clipboard. **📱 Upload from mobile** shows a QR code; a phone on the same Wi-Fi takes a photo, and the result appears on the laptop by itself.
- **Text tab:** a typed description gives a netlist, validation and a simulation. Any assumed value or source is reported.
- **Reports state uncertainty:** each part's confidence, parts left out, defaulted values, a guessed ground, a symbol detected under two labels, and every value read by OCR.

## Tasks

| ID | Task | State |
|---|---|---|
| T-1 | Reproduce the tests and synthetic results | Done 2026-11-02 |
| T-2 | Independent review of Gates 1 and 2 | Done: Pranshu agreed ([issue #7](https://github.com/sudo-pranshu/CircuitVision/issues/7)) |
| T-3 | Reference netlists for drafter 12 | Done: 12 files, drafted by Claude from the photos, checked by Pawan |
| T-4 | Literature review | Done: report Chapter 2, sources reviewed by Pawan |
| T-7 | New frozen NLP test set | Done: 20 sentences, evaluated once, 0.75 |
| P-1 to P-5 | Audit, gates, training, test, downstream, evidence | Done 2026-11-04 |
| C-2, C-3, C-4 | Results into the docs, error analysis, scoring against references | Done |
| C-5 | Fixes for what real data exposed | Done: faint-wire bridging, "+5V", unit words, reported assumptions, app warnings |
| Final report | Fill the real results; add the figures and user testing | Done 2026-11-04. The numbers were filled by Claude from the evidence at Pawan's request, rather than by hand as planned on 2026-11-04 |
| **T-5** | **Check every number in the report against `results/baseline_v1/`** | **Open (Pawan)** |
| **T-6** | **Re-run `report_tables` from the zip on a clean machine** | **Open (Pranshu, who didn't run the last session)** |

## Next actions

| # | Who | Action |
|---|---|---|
| 1 | Pawan | **T-5:** read §6.3 and §6.4 of the PDF against `results/baseline_v1/tables.md` and `EXPERIMENTS.md`, and note any mismatch. |
| 2 | Pranshu | **T-6:** `python -m circuitvision.report_tables --exp <unzipped evidence> --out tables.md` on a clean machine; it must match `results/baseline_v1/tables.md`. |
| 3 | Team | Confirm the exact submission date (the cover says "November 2026"), then rebuild with `bash report/build.sh`. |
| 4 | Optional | More user photos per `results/user_input_testing/images/README.md`: 2 of the 10 planned are done. |

## Known limitations

- Batteries: lowest detection AP; often also boxed as a capacitor; half of the dropped parts.
- Handwritten values are barely present in the test drafter; EasyOCR misreads some signs ("+5V" as "45V").
- Out-of-distribution drawings fail: ruled notebook paper, scanned figures, unusually large or outlined symbols, and AC sources (out of scope).
- The text path misses some sentence structures ("two parallel resistors of 2k and 4k", "a parallel combination of …", multi-branch descriptions).
- Not re-measured after the faint-wire fix: detector-mode downstream on drafter 12 (only the GT-box re-check was run).

## History in brief

| Date | Milestone |
|---|---|
| 2026-09-30 → 10-23 | Pipeline, NLP path, evaluation harness, restart-safe Colab notebook, project bible (PRs #1–#4) |
| 2026-11-02 | Teammate reproduction (T-1); full dataset audit; Gates 1 and 2 decided; label fixes (PR #5) |
| 2026-11-04 | T-2, T-3, T-4, T-7, C-4; demo script; sync audit; baseline_v1 trained and evaluated; results recorded (PRs #6–#15) |
| 2026-11-04 | User-input testing, app uncertainty reporting, phone upload, faint-wire fix, final report (PRs #16, #17) |

## Commands

```bash
pip install -e ".[detect,app,dev]" && bash scripts/pin_opencv.sh && pytest -q
python -m circuitvision.app --weights runs/baseline_v1/best.pt --orient experiments/baseline_v1/models --lan
python -m circuitvision.nlp.text2netlist "a 9V battery driving a 1k resistor in series with a 2k resistor" --simulate
python -m circuitvision.report_tables --exp <evidence folder> --out tables.md
bash report/build.sh        # rebuild the report (needs LibreOffice, poppler, npm docx)
```

## Links

- [README](../README.md) · [EXPERIMENTS](../EXPERIMENTS.md) · [Project Bible](PROJECT_BIBLE.md) · [User-input testing](USER_INPUT_TESTING.md) · [Demo script](DEMO.md) · [Colab notebook](../notebooks/train_colab.ipynb)
- Results: [`results/baseline_v1/tables.md`](../results/baseline_v1/tables.md) (REAL) · [`results/tables.md`](../results/tables.md) (SYNTHETIC and HAND-WRITTEN)

*Update this file whenever a significant task finishes. Update the Bible only when architecture, methodology, assumptions or project facts change.*
