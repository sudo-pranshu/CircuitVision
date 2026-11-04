# CircuitVision: Final Demo Script

A 6–8 minute live demo of the image path (Gradio app) and the text path (terminal). Everything except the trained weights is ready. The app was checked on 2026-11-04: it starts, serves on localhost and processes a photo end to end.

## Before the demo day (once)

1. **Get the weights from Drive.** The evidence zip leaves out weights and models, so download them from `MyDrive/CircuitVision/experiments/baseline_v1/`:
   - `runs/train/weights/best.pt` → save as `runs/baseline_v1/best.pt`
   - the `models/` folder (orientation classifiers) → save as `experiments/baseline_v1/models/`

   Both locations are ignored by Git, so the weights never get committed.
2. **Start the app once on the demo laptop** (command below) and upload one photo. The first run downloads the EasyOCR model, about 100 MB. Do this on good Wi-Fi, not during the demo.
3. **Draw the live circuit in advance as a backup:** a battery with two resistors in series, in dark pen on plain white paper, with clear corners and no values. Photograph it once to check it works.
4. **Take screenshots** of each step working, in case the laptop or Wi-Fi fails on the day.
5. **Rehearse** with the timings below.

## Launch

```bash
cd ~/Developer/CircuitVision
.venv/bin/python -m circuitvision.app --weights runs/baseline_v1/best.pt --orient experiments/baseline_v1/models
```

Open http://localhost:7860. Add `--no-ocr` if EasyOCR misbehaves, or `--share` to get a public link for a projector on another machine.

## Flow

| # | Time | Input | Show | Say |
|---|---|---|---|---|
| 1 | 1 min | — | The app's three panels | "Photo in; detected symbols, traced wires and the SPICE netlist out; then grammar validation and an ngspice operating point." |
| 2 | 1.5 min | `data/cghd_raw/drafter_12/images/C137_D1_P1.jpg` | Overlay, netlist, validation | This drawing comes from the test drafter, whose handwriting the model never saw in training. Compare the netlist with `references/drafter_12/C137_D1.cir`, which a person checked against the photo. C137 is the only test circuit with every part in scope. |
| 3 | 1 min | `.../C136_D1_P1.jpg` | The switch is masked, not drawn as a wire | Out-of-scope symbols (switches, AC sources, logic gates) are masked so their ink can't short two wires; that part of the circuit then can't be completed. This is a stated limitation. |
| 4 | 1.5 min | The live photo (step 3 above), taken on a phone and uploaded | Whatever it produces | Shows it works on a new drawing. If something is off, point at the overlay: wrong box, wrong net, or a missed junction, so each stage can be checked on its own. |
| 5 | 1.5 min | Terminal: the text path | Two sentences, below | Natural language to the same netlist format, through an HMM tagger and a series/parallel parser. |
| 6 | 1 min | — | The results table | Quote numbers only from `tables.md` in the evidence zip, with REAL and SYNTHETIC kept apart. |

### Text path (step 5)

```bash
.venv/bin/python -m circuitvision.nlp.text2netlist "a 9V battery driving a 1k resistor in series with a 2k resistor"
.venv/bin/python -m circuitvision.nlp.text2netlist "Two 220 ohm resistors are connected in series, and another 1k resistor is connected across both of them."
```

The first works. The second is from the teammate's test set and comes out wrong: it is read as series, not parallel. Say so openly. It is one of the five structural failures behind the 0.75 score on the set written without seeing the parser, against 0.90 on the earlier fresh set.

## Honest answers to likely questions

- **Real accuracy?** Only the numbers in the evidence zip's `tables.md`. Detection is measured on test drafter 12. Real topology is scored against the 12 reference netlists, and only C137 is complete enough to simulate.
- **Why are values defaults?** The drafter-12 drawings have no values written on them, only labels such as "Resistor". On a drawing with values, OCR attaches them.
- **Why not the paper's split?** The split was fixed before training (validation drafter 11, test drafter 12), so the numbers aren't directly comparable with Bayer *et al.* This is noted in the report.
- **Polarity?** The HOG + SVM orientation classifier uses the convention fixed at Gate 1 from the real rotation sheets, reviewed by a second person (issue #7).
