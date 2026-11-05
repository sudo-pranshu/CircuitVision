# User-Input Testing

Does CircuitVision work when someone who doesn't know its internals types a circuit description or uploads a hand-drawn circuit? This records what was actually tested on 2026-11-04, and what was not.

**Summary.**
- **Text input:** tested on 20 typed descriptions through the real CLI and the app. 12 of the 15 with a definite answer give the electrically correct circuit. Three sentence structures still give a wrong circuit. Unclear or invalid input is now reported instead of being silently guessed.
- **Image input:** **2 real user photos tested** (section 4). A hand-drawn circuit on plain paper is reconstructed with the correct topology after the faint-wire fix; one on ruled notebook paper fails. The real model was also run through the app on two CGHD test photos as a sanity check.

Results here are kept apart from the CGHD benchmark (`results/baseline_v1/`) and the synthetic experiments.

## 1. Method

- **Text.** Each description was run as a user would run it: `python -m circuitvision.nlp.text2netlist "<text>"` in a fresh process. The case with an LED was also typed into the app's Text tab in a browser. For each case, a reference netlist was written by hand before running, and the expected currents and voltages were worked out by hand. The output was then compared with the reference: graph isomorphism (`compare.py`), exact values, and ngspice operating points within 0.1%. Cases with no single right answer (missing, ambiguous or invalid input) were judged on whether the user is told what happened. All 20 were run before and after the fixes.
- **Image.** The app's image handler (`app.build_ui(...).process`, the same code the Photo tab calls) runs through `scripts/run_user_image_tests.py`. It saves the overlay, netlist and report, and compares each result with a hand-written reference. The runner and the 10-case plan are ready (`results/user_input_testing/images/README.md`), but no user photos exist yet.
- **Baseline evidence.** `baseline_v1_evidence.zip` (SHA-256 `b95fbf15…`) was read directly, without modifying it: `tables.md`, `test/metrics.json`, the confusion matrix, `per_image.csv`, `downstream/summary.json`, the CSVs and netlists, `orientation.json`, the logs and the config. See section 7.

## 2. Environment

macOS 27.0.1 on Apple Silicon (arm64), Python 3.13.18, ngspice 47, gradio 6.31.0, ultralytics 8.4.184, torch 2.15.0. The code is on `feature/Pawan`, based on `main` at `5d5c9d2` plus the changes in section 5. Weights: `best.pt` from baseline_v1 (epoch 43), copied from Drive to `runs/baseline_v1/`, which Git ignores; orientation models in `experiments/baseline_v1/models/`.

## 3. Text input: cases and results

Outputs are in [`results/user_input_testing/text/`](../results/user_input_testing/text/), with one transcript per case before and after the fixes, plus the case list with references (`cases.py`).

| # | Input (as typed) | Expected | After fixes | Before fixes |
|---|---|---|---|---|
| T01 | A 9V battery connected to a 1k resistor and a 2k resistor in series. | V(2) = 6 V, 3 mA | ✅ correct | ✅ |
| T02 | A 12V source connected to two parallel resistors of 2k and 4k. | 2k ∥ 4k, 9 mA | ❌ built in **series** (2 mA) | ❌ |
| T03 | A 5V source connected to a 1k resistor in series with a parallel combination of 2k and 3k resistors. | V(2) = 2.727 V | ❌ 1k across the source, 2k and 3k in series | ❌ |
| T04 | A diode connected in series with a 470 ohm resistor and a 5V DC source. | diode drop of about 0.7 V | ✅ | ✅ |
| T05 | A 10V supply feeds a 100 ohm resistor, followed by two branches in parallel: one with … 220 and 330 in series, … the other with a single 1k. | V(2) = 7.80 V | ❌ all four in a wrong chain | ❌ |
| T06 | Put a 4k7 and a 10k resistor in parallel across a 3.3V battery | 1.032 mA | ✅ | ✅ |
| T07 | 12 volts across three 1k resistors in series | V(2) = 8, V(3) = 4 | ✅ | ✅ |
| T08 | 2.2kΩ resistor in series with 0.1uF cap, powered by 6 V | V(2) = 6 V (capacitor open at DC) | ✅ | ✅ |
| T09 | a 1M5 resistor and a 100n capacitor in parallel | no source given | ⚠️ builds it, **with a note** that a 5 V source was added | ⚠️ 5 V source added **silently** |
| T10 | hook up a 9 volt battery to an LED and a 330 ohm resistor | LED in series with 330 Ω | ✅ | ✅ |
| T11 | a resistor in series with a capacitor | no values, no source | ⚠️ defaults used, **with notes** | ⚠️ silent defaults |
| T12 | a 5V source with some resistors | ambiguous | ⚠️ one 1k resistor, **with notes** (plural without a number; default value) | ⚠️ silent |
| T13 | make me a circuit | invalid | ✅ clear message: "No circuit components were found…" | ❌ Python traceback |
| T14 | a 9V battery connected to | incomplete | ✅ same clear message | ❌ Python traceback |
| T15 | resistors of 1k, 2k and 3k all in parallel with a 9V battery | 16.5 mA | ✅ | ✅ |
| T16 | a 12V battery in series with a 1000 ohm resistor and a 1 kohm resistor | V(2) = 6 V | ✅ | ✅ |
| T17 | A 24 V supply drives a 2 mH inductor in series with a 47 Ω resistor | 0.511 A (inductor short at DC) | ✅ | ✅ |
| T18 | TWO 220R RESISTORS IN SERIES WITH A 9V BATTERY | V(2) = 4.5 V | ✅ | ✅ |
| T19 | a pair of 1k resistors in parallel, in series with a 2k resistor, all across 5V | V(2) = 4 V | ✅ | ✅ |
| T20 | a 9V battery and a resistor of 2 megohms in series | 4.5 µA | ✅ | ❌ **2 Ω plus a phantom capacitor** |

**Totals (after fixes):**
- 15 cases have a definite circuit: **12 correct** (topology, values and simulated operating point all match), **3 wrong**.
- 5 have no definite circuit:
    - 2 are rejected with a clear message;
    - 3 are built from stated assumptions, with every assumption reported.

Before the fixes: 11 of 15 correct, 3 silent guesses and 2 crashes.

**Electrical correctness.** For the 12 correct cases, the node voltages and source current from ngspice match the hand calculations to within 0.1%:
- the divider at 6 V;
- 12 V across three equal resistors at 8 V and 4 V;
- the diode drop of 0.71 V in T04 (V(2) = 4.29 V);
- 0.511 A through the inductor circuit.

Source polarity is always + on node 1, and ground is the − terminal. **The 3 wrong cases are valid netlists that simulate without error,** so a user can't tell from validation that they're wrong. Only the circuit itself shows it.

## 4. Image input

| What | Status |
|---|---|
| Real user photos (hand-drawn, photographed) | **2 tested** (`results/user_input_testing/images/`): U01 on plain paper is reconstructed with the correct topology after the faint-wire fix ("+5V" misread as "45V" by OCR); U02 on ruled notebook paper fails, because the ruling is traced as wires and the AC source is out of scope. A scanned figure from a published paper was also tried and failed (AC read as DC, diode missed); it is not stored, because it is third-party material. The 10-case plan remains open. |
| App with the real model, CGHD test photos (sanity check, **not** user input) | 2 photos, [`results/user_input_testing/app_sanity_cghd/`](../results/user_input_testing/app_sanity_cghd/) |

**Sanity check.** These are drafter-12 benchmark photos, so they don't count as user tests.
- **C137_D1_P1:** the app's netlist is **identical** to the Colab evidence for this photo, so the real model and pipeline reproduce on this Mac. The netlist is valid but wrong: the battery is also detected as a capacitor (confidence 0.75 against 0.61 for the voltage source, with the boxes overlapping at IoU 0.99). The report now says so.
- **C136_D1_P1:** the battery is left out, because only 1 wire end was found at it (the switch beside it is out of scope). The report says so.

## 5. Fixes implemented

Each fix is general, has a regression test, and was checked against all previous behaviour (section 6).

| Fix | Where | Why |
|---|---|---|
| "megohm(s)", "megaohm", "MΩ", "kilohm(s)" and "kiloohm" read as multipliers | `nlp/lexicon.py` (`tokenize`) | "2 megohms" became 2 Ω plus a phantom capacitor; "4.7 kilohm" became 4.7 Ω. Both wrong silently. |
| Assumptions reported as notes: default source, missing values, plural without a count | `nlp/text2netlist.py` (`text_to_netlist_with_notes`, `assumptions`) | Guesses were presented as the user's circuit. |
| A description with no component gives a clear message, not a traceback | `text2netlist` CLI | T13 and T14 crashed. |
| `--simulate` on the text CLI | `text2netlist` CLI | Parity with the image CLI. |
| App **Text tab**; the app starts without weights (text only) | `app.py` | Text input had no UI, and the app couldn't start without trained weights. |
| Photo report: each part's confidence, uncertain parts flagged, parts left out, defaulted values, guessed ground, **one symbol with two labels**, and validation errors explained | `app.py` (`image_report`) | A wrong reconstruction was shown with no sign of doubt. |
| Overlay labels scale with the image, on a white background | `pipeline.overlay` | Labels were unreadable on full-size phone photos. |
| Image test runner | `scripts/run_user_image_tests.py` | Scores real photos through the app's own handler. |
| Straight-line gap bridging for faint wires | `topology.extract_nets` | User photo U01 split one wire at a faint pen stroke, leaving the netlist invalid; now correct. Post-hoc GT-box re-check on drafter 12: 18 → 21 of 48, none worse. |
| "+5V" accepted; every OCR-read value listed in the report | `netlist.parse_value`, `ocr.py`, `app.py` | EasyOCR reads the handwritten + as 4 ("45V"), which can't be fixed safely, so the user is shown each value to check. |
| Phone upload: QR code, Send from phone, result shown on the laptop | `app.py` (`--lan`) | Lets a user photograph a drawing with a phone during a demo. |

**Not changed:** the baseline config, the model, the label conversion and the parser's structure rules. The three structural text failures (T02, T03, T05) would need new parser rules. Writing them to fit these sentences and then reporting the same sentences as passing would be tuning to the test.

## 6. Regression testing

- `pytest -q`: **139 passed** (129 before this work; the new tests cover the text fixes, the app reports, phone upload and faint-wire bridging).
- **The frozen NLP sets give per-sentence output identical** to their recorded runs, so the tokenizer change affects none of them:
    - 30-sentence set: 1.000;
    - fresh set: 1.000 (re-run after the audit fixes);
    - Pawan's set: 0.750.
- The synthetic divider example is unchanged: v(2) = 0.4545 V.
- The real model's netlist for C137_D1_P1 is unchanged: identical to the Colab evidence.

## 7. What the baseline evidence says about real users

| Reported weakness | Cause (from the evidence) | Effect on a user |
|---|---|---|
| Detection mAP50 0.904 overall | Strong on resistors, crossovers and text (AP50 ≈ 0.99) | Most symbols are found |
| **Voltage source AP50 0.482** | The confusion matrix has 4 voltage sources predicted as capacitors (9 at confidence 0.25, `test/metrics.json`), and in C137 the same box is labelled both. The two-plate symbols look alike, and voltage sources are rare in training (836 crops in the orientation set, against 5,013 diodes). | **The battery is often doubled or missed. Every circuit needs one, so this hits nearly every user.** The report now flags the double label. |
| **OCR exact 0.021** | `ocr.py` restricts EasyOCR to value characters, so word labels ("Resistor", "Battery") can't be read and come out as "Ruv6r" or "Gfu". Of 656 test labels, only 4 are values. | **The 2% measures labels the reader was never meant to read.** Real value reading on handwriting is **unmeasured**. The user tests should include written values. |
| **Detector exact netlist match 0.062** (3 of 48 analog photos) | Wire tracing drops parts whose terminal count isn't 2; the battery causes 16 of the 32 drops. Detector false positives add parts. Text labels are attached as values. | Expect partial netlists. The report now lists every dropped part and defaulted value instead of presenting the result as certain. |
| 48 of the 96 test photos can't produce a netlist | C139–C144 are logic-gate diagrams | Out-of-scope drawings get an empty or partial netlist; the Photo tab now says which symbols are supported. |

## 8. Failure analysis by stage

| Stage | Failure | Source | Status |
|---|---|---|---|
| NLP parsing | "two parallel resistors of 2k and 4k", "in series with a parallel combination of …", and two-branch descriptions | Algorithm: the parser covers chain and list forms only | Open, a known limitation |
| NLP lexicon | spelled-out multipliers | Algorithm | **Fixed** |
| NLP output | silent defaults and invented source | UX | **Fixed** (notes) |
| CLI | traceback on empty input | UX | **Fixed** |
| Detection | battery read as a capacitor (same box) | Model: similar symbols, few training examples | Flagged in the report; the model is unchanged |
| Wire tracing | battery with 1 or 3 wire ends; wire gaps | Algorithm on real photos | Flagged in the report; open |
| OCR | handwritten words unreadable, values untested | By design (value allowlist) plus missing test data | Open; needs photos with written values |

## 9. Remaining limitations and blockers

1. **Only 2 user photos have been tested** (1 correct topology, 1 failure on ruled paper). The 10-case plan in `results/user_input_testing/images/README.md` is 2/10 done.
2. The parser misses three common sentence structures (T02, T03, T05). They produce **valid but wrong** circuits with no warning.
3. Battery detection and battery wire tracing are the weakest points of the image path.
4. Reading handwritten values hasn't been measured on real data.
