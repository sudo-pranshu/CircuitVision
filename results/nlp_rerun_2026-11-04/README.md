# NLP re-run after the 2026-11-04 fixes (HAND-WRITTEN)

Not the headline results. The once-evaluated results stay in `results/nlp_eval_fresh.*` and `results/nlp_eval_pawan.*`.

Two lexical fixes were made on 2026-11-04, both motivated by the two failures of the fresh 20-sentence set:
- "330R" (BS 1852 notation for 330 ohm) is now read as a value, in the text path and in the image path (`netlist.parse_value`), where a handwritten "330R" label was previously ignored.
- "a pair of 10k resistors" now gives two resistors.

The error classifier was also fixed: a final full stop, which the tokenizer drops by design, no longer makes a failure count as "tokenization". Scores are unaffected.

| Set | Exact circuit (recorded) | Exact circuit (this re-run) | Note |
|---|---|---|---|
| 30-sentence (`eval`) | 1.000 (after Oct changes, optimistic) | 1.000 | No regression |
| Fresh 20 (`fresh`) | 0.900 (evaluated once) | 1.000 | **Optimistic**: the fixes target this set's two failures |
| Pawan's 20 (`pawan`) | 0.750 (evaluated once) | 0.750 | Unchanged; identical per-sentence output. Error types now 2 tagging, 1 structure, 2 components/values (were all labelled "tokenization") |

Reproduce: `python -m circuitvision.nlp.evaluate --set src/circuitvision/nlp/data/handwritten_<set>.jsonl --out results/nlp_rerun_2026-11-04/<set>`
