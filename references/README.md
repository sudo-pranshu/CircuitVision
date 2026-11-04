# Reference netlists for drafter 12 (T-3)

Hand-written netlists for the test drawings, so real topology can be scored against a known answer (C-4). CGHD v14 has no circuit graphs, so these files are the only real topology ground truth.

## Which drawings

Drafter 12 has 12 circuits, each drawn twice (`D1`, `D2`) and photographed four times (`P1`–`P4`).

| Drawings | In-scope parts | Reference? |
|---|---|---|
| `C133`–`C138`, `D1` and `D2` | 2–5 each; most also have one switch or AC source | **Yes: 12 files** |
| `C139`–`C144`, `D1` and `D2` | none (logic gates and terminals) | No: nothing a two-terminal pipeline can rebuild |

Only `C137` has no out-of-scope parts. That makes it the only circuit where a complete, simulable reference exists.

## Procedure

1. Open the `P1` photo of the drawing, from `data/cghd_raw/drafter_12/images/` (not in Git). Check `P2`–`P4` only if something is unclear.
2. Write the netlist **from the drawing alone**. Don't run the pipeline on it or look at its output first; otherwise the reference just copies the pipeline's mistakes.
3. Save it as `references/drafter_12/<circuit>_<drawing>.cir`, for example `C137_D1.cir`. One file covers all four photos of that drawing.
4. Write `D2` from its own photo, even when it looks like `D1`. Values or wiring can differ.
5. Run `pytest -q tests/test_references.py`.

## Rules

Write what a perfect two-terminal pipeline should produce from the drawing.

- **First line:** `* <circuit>_<drawing>` as a title, then `* by <your name>, <date>`.
- **In-scope parts:** resistor (`R`), capacitor (`C`), inductor (`L`), DC source or battery (`V`), diode, LED or Zener (`D`).
- **Values:** as written on the drawing, in SPICE form: `4k7` → `4.7k`, `10µF` → `10u`, `9V` → `DC 9`. With no value written, use the default: `1k`, `1u`, `1m`, `DC 5`.
- **Polarity:** `V` and `D` list the + or anode node first: `V1 1 0 DC 9`, `D1 3 4 DMOD`. Add `.model DMOD D` if there is a diode.
- **Nodes:** each wire joining parts is one node. A junction dot joins wires; a crossing without a dot doesn't. Node `0` is the − terminal of the DC source. With no DC source, pick any node as `0`, because comparison doesn't depend on which node is ground.
- **Out-of-scope parts** (switch, AC source and the like): leave them out, and keep their two sides as **separate nodes**. The pipeline masks them the same way. Record each one in a comment, for example `* omitted: switch between nodes 2 and 3`.
- **Last two lines:** `.op` and `.end`.

## What the check enforces

`tests/test_references.py` runs `grammar.validate` on every file and checks the naming.

- A file with no `* omitted:` line must pass with no errors.
- A file with an `* omitted:` line may only report `node … is floating` or `no ground node (0)`, which are the expected effects of removing a part. Any other error fails.

Pranshu spot-checks a sample against the drawings (Bible §11.3, T-3).

## Example

[`TEMPLATE.cir.example`](TEMPLATE.cir.example) shows the format. Its circuit is made up.
