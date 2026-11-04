# Real hand-drawn test photos (user input)

**Empty until real photos are added.** No image in this folder has been tested yet. They are kept separate from the CGHD benchmark and the synthetic experiments.

## How to add a case

1. Draw the circuit by hand on paper and photograph it with a phone.
2. Save it here as `<id>.jpg`, e.g. `U01.jpg`.
3. Write the reference netlist **from the drawing**, before running anything, as `<id>.cir` (format in [`references/README.md`](../../../references/README.md)). Use node 0 for the DC source's − terminal and the values written on the drawing.
4. Run:

```bash
python scripts/run_user_image_tests.py --weights runs/baseline_v1/best.pt --orient experiments/baseline_v1/models --cases results/user_input_testing/images
```

Outputs land in `out/<id>/` (overlay, netlist, report with detections, confidences and checks), plus `out/summary.json` with the topology, exact and operating-point match against each reference.

## Planned set (10)

| id | Circuit | Notes for drawing |
|---|---|---|
| U01 | Voltage divider: 9V battery, 1k and 2k in series | write the values next to the parts |
| U02 | Series resistors: 12V, three resistors | |
| U03 | Parallel resistors: 5V, two resistors in parallel | junction dots at both branch points |
| U04 | Series-parallel: 1k in series with (2k ∥ 3k) | |
| U05 | Diode: 5V, diode or LED and a 330R resistor | mark + on the battery |
| U06 | Capacitor: 6V, 2.2k and 100nF in series | |
| U07 | Several parts and junctions: R, C, L and two branches | |
| U08 | Crossing wires: two loops whose wires cross without connecting | draw the crossover hump |
| U09 | Any of the above on grid paper | |
| U10 | Any of the above photographed with uneven light, a shadow or a slight tilt | |
