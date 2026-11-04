"""Run real user photos through the demo app's image handler and score them.

  python scripts/run_user_image_tests.py --weights runs/baseline_v1/best.pt \
      --orient experiments/baseline_v1/models --cases results/user_input_testing/images

Each case is <id>.jpg (or .png/.jpeg/.heic converted to .jpg) plus <id>.cir, a
reference netlist written by hand from the drawing before running this script.
Outputs go to <cases>/out/<id>/: overlay.png, netlist.cir, report.txt, and one
summary.json for all cases. Nothing is written to the inputs.
"""
import argparse
import json
from pathlib import Path

import cv2

from circuitvision.app import build_ui
from circuitvision.compare import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--orient")
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()

    ui = build_ui(a.weights, not a.no_ocr, a.orient)
    rows = []
    for img_path in sorted(p for p in a.cases.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
        out = a.cases / "out" / img_path.stem
        out.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(img_path))
        vis, spice, report = ui.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cv2.imwrite(str(out / "overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        (out / "netlist.cir").write_text(spice)
        (out / "report.txt").write_text(report)
        ref = img_path.with_suffix(".cir")
        row = {"id": img_path.stem, "image": img_path.name, "reference": ref.exists(),
               "valid": "Validation: OK" in report}
        if ref.exists():
            row.update(compare(spice, ref.read_text(), simulate_op=True))
        rows.append(row)
        print(row)
    (a.cases / "out" / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
