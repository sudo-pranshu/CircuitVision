"""Stage-wise downstream evaluation on real CGHD drafters.

  python -m circuitvision.evaluate --cghd ~/data/cghd --drafters 12 \
      --exp experiments/baseline_v1 [--orient models] [--ocr] [--weights best.pt]

Two modes, reported separately so detector errors don't hide downstream errors:

  GT boxes   ground-truth component boxes, GT rotation and GT text as oracle
             inputs to the netlist; each stage is also scored on its own:
               orientation  predicted rotation vs GT rotation   (needs --orient)
               ocr          OCR string vs GT text on GT boxes   (needs --ocr)
               topology     share of elements with exactly 2 terminals (PROXY:
                            CGHD has no ground-truth netlists, so this checks
                            plausibility, not correctness)
               netlist / validation / simulation
  detector   detector boxes; rotation from --orient, values from --ocr (none
             otherwise); same stages, plus struct_match against the GT-box
             netlist (same element counts per class and same node count)

Writes <exp>/downstream/{gt_boxes.csv, detector.csv, summary.json}.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import grammar, netlist, preprocess, simulate
from .cghd_to_yolo import find_image
from .detector_eval import iou_matrix
from . import orientation as orient_mod
from .orientation import POLARIZED
from .pipeline import detections_from_xml, run
from .types import Detection


def signature(comps):
    two = [c for c in comps if len(c.nets) == 2]
    return Counter(c.cls for c in two), len({n for c in two for n in c.nets})


def norm_text(s):
    return "".join((s or "").replace("Ω", "").lower().split()).replace("ohm", "")


def netlist_stages(img, dets, sim: bool):
    comps, _ = run(img, dets)
    text = netlist.to_spice(comps)
    errors = grammar.validate(text)
    r = {"elements": len(comps),
         "terminals_ok": round(sum(len(c.nets) == 2 for c in comps) / len(comps), 3) if comps else "",
         "valid": int(not errors), "first_error": errors[0] if errors else ""}
    if sim:
        try:
            simulate.operating_point(text)
            r["simulates"] = 1
        except (RuntimeError, OSError):
            r["simulates"] = 0
    return r, comps


def orientation_stage(model, img, gt_dets, pred_dets=None):
    """-> (correct, total) for polarized parts with a GT rotation."""
    gt = [d for d in gt_dets if d.cls in POLARIZED and d.rotation is not None]
    if not gt:
        return 0, 0
    if pred_dets is None:                              # GT boxes: predict on each GT box
        probe = [Detection(d.cls, d.box) for d in gt]
        model.annotate(img, probe)
        return sum(p.rotation == g.rotation for p, g in zip(probe, gt)), len(gt)
    pol = [d for d in pred_dets if d.cls in POLARIZED]   # detector boxes: match by IoU
    if not pol:
        return 0, len(gt)
    iou = iou_matrix(np.array([d.box for d in gt], float), np.array([d.box for d in pol], float))
    ok = 0
    for i, g in enumerate(gt):
        j = int(iou[i].argmax())
        ok += iou[i, j] >= 0.5 and pol[j].cls == g.cls and pol[j].rotation == g.rotation
    return int(ok), len(gt)


def mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(v) / len(v), 4) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghd", required=True, type=Path)
    ap.add_argument("--drafters", default="12")
    ap.add_argument("--exp", required=True, type=Path)
    ap.add_argument("--weights")
    ap.add_argument("--orient", help="orientation model folder")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    sim = simulate.available()
    orient = None
    if a.orient:
        from .orientation import OrientationModel
        orient = OrientationModel.load(a.orient)
    det = None
    if a.weights:
        from .detect import Detector
        det = Detector(a.weights)
    read_values = None
    if a.ocr:
        from .ocr import read_values

    xmls = [x for d in a.drafters.split(",")
            for x in sorted((a.cghd / f"drafter_{d}" / "annotations").glob("*.xml"))][:a.limit]
    gt_rows, det_rows, stage = [], [], Counter()
    for xml in xmls:
        img_path = find_image(xml.parent.parent / "images", xml.stem)
        if img_path is None:
            continue
        img = preprocess.load(str(img_path))
        gt = detections_from_xml(xml)

        row = {"image": f"{xml.parent.parent.name}/{img_path.name}"}
        if orient:
            ok, n = orientation_stage(orient, img, gt)
            stage["gt_orient_ok"] += ok; stage["gt_orient_n"] += n
        if read_values:
            texts = [d for d in gt if d.cls == "text" and d.text]
            probe = [Detection("text", d.box) for d in texts]
            read_values(img, probe)
            ok = sum(norm_text(p.text) == norm_text(g.text) for p, g in zip(probe, texts))
            stage["gt_ocr_ok"] += ok; stage["gt_ocr_n"] += len(texts)
            row["ocr_exact"] = round(ok / len(texts), 3) if texts else ""
        r, gt_comps = netlist_stages(img, gt, sim)          # oracle rotation + text
        gt_rows.append({**row, **r})

        if det:
            pred = det(img)
            if orient:
                orient.annotate(img, pred)
                ok, n = orientation_stage(orient, img, gt, pred)
                stage["det_orient_ok"] += ok; stage["det_orient_n"] += n
            if read_values:
                read_values(img, pred)
            r, comps = netlist_stages(img, pred, sim)
            r["struct_match"] = int(signature(comps) == signature(gt_comps))
            det_rows.append({"image": row["image"], **r})

    out = a.exp / "downstream"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("gt_boxes", gt_rows), ("detector", det_rows)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    acc = lambda ok, n: round(stage[ok] / stage[n], 4) if stage[n] else None
    summary = {
        "drafters": a.drafters, "images": len(gt_rows), "ngspice": sim,
        "rotation_convention": {"PLUS_AT_0": orient_mod.PLUS_AT_0,
                                "CGHD_CLOCKWISE": orient_mod.CGHD_CLOCKWISE,
                                "source": orient_mod.CONVENTION_SOURCE},
        "detector_conf": det.conf if det else None,
        "gt_boxes": {
            "orientation_accuracy": acc("gt_orient_ok", "gt_orient_n"), "orientation_n": stage["gt_orient_n"],
            "ocr_exact_match": acc("gt_ocr_ok", "gt_ocr_n"), "ocr_n": stage["gt_ocr_n"],
            "topology_terminals_ok_PROXY": mean(gt_rows, "terminals_ok"),
            "netlist_valid": mean(gt_rows, "valid"), "simulates": mean(gt_rows, "simulates"),
            "top_validation_errors": Counter(r["first_error"].split(" (")[0] for r in gt_rows
                                             if r["first_error"]).most_common(5)},
    }
    if det_rows:
        summary["detector"] = {
            "orientation_accuracy_on_matched_gt": acc("det_orient_ok", "det_orient_n"),
            "topology_terminals_ok_PROXY": mean(det_rows, "terminals_ok"),
            "netlist_valid": mean(det_rows, "valid"), "simulates": mean(det_rows, "simulates"),
            "struct_match_vs_gt_boxes": mean(det_rows, "struct_match")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
