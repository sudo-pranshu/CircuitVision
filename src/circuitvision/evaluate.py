"""Stage-wise downstream evaluation on CGHD-shaped data.

  python -m circuitvision.evaluate --cghd ~/data/cghd --drafters 12 \
      --exp experiments/baseline_v1 [--orient models] [--ocr] [--weights best.pt]

Two modes, reported separately so detector errors don't hide downstream errors:

  gt_boxes   ground-truth boxes, GT rotation and GT text as oracle inputs
  detector   detector boxes; rotation from --orient, values from --ocr

Per stage:
  orientation  predicted rotation vs GT <rotation> (needs --orient; mirrored excluded)
  ocr          EasyOCR on GT text boxes: exact match and character accuracy
               (1 - CER), on all labels and on value-like labels ("10k", "4,7uF")
  topology     PROXY on real CGHD: share of elements with exactly 2 terminals,
               and cross-photo CONSISTENCY: the 4 photos of one drawing should
               give the same circuit graph (consistency is not correctness)
  reference    only where a reference netlist exists (references/<stem>.cir,
               e.g. synthetic data): topology_match, exact_match, op_match
  netlist / validation / simulation

Images containing out-of-scope symbols (transistors, ICs, lamps, ...) cannot be
reconstructed completely by a two-terminal pipeline, so every metric is also
reported on the in-scope-only subset.

Writes <exp>/downstream/{gt_boxes.csv, detector.csv, ocr.csv, summary.json, netlists/}.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from . import grammar, netlist, simulate
from . import orientation as orient_mod
from .cghd_to_yolo import find_image, load_for_annotation, parse_voc
from .classes import OTHER
from .compare import compare, isomorphic
from .detector_eval import iou_matrix
from .orientation import POLARIZED
from .pipeline import detections_from_xml, run
from .types import Detection

STEM = re.compile(r"^(C-?\d+)_(D\d+)_")


def signature(comps):
    two = [c for c in comps if len(c.nets) == 2]
    return Counter(c.cls for c in two), len({n for c in two for n in c.nets})


def norm_text(s):
    return "".join((s or "").replace("Ω", "").lower().split()).replace("ohm", "")


def char_accuracy(pred: str, gold: str) -> float:
    """1 - character error rate (Levenshtein / len(gold)), floored at 0."""
    p, g = norm_text(pred), norm_text(gold)
    if not g:
        return float(p == g)
    d = list(range(len(p) + 1))
    for i, cg in enumerate(g, 1):
        prev, d[0] = d[0], i
        for j, cp in enumerate(p, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (cg != cp))
    return max(0.0, 1 - d[len(p)] / len(g))


def netlist_stages(img, dets, sim: bool, title: str):
    info = {}
    comps, _ = run(img, dets, info)
    text = netlist.to_spice(comps, title=title, notes=[f"ground: {info['ground']}"])
    errors = grammar.validate(text)
    r = {"elements": len(comps),
         "terminals_ok": round(sum(len(c.nets) == 2 for c in comps) / len(comps), 3) if comps else "",
         "valid": int(not errors), "first_error": errors[0] if errors else "",
         "ground": info["ground"]}
    if sim:
        try:
            simulate.operating_point(text)
            r["simulates"] = 1
        except (RuntimeError, OSError):
            r["simulates"] = 0
    return r, comps, text


def orientation_stage(model, img, gt_dets, pred_dets=None):
    """-> (correct, total) for polarized, non-mirrored parts with a GT rotation."""
    gt = [d for d in gt_dets if d.cls in POLARIZED and d.rotation is not None and not d.mirror]
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


def consistency(rows, key="netlist"):
    """Cross-photo consistency PROXY: share of photo pairs of the same drawing
    (same drafter, circuit C and drawing D) whose circuit graphs are isomorphic."""
    groups = defaultdict(list)
    for r in rows:
        m = STEM.match(Path(r["image"]).name)
        if m and r.get(key):
            groups[(r["image"].split("/")[0], m.group(1), m.group(2))].append(r[key])
    pairs = same = 0
    for texts in groups.values():
        for a, b in combinations(texts, 2):
            pairs += 1
            same += isomorphic(a, b, values=False)
    return {"pairs": pairs, "consistent": round(same / pairs, 4) if pairs else None,
            "drawings": sum(len(t) > 1 for t in groups.values())}


def mean(rows, key):
    v = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float, bool))]
    return round(sum(v) / len(v), 4) if v else None


def summarize(rows, ref_rows):
    out = {"images": len(rows),
           "topology_terminals_ok_PROXY": mean(rows, "terminals_ok"),
           "netlist_valid": mean(rows, "valid"), "simulates": mean(rows, "simulates"),
           "ground_rule": dict(Counter(r["ground"] for r in rows)),
           "top_validation_errors": Counter(re.sub(r"\d+", "N", r["first_error"])
                                            for r in rows if r["first_error"]).most_common(5),
           "consistency_PROXY": consistency(rows)}
    if ref_rows:
        out["vs_reference"] = {k: mean(ref_rows, k) for k in ("topology_match", "exact_match", "op_match")}
        out["vs_reference"]["n"] = len(ref_rows)
    return out


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
    orient = orient_mod.OrientationModel.load(a.orient) if a.orient else None
    det = None
    if a.weights:
        from .detect import Detector
        det = Detector(a.weights)
    read_values = None
    if a.ocr:
        from .ocr import read_values

    out = a.exp / "downstream"
    for sub in ("gt_boxes", "detector"):
        (out / "netlists" / sub).mkdir(parents=True, exist_ok=True)
    xmls = [x for d in a.drafters.split(",")
            for x in sorted((a.cghd / f"drafter_{d}" / "annotations").glob("*.xml"))][:a.limit]
    gt_rows, det_rows, ocr_rows, stage = [], [], [], Counter()
    ref_rows = {"gt_boxes": [], "detector": []}
    for n_done, xml in enumerate(xmls, 1):
        img_path = find_image(xml.parent.parent / "images", xml.stem)
        if img_path is None:
            stage["missing_images"] += 1
            continue
        xw, xh, _ = parse_voc(xml)
        img, frame = load_for_annotation(img_path, xw, xh)
        if img is None:
            stage["unreadable_images"] += 1
            continue
        stage[f"frame_{frame}"] += 1
        gt = detections_from_xml(xml)
        drafter = xml.parent.parent.name
        name = f"{drafter}/{img_path.name}"
        unsupported = sum(d.cls == OTHER for d in gt)
        ref_path = xml.parent.parent / "references" / f"{xml.stem}.cir"
        ref = ref_path.read_text() if ref_path.exists() else None
        row = {"image": name, "unsupported_symbols": unsupported, "in_scope": int(unsupported == 0)}

        if orient:
            ok, n = orientation_stage(orient, img, gt)
            stage["gt_orient_ok"] += ok; stage["gt_orient_n"] += n
        if read_values:
            texts = [d for d in gt if d.cls == "text" and d.text]
            probe = [Detection("text", d.box) for d in texts]
            read_values(img, probe)
            for p, g in zip(probe, texts):
                ocr_rows.append({"image": name, "gold": g.text, "pred": p.text or "",
                                 "value_like": int(netlist.parse_value(g.text)[0] is not None),
                                 "exact": int(norm_text(p.text) == norm_text(g.text)),
                                 "char_acc": round(char_accuracy(p.text or "", g.text), 4)})
        r, gt_comps, text = netlist_stages(img, gt, sim, name)          # oracle rotation + text
        (out / "netlists" / "gt_boxes" / f"{drafter}__{xml.stem}.cir").write_text(text)
        r["netlist"] = text
        if ref:
            c = compare(text, ref, simulate_op=sim)
            r.update(c)
            ref_rows["gt_boxes"].append({**c, "in_scope": row["in_scope"]})
        gt_rows.append({**row, **r})

        if det:
            pred = det(img)
            if orient:
                orient.annotate(img, pred)
                ok, n = orientation_stage(orient, img, gt, pred)
                stage["det_orient_ok"] += ok; stage["det_orient_n"] += n
            if read_values:
                read_values(img, pred)
            r, comps, text = netlist_stages(img, pred, sim, name)
            (out / "netlists" / "detector" / f"{drafter}__{xml.stem}.cir").write_text(text)
            r["netlist"] = text
            r["struct_match_vs_gt_boxes"] = int(signature(comps) == signature(gt_comps))
            r["isomorphic_to_gt_boxes"] = int(isomorphic(text, gt_rows[-1]["netlist"], values=False))
            if ref:
                c = compare(text, ref, simulate_op=sim)
                r.update(c)
                ref_rows["detector"].append({**c, "in_scope": row["in_scope"]})
            det_rows.append({**row, **r})
        if n_done % 25 == 0 or n_done == len(xmls):
            print(f"Evaluated {n_done}/{len(xmls)} images", flush=True)

    for fname, rows in (("gt_boxes", gt_rows), ("detector", det_rows), ("ocr", ocr_rows)):
        if rows:
            keys = [k for k in dict.fromkeys(k for r in rows for k in r) if k != "netlist"]
            with open(out / f"{fname}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader(); w.writerows(rows)

    acc = lambda ok, n: round(stage[ok] / stage[n], 4) if stage[n] else None
    ins = lambda rows: [r for r in rows if r.get("in_scope")]
    summary = {
        "drafters": a.drafters, "images": len(gt_rows), "in_scope_images": len(ins(gt_rows)),
        "ngspice": sim,
        "rotation_convention": {"PLUS_AT_0": orient_mod.PLUS_AT_0,
                                "CGHD_CLOCKWISE": orient_mod.CGHD_CLOCKWISE,
                                "source": orient_mod.CONVENTION_SOURCE},
        "detector_conf": det.conf if det else None,
        "images_by_frame": {k: v for k, v in stage.items() if k.startswith(("frame_", "missing", "unreadable"))},
        "orientation": ({"gt_boxes_accuracy": acc("gt_orient_ok", "gt_orient_n"), "gt_boxes_n": stage["gt_orient_n"],
                         "detector_accuracy_on_matched_gt": acc("det_orient_ok", "det_orient_n")}
                        if orient else "UNTESTED (no --orient)"),
        "ocr": ({"labels": len(ocr_rows), "exact": mean(ocr_rows, "exact"), "char_accuracy": mean(ocr_rows, "char_acc"),
                 "value_like_labels": sum(r["value_like"] for r in ocr_rows),
                 "value_like_exact": mean([r for r in ocr_rows if r["value_like"]], "exact"),
                 "value_like_char_accuracy": mean([r for r in ocr_rows if r["value_like"]], "char_acc")}
                if ocr_rows else "UNTESTED (no --ocr)"),
        "gt_boxes": {"all": summarize(gt_rows, ref_rows["gt_boxes"]),
                     "in_scope_only": summarize(ins(gt_rows), ins(ref_rows["gt_boxes"]))},
    }
    if det_rows:
        summary["detector"] = {
            "all": {**summarize(det_rows, ref_rows["detector"]),
                    "struct_match_vs_gt_boxes": mean(det_rows, "struct_match_vs_gt_boxes"),
                    "isomorphic_to_gt_boxes": mean(det_rows, "isomorphic_to_gt_boxes")},
            "in_scope_only": {**summarize(ins(det_rows), ins(ref_rows["detector"])),
                              "isomorphic_to_gt_boxes": mean(ins(det_rows), "isomorphic_to_gt_boxes")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
