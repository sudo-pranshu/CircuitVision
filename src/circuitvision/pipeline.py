"""End-to-end: image -> SPICE netlist (+ debug overlay).

  # with trained detector
  python -m circuitvision.pipeline circuit.jpg --weights runs/detect/train/weights/best.pt

  # with CGHD ground-truth boxes (evaluates wire tracing alone)
  python -m circuitvision.pipeline img.jpg --xml img.xml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from . import netlist, preprocess, topology
from .cghd_to_yolo import parse_voc
from .types import Detection


def detections_from_xml(xml_path) -> list[Detection]:
    _, _, objs = parse_voc(Path(xml_path))
    return [Detection(cls, tuple(int(v) for v in (x1, y1, x2, y2)), 1.0, text)
            for cls, x1, y1, x2, y2, text in objs]


def run(img, dets: list[Detection]):
    mask = preprocess.remove_small_blobs(preprocess.binarize(img))
    nets, labels = topology.extract_nets(mask, dets)
    comps = netlist.build(dets, nets)
    netlist.attach_values(comps, dets)
    return comps, labels


def overlay(img, comps, labels):
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, (labels.max() + 1, 3), dtype=np.uint8)
    colors[0] = 0
    vis = img.copy()
    wire = labels > 0
    vis[wire] = colors[labels[wire]]
    for c in comps:
        x1, y1, x2, y2 = c.box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{c.ref} {c.nets}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--weights")
    src.add_argument("--xml")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    img = preprocess.load(a.image)
    if a.xml:
        dets = detections_from_xml(a.xml)
    else:
        from .detect import Detector
        dets = Detector(a.weights)(img)

    comps, labels = run(img, dets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.image).stem
    spice = netlist.to_spice(comps, title=stem)
    (out / f"{stem}.cir").write_text(spice)
    cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay(img, comps, labels))
    print(spice)


if __name__ == "__main__":
    main()
