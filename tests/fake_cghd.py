"""Tiny fake CGHD tree for tests: 3 drafters, VOC XML, known problems."""
from pathlib import Path

import cv2
import numpy as np


def _obj(name, box, rotation=None, text=None):
    extra = (f"<rotation>{rotation}</rotation>" if rotation is not None else "") + \
            (f"<text>{text}</text>" if text is not None else "")
    x1, y1, x2, y2 = box
    return (f"<object><name>{name}</name>{extra}<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
            f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>")


def make(root: Path, w=200, h=100) -> Path:
    for d in ("drafter_1", "drafter_11", "drafter_12"):
        (root / d / "images").mkdir(parents=True)
        (root / d / "annotations").mkdir(parents=True)
        for k in range(2):
            stem = f"C{k}_D1_P1"
            cv2.imwrite(str(root / d / "images" / f"{stem}.jpg"), np.full((h, w, 3), 255, np.uint8))
            objs = [
                _obj("resistor", (10, 10, 50, 30)),
                _obj("diode", (60, 10, 90, 30), rotation=90),
                _obj("voltage.dc", (100, 10, 130, 40), rotation=0),
                _obj("text", (10, 40, 40, 55), text="10k"),
                _obj("transistor.bjt", (140, 10, 170, 40)),      # out of scope
                _obj("capacitor.unpolarized", (180, 60, 230, 90)),  # exceeds width -> clipped
                _obj("inductor", (50, 50, 50, 70)),               # zero width -> dropped
            ]
            (root / d / "annotations" / f"{stem}.xml").write_text(
                f"<annotation><size><width>{w}</width><height>{h}</height></size>"
                + "".join(objs) + "</annotation>")
    # annotation whose image is missing
    (root / "drafter_1" / "annotations" / "orphan.xml").write_text(
        f"<annotation><size><width>{w}</width><height>{h}</height></size></annotation>")
    return root
