"""Wire tracing: component boxes + stroke mask -> electrical nets.

Idea: erase every component box from the stroke mask so only wires remain,
label the connected wire regions (each region = one net), then look at a thin
ring just outside each box to see which nets touch that component.
Crossovers are masked too, then their opposite sides are re-joined
(left<->right, top<->bottom) so crossing wires don't short.
"""
from dataclasses import replace

import cv2
import numpy as np

from .orientation import OPPOSITE, SIDES, positive_side
from .types import Detection

MASKED = {"resistor", "capacitor", "inductor", "voltage_source", "diode",
          "gnd", "crossover", "text"}


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _clip(box, shape, pad=0):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)


def _ring_labels(labels, box, pad):
    """Wire labels in a ring of width `pad` around box, split by side."""
    x1, y1, x2, y2 = _clip(box, labels.shape)
    X1, Y1, X2, Y2 = _clip(box, labels.shape, pad)
    sides = {
        "left": labels[y1:y2, X1:x1],
        "right": labels[y1:y2, x2:X2],
        "top": labels[Y1:y1, x1:x2],
        "bottom": labels[y2:Y2, x1:x2],
    }
    return {k: set(np.unique(v)) - {0} for k, v in sides.items()}


def extract_nets(mask: np.ndarray, dets: list[Detection], pad: int = 6,
                 gap_close: int = 3, inflate: int = 3, min_wire: int = 40):
    """Return {detection index: [net ids]} with nets numbered from 1.

    mask: 255 = ink. pad: ring width for terminal search (px).
    gap_close: dilation to bridge small pen gaps in wires (px).
    inflate: grow boxes before masking, since detector boxes are tight and
             symbol strokes leak a few px past them.
    min_wire: wire fragments smaller than this (px) are ignored as residue.
    """
    dets = [replace(d, box=_clip(d.box, mask.shape, inflate)) for d in dets]
    wires = mask.copy()
    for d in dets:
        if d.cls in MASKED:
            x1, y1, x2, y2 = _clip(d.box, wires.shape)
            wires[y1:y2, x1:x2] = 0

    if gap_close:
        k = np.ones((gap_close, gap_close), np.uint8)
        wires = cv2.dilate(wires, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(wires, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < min_wire
    small[0] = False
    labels[small[labels]] = 0
    dsu = DSU(n)

    rings = {i: _ring_labels(labels, d.box, pad) for i, d in enumerate(dets)
             if d.cls in MASKED and d.cls != "text"}

    # Crossover: join opposite sides only
    for i, d in enumerate(dets):
        if d.cls == "crossover":
            s = rings[i]
            for a, b in (("left", "right"), ("top", "bottom")):
                ids = list(s[a] | s[b])
                for j in ids[1:]:
                    dsu.union(ids[0], j)

    # Canonical, compact net numbering
    roots, remap = {}, {}
    for lab in range(1, n):
        r = dsu.find(lab)
        remap[lab] = roots.setdefault(r, len(roots) + 1)

    out = {}
    for i, d in enumerate(dets):
        if i not in rings or d.cls == "crossover":
            continue
        s = rings[i]
        # Order terminals: positive/anode side first when orientation is known,
        # otherwise geometric (horizontal left->right, vertical top->bottom)
        pos = positive_side(d.cls, d.rotation)
        x1, y1, x2, y2 = d.box
        if pos:
            order = (pos, OPPOSITE[pos]) + tuple(k for k in SIDES if k not in (pos, OPPOSITE[pos]))
        elif (x2 - x1) >= (y2 - y1):
            order = ("left", "right", "top", "bottom")
        else:
            order = ("top", "bottom", "left", "right")
        nets = []
        for side in order:
            for lab in sorted(s[side]):
                net = remap[lab]
                if net not in nets:
                    nets.append(net)
        out[i] = nets
    return out, labels
