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

from .classes import OTHER
from .orientation import OPPOSITE, SIDES, positive_side
from .types import Detection

MASKED = {"resistor", "capacitor", "inductor", "voltage_source", "diode",
          "gnd", "crossover", "text", OTHER}


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


def symbol_scale(dets: list[Detection], ref: float = 24.0) -> float:
    """Drawing scale relative to the size the defaults were tuned for: median
    short side of the element boxes / ref. Phone photos are often 4-10x larger."""
    sides = [min(d.box[2] - d.box[0], d.box[3] - d.box[1]) for d in dets
             if d.cls in MASKED and d.cls not in ("text", "crossover", OTHER)]
    return max(1.0, float(np.median(sides)) / ref) if sides else 1.0


def extract_nets(mask: np.ndarray, dets: list[Detection], pad: int | None = None,
                 gap_close: int | None = None, inflate: int | None = None,
                 min_wire: int | None = None):
    """Return ({detection index: [net ids]}, wire label image), nets numbered from 1.

    mask: 255 = ink. All size parameters default to values scaled by
    symbol_scale(), so a 4000 px phone photo behaves like a 400 px drawing:
      pad        ring width for terminal search (px)
      gap_close  dilation to bridge small pen gaps in wires (px)
      inflate    grow boxes before masking (detector boxes are tight and
                 symbol strokes leak a few px past them)
      min_wire   wire fragments smaller than this area (px) are residue
    Fragments that never leave the neighbourhood of a single masked box (pieces
    of the symbol's own outline sticking out of a tight box) are not wires.
    """
    s = symbol_scale(dets)
    pad = pad if pad is not None else round(6 * s)
    gap_close = gap_close if gap_close is not None else max(3, round(3 * s))
    inflate = inflate if inflate is not None else round(3 * s)
    min_wire = min_wire if min_wire is not None else round(40 * s * s)

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
    drop = stats[:, cv2.CC_STAT_AREA] < min_wire
    reach = pad + gap_close
    boxes = [_clip(d.box, mask.shape, reach) for d in dets if d.cls in MASKED]
    for lab in range(1, n):
        if drop[lab]:
            continue
        lx, ly, lw, lh = stats[lab, :4]
        drop[lab] = any(bx1 <= lx and by1 <= ly and lx + lw <= bx2 and ly + lh <= by2
                        for bx1, by1, bx2, by2 in boxes)
    drop[0] = False
    labels[drop[labels]] = 0
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
