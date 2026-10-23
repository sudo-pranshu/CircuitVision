"""Synthetic hand-drawn-style circuits with KNOWN topology, for testing wire tracing.

CGHD has no ground-truth circuit graphs, so topology accuracy can only be
measured exactly on images whose graph we know. Each circuit below is drawn
from a spec (components, wires, junction dots, crossovers) and can be degraded
the way phone photos are: higher resolution, grid paper, shadows, noise, blur
and JPEG compression.

  python -m circuitvision.synth --out results/synthetic_topology.json

Results from this module are SYNTHETIC/CONTROLLED, never real-data results.
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .types import Detection

INK = (35, 35, 35)


@dataclass
class Part:
    cls: str
    center: tuple[int, int]
    vertical: bool = False
    nets: tuple[int, int] = (0, 0)       # GT net of (left/top, right/bottom) terminal
    size: tuple[int, int] = (60, 24)     # long side, short side (unscaled)


@dataclass
class Circuit:
    name: str
    parts: list[Part]
    wires: list[list[tuple[int, int]]]   # polylines, unscaled coordinates
    junctions: list[tuple[int, int]] = field(default_factory=list)
    crossovers: list[tuple[int, int]] = field(default_factory=list)
    canvas: tuple[int, int] = (500, 420)  # w, h


def _box(p: Part, s: float):
    L, S = p.size
    w, h = (S, L) if p.vertical else (L, S)
    cx, cy = p.center
    return (int((cx - w / 2) * s), int((cy - h / 2) * s), int((cx + w / 2) * s), int((cy + h / 2) * s))


def terminals(p: Part):
    """Unscaled terminal points (left/top first)."""
    cx, cy = p.center
    half = p.size[0] / 2
    return ((cx, cy - half), (cx, cy + half)) if p.vertical else ((cx - half, cy), (cx + half, cy))


def _draw_symbol(img, p: Part, s: float, t: int):
    x1, y1, x2, y2 = _box(p, s)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    v = p.vertical
    if p.cls == "resistor":
        m = int(6 * s)
        cv2.rectangle(img, (x1 + (0 if v else m), y1 + (m if v else 0)),
                      (x2 - (0 if v else m), y2 - (m if v else 0)), INK, t)
        if v:
            cv2.line(img, (cx, y1), (cx, y1 + m), INK, t); cv2.line(img, (cx, y2 - m), (cx, y2), INK, t)
        else:
            cv2.line(img, (x1, cy), (x1 + m, cy), INK, t); cv2.line(img, (x2 - m, cy), (x2, cy), INK, t)
    elif p.cls == "capacitor":
        g = int(5 * s)
        if v:
            cv2.line(img, (cx, y1), (cx, cy - g), INK, t); cv2.line(img, (cx, cy + g), (cx, y2), INK, t)
            cv2.line(img, (x1, cy - g), (x2, cy - g), INK, t); cv2.line(img, (x1, cy + g), (x2, cy + g), INK, t)
        else:
            cv2.line(img, (x1, cy), (cx - g, cy), INK, t); cv2.line(img, (cx + g, cy), (x2, cy), INK, t)
            cv2.line(img, (cx - g, y1), (cx - g, y2), INK, t); cv2.line(img, (cx + g, y1), (cx + g, y2), INK, t)
    elif p.cls == "voltage_source":
        r = int(min(x2 - x1, y2 - y1) / 2) - 1
        cv2.circle(img, (cx, cy), r, INK, t)
        if v:
            cv2.line(img, (cx, y1), (cx, cy - r), INK, t); cv2.line(img, (cx, cy + r), (cx, y2), INK, t)
        else:
            cv2.line(img, (x1, cy), (cx - r, cy), INK, t); cv2.line(img, (cx + r, cy), (x2, cy), INK, t)
    elif p.cls == "diode":
        if v:
            tri = np.array([[x1, cy - int(8 * s)], [x2, cy - int(8 * s)], [cx, cy + int(6 * s)]])
            cv2.line(img, (cx, y1), (cx, cy - int(8 * s)), INK, t); cv2.line(img, (cx, cy + int(6 * s)), (cx, y2), INK, t)
            cv2.line(img, (x1, cy + int(6 * s)), (x2, cy + int(6 * s)), INK, t)
        else:
            tri = np.array([[cx - int(8 * s), y1], [cx - int(8 * s), y2], [cx + int(6 * s), cy]])
            cv2.line(img, (x1, cy), (cx - int(8 * s), cy), INK, t); cv2.line(img, (cx + int(6 * s), cy), (x2, cy), INK, t)
            cv2.line(img, (cx + int(6 * s), y1), (cx + int(6 * s), y2), INK, t)
        cv2.polylines(img, [tri], True, INK, t)
    elif p.cls == "inductor":
        n = 4
        if v:
            seg = (y2 - y1) / n
            for k in range(n):
                cv2.ellipse(img, (cx, int(y1 + seg * (k + 0.5))), (int(seg / 2), int(seg / 2)), 0, -90, 90, INK, t)
        else:
            seg = (x2 - x1) / n
            for k in range(n):
                cv2.ellipse(img, (int(x1 + seg * (k + 0.5)), cy), (int(seg / 2), int(seg / 2)), 0, 180, 360, INK, t)
    elif p.cls == "gnd":
        cv2.line(img, (cx, y1), (cx, cy), INK, t)
        for k, wd in enumerate((1.0, 0.6, 0.25)):
            yy = cy + k * int(4 * s)
            hw = int((x2 - x1) / 2 * wd)
            cv2.line(img, (cx - hw, yy), (cx + hw, yy), INK, t)


def _wobble(poly, amp, rng, step=15):
    """Hand-drawn wires: resample each segment and jitter the inner points."""
    out = [poly[0]]
    for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
        n = max(1, int(np.hypot(x1 - x0, y1 - y0) / step))
        for k in range(1, n + 1):
            t = k / n
            j = rng.normal(0, amp, 2) if k < n else (0, 0)
            out.append((x0 + (x1 - x0) * t + j[0], y0 + (y1 - y0) * t + j[1]))
    return out


def render(c: Circuit, scale: float = 1.0, stroke: int = 3, grid: bool = False, shadow: bool = False,
           noise: float = 0.0, blur: int = 0, jpeg: int = 0, seed: int = 0, ink: int = 35,
           gaps: int = 0, wobble: float = 0.0, box_jitter: float = 0.0):
    """-> (BGR image, detections in Part order then crossovers).

    ink: stroke grey level (35 pen, ~130 pencil); gaps: pen gaps per wire;
    wobble: hand-drawn wire jitter (unscaled px); box_jitter: detector-like box
    error as a fraction of box size (shift and resize).
    """
    rng = np.random.default_rng(seed)
    global INK
    INK = (ink, ink, ink)
    W, H = int(c.canvas[0] * scale), int(c.canvas[1] * scale)
    t = max(1, int(round(stroke * scale)))
    img = np.full((H, W, 3), 238, np.uint8)
    if grid:
        step = max(8, int(12 * scale))
        for x in range(0, W, step):
            cv2.line(img, (x, 0), (x, H), (205, 195, 180), max(1, t // 3))
        for y in range(0, H, step):
            cv2.line(img, (0, y), (W, y), (205, 195, 180), max(1, t // 3))
    for poly in c.wires:
        pts_f = _wobble(poly, wobble, rng) if wobble else poly
        pts = np.array([(int(x * scale), int(y * scale)) for x, y in pts_f])
        cv2.polylines(img, [pts], False, INK, t)
        for _ in range(gaps):                       # small breaks in the pen stroke
            k = rng.integers(1, len(pts)) if len(pts) > 1 else 0
            a, b = pts[k - 1], pts[k]
            f = rng.uniform(0.3, 0.7)
            p = (int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f))
            cv2.circle(img, p, max(1, int(1.2 * scale)), (238, 238, 238), -1)
    for x, y in c.junctions:
        cv2.circle(img, (int(x * scale), int(y * scale)), int(t * 1.8), INK, -1)
    dets = []
    for p in c.parts:
        _draw_symbol(img, p, scale, t)
        dets.append(Detection(p.cls, _box(p, scale)))
    for x, y in c.crossovers:
        r = int(10 * scale)
        dets.append(Detection("crossover", (int(x * scale) - r, int(y * scale) - r,
                                            int(x * scale) + r, int(y * scale) + r)))
    if box_jitter:
        jittered = []
        for d in dets:
            x1, y1, x2, y2 = d.box
            w, h = x2 - x1, y2 - y1
            dx, dy = rng.uniform(-box_jitter, box_jitter, 2) * (w, h)
            sw, sh = 1 + rng.uniform(-box_jitter, box_jitter, 2)
            cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
            jittered.append(Detection(d.cls, (int(cx - w * sw / 2), int(cy - h * sh / 2),
                                              int(cx + w * sw / 2), int(cy + h * sh / 2))))
        dets = jittered
    img = img.astype(np.float32)
    if shadow:
        gx = np.linspace(1.0, 0.45, W)[None, :, None]
        gy = np.linspace(1.0, 0.8, H)[:, None, None]
        img *= gx * gy
    if noise:
        img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (blur | 1, blur | 1), 0)
    if jpeg:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
    return img, dets


def gt_partition(c: Circuit):
    """GT terminal->net mapping for the two-terminal parts (gnd has one terminal)."""
    return [p.nets for p in c.parts]


def same_partition(pred: list[list[int]], gt: list[tuple]) -> bool:
    """True if predicted terminal nets induce exactly the GT partition (labels may differ)."""
    if any(len(p) != len(g) for p, g in zip(pred, gt)):
        return False
    fwd, back = {}, {}
    for p, g in zip(pred, gt):
        for a, b in zip(p, g):
            if fwd.setdefault(a, b) != b or back.setdefault(b, a) != a:
                return False
    return True


# ---------------------------------------------------------------- circuits
# Net 0 is ground. Terminals: (left, right) for horizontal, (top, bottom) for vertical.

def divider():
    return Circuit("divider", [
        Part("voltage_source", (80, 210), True, (1, 0), (70, 40)),
        Part("resistor", (240, 80), False, (1, 2)),
        Part("resistor", (400, 210), True, (2, 0)),
        Part("gnd", (240, 368), False, (0,), (30, 30)),
    ], wires=[[(80, 175), (80, 80), (210, 80)], [(270, 80), (400, 80), (400, 180)],
              [(400, 240), (400, 340), (80, 340), (80, 245)], [(240, 340), (240, 353)]])


def parallel_rc():
    """Junction dots: R and C both across the source, then a series diode."""
    return Circuit("parallel_rc_junctions", [
        Part("voltage_source", (60, 210), True, (1, 0), (70, 40)),
        Part("resistor", (200, 210), True, (2, 0)),
        Part("capacitor", (320, 210), True, (2, 0), (50, 34)),
        Part("diode", (130, 80), False, (1, 2), (50, 28)),
        Part("gnd", (200, 380), False, (0,), (30, 30)),
    ], wires=[[(60, 175), (60, 80), (105, 80)], [(155, 80), (320, 80), (320, 185)],
              [(200, 80), (200, 180)], [(200, 240), (200, 330)], [(320, 235), (320, 330), (60, 330), (60, 245)],
              [(200, 330), (200, 365)]],
        junctions=[(200, 80), (200, 330)])


def crossover():
    """Two independent loops whose wires cross twice without connecting."""
    return Circuit("crossover", [
        Part("resistor", (90, 200), False, (1, 2)),
        Part("resistor", (390, 200), False, (2, 1)),
        Part("resistor", (240, 60), True, (3, 4)),
        Part("resistor", (240, 340), True, (4, 3)),
    ], wires=[[(120, 200), (360, 200)], [(240, 90), (240, 310)],
              [(60, 200), (30, 200), (30, 400), (470, 400), (470, 200), (420, 200)],
              [(240, 30), (240, 15), (490, 15), (490, 420), (240, 420), (240, 370)]],
        crossovers=[(240, 200), (240, 400)], canvas=(520, 440))


def series_rlc():
    return Circuit("series_rlc_corners", [
        Part("voltage_source", (60, 200), True, (1, 0), (70, 40)),
        Part("resistor", (170, 60), False, (1, 2)),
        Part("inductor", (330, 60), False, (2, 3)),
        Part("capacitor", (420, 200), True, (3, 0), (50, 34)),
        Part("gnd", (240, 370), False, (0,), (30, 30)),
    ], wires=[[(60, 165), (60, 60), (140, 60)], [(200, 60), (300, 60)], [(360, 60), (420, 60), (420, 175)],
              [(420, 225), (420, 330), (60, 330), (60, 235)], [(240, 330), (240, 355)]])


def bridge():
    """Wheatstone-style bridge with four junctions."""
    return Circuit("bridge_junctions", [
        Part("voltage_source", (40, 220), True, (1, 0), (70, 40)),
        Part("resistor", (180, 120), True, (1, 2)),
        Part("resistor", (180, 300), True, (2, 0)),
        Part("resistor", (360, 120), True, (1, 3)),
        Part("resistor", (360, 300), True, (3, 0)),
        Part("resistor", (270, 210), False, (2, 3)),
    ], wires=[[(40, 185), (40, 40), (360, 40), (360, 90)], [(180, 40), (180, 90)],
              [(180, 150), (180, 270)], [(360, 150), (360, 270)], [(180, 210), (240, 210)], [(300, 210), (360, 210)],
              [(180, 330), (180, 400), (40, 400), (40, 255)], [(360, 330), (360, 400), (180, 400)]],
        junctions=[(180, 40), (180, 210), (360, 210), (180, 400)], canvas=(420, 420))


CIRCUITS = [divider, parallel_rc, crossover, series_rlc, bridge]

CONDITIONS = {
    "clean": dict(),
    "photo_scale_x6": dict(scale=6.0),
    "grid_paper": dict(scale=3.0, grid=True),
    "shadow": dict(scale=3.0, shadow=True),
    "noise_blur_jpeg": dict(scale=3.0, noise=12.0, blur=5, jpeg=60),
    "all_combined": dict(scale=6.0, grid=True, shadow=True, noise=10.0, blur=5, jpeg=70),
    # harder conditions, added after the first six passed (not tuned against)
    "pen_gaps": dict(scale=3.0, gaps=2),
    "pencil_on_grid": dict(scale=3.0, grid=True, ink=130, stroke=2),
    "wobbly_lines": dict(scale=3.0, wobble=2.5),
    "box_jitter_10pct": dict(scale=3.0, box_jitter=0.10),
    "box_jitter_15pct": dict(scale=3.0, box_jitter=0.15),
}


def benchmark(extract, seeds=(0, 1, 2)) -> dict:
    """extract(img, dets) -> {det index: [nets]}. -> {condition: {circuit: correct/len(seeds)}}."""
    out = {}
    for cond, kw in CONDITIONS.items():
        row = {}
        for make in CIRCUITS:
            c = make()
            ok = 0
            for seed in seeds:
                img, dets = render(c, seed=seed, **kw)
                nets = extract(img, dets)
                pred = [nets.get(i, []) for i in range(len(c.parts))]
                ok += same_partition(pred, gt_partition(c))
            row[c.name] = ok / len(seeds)
        out[cond] = row
    return out


def _default_extract(img, dets):
    from .pipeline import trace
    return trace(img, dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    res = benchmark(_default_extract)
    names = [m().name for m in CIRCUITS]
    print(f"{'condition':<18}" + "".join(f"{n[:12]:>14}" for n in names) + f"{'mean':>8}")
    for cond, row in res.items():
        print(f"{cond:<18}" + "".join(f"{row[n]:>14.2f}" for n in names) + f"{np.mean(list(row.values())):>8.2f}")
    overall = float(np.mean([v for r in res.values() for v in r.values()]))
    print(f"exact-topology rate over {len(res)} conditions x {len(names)} circuits x 3 seeds: {overall:.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"label": "SYNTHETIC/CONTROLLED", "metric": "exact terminal-net partition",
                                     "seeds": 3, "overall": overall, "results": res}, indent=1))


if __name__ == "__main__":
    main()
