"""Nets + detections -> SPICE netlist text."""
import re

from .classes import SPICE
from .types import Component, Detection

VALUE_RE = re.compile(r"^\s*(\d+(\.\d+)?)\s*([pnuµmkKMG]|meg)?\s*(ohm|Ω|f|F|h|H|v|V)?\s*$")


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _normalise_value(text: str) -> str | None:
    m = VALUE_RE.match(text.replace(",", "."))
    if not m:
        return None
    num, unit = m.group(1), (m.group(3) or "")
    unit = {"µ": "u", "K": "k", "M": "meg", "G": "g"}.get(unit, unit)
    return num + unit


def attach_values(comps: list[Component], dets: list[Detection], max_dist=150):
    """Give each element the nearest parsable text label within max_dist px."""
    texts = [d for d in dets if d.cls == "text" and d.text]
    for c in comps:
        cx, cy = _center(c.box)
        best, bd = None, max_dist
        for t in texts:
            v = _normalise_value(t.text)
            if v is None:
                continue
            tx, ty = _center(t.box)
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < bd:
                best, bd = v, dist
        c.value = best


def build(dets: list[Detection], nets: dict[int, list[int]]):
    gnd_nets = {n for i, d in enumerate(dets) if d.cls == "gnd"
                for n in nets.get(i, [])}
    counters, comps = {}, []
    for i, d in enumerate(dets):
        if d.cls not in SPICE:
            continue
        prefix = SPICE[d.cls][0]
        counters[prefix] = counters.get(prefix, 0) + 1
        node = [0 if n in gnd_nets else n for n in nets.get(i, [])]
        comps.append(Component(f"{prefix}{counters[prefix]}", d.cls, d.box, node))
    return comps


def to_spice(comps: list[Component], title="CircuitVision netlist") -> str:
    lines, warnings = [f"* {title}"], []
    needs_dmod = False
    for c in comps:
        if len(c.nets) != 2:
            warnings.append(f"* WARNING {c.ref}: found {len(c.nets)} terminals {c.nets}, skipped")
            continue
        value = c.value or SPICE[c.cls][1]
        needs_dmod |= c.cls == "diode"
        lines.append(f"{c.ref} {c.nets[0]} {c.nets[1]} {value}")
    if needs_dmod:
        lines.append(".model DMOD D")
    lines += warnings
    lines += [".op", ".end"]
    return "\n".join(lines) + "\n"
