"""Nets + detections -> SPICE netlist text."""
import re

from .classes import SPICE
from .types import Component, Detection

VALUE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(meg|[pnuµμmkKMG])?\s*(ohms?|Ω|[fFhHvV])?\s*$")
RKM_RE = re.compile(r"^\s*(\d+)([pnuµμmkKMGR])(\d+)\s*(ohms?|Ω|[fFhHvV])?\s*$")   # 4k7, 2R2, 1M5
UNIT_CLASS = {"ohm": "resistor", "ohms": "resistor", "Ω": "resistor", "f": "capacitor",
              "h": "inductor", "v": "voltage_source"}
PREFIX = {"µ": "u", "μ": "u", "K": "k", "M": "meg", "G": "g", "R": ""}


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def parse_value(text: str):
    """'10k' -> ('10k', None); '4,7uF' -> ('4.7u', 'capacitor'); '4k7' -> ('4.7k', None);
    '2R2' -> ('2.2', None); 'R1' / 'S2' / labels -> (None, None).

    Returns (SPICE value, class implied by an explicit unit or None)."""
    t = text.strip().replace(",", ".")
    m = RKM_RE.match(t)
    if m:
        num, pre, unit = f"{m.group(1)}.{m.group(3)}", m.group(2), m.group(4)
    else:
        m = VALUE_RE.match(t)
        if not m:
            return None, None
        num, pre, unit = m.group(1), m.group(2) or "", m.group(3)
    pre = PREFIX.get(pre, pre)
    cls = UNIT_CLASS.get(unit if unit in ("Ω",) else (unit or "").lower()) if unit else None
    return num + pre, cls


def _normalise_value(text: str) -> str | None:
    return parse_value(text)[0]


def attach_values(comps: list[Component], dets: list[Detection], max_dist=None):
    """Attach value labels to elements: closest pairs first, each label used once.

    A label is a candidate only if it parses as a value and any explicit unit
    agrees with the element class ('10uF' never goes to a resistor). Distance
    limit: max_dist, or by default 1.5x the element's longer box side (scales
    with photo resolution), at least 150 px.
    """
    labels = []
    for t in (d for d in dets if d.cls == "text" and d.text):
        value, unit_cls = parse_value(t.text)
        if value is not None:
            labels.append((t, value, unit_cls))
    pairs = []
    for ci, c in enumerate(comps):
        cx, cy = _center(c.box)
        limit = max_dist or max(150.0, 1.5 * max(c.box[2] - c.box[0], c.box[3] - c.box[1]))
        for li, (t, value, unit_cls) in enumerate(labels):
            if unit_cls and unit_cls != c.cls:
                continue
            tx, ty = _center(t.box)
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist <= limit:
                pairs.append((dist, ci, li))
    used_c, used_l = set(), set()
    for dist, ci, li in sorted(pairs):
        if ci in used_c or li in used_l:
            continue
        comps[ci].value = labels[li][1]
        used_c.add(ci)
        used_l.add(li)


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
