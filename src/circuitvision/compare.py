"""Is a reconstructed netlist the same circuit as a reference netlist?

A syntactically valid netlist is not necessarily the right circuit. This module
compares a prediction against a reference in three increasingly strict ways:

  topology_match   same circuit graph, ignoring values (element classes and how
                   they connect; net names may differ; which node is ground
                   does not matter)
  exact_match      same graph AND same component values
  op_match         ngspice DC operating points agree: every element sees the
                   same voltage (reference-node independent), within tolerance

Graph equality is an exact isomorphism test by backtracking over net
mappings; circuits here are small (tens of nets).
"""
import re
from collections import Counter

SUFFIX = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}
NUM = re.compile(r"^(-?\d+(?:\.\d+)?(?:e-?\d+)?)(meg|[fpnumkgt])?", re.I)
POLAR = {"D", "V"}


def spice_float(v: str | None):
    """'4.7k' -> 4700.0, 'DC 5' -> 5.0, 'DMOD' -> 'DMOD', None -> None."""
    if v is None:
        return None
    t = v.strip().lower().removeprefix("dc").strip()
    m = NUM.match(t)
    if not m:
        return v.strip().upper()
    return float(m.group(1)) * SUFFIX.get((m.group(2) or "").lower(), 1.0)


def elements(text: str):
    """-> [(prefix, n1, n2, value)] for R/C/L/V/D lines."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 3 and p[0][0].upper() in "RCLVD" and not line.startswith("*"):
            val = " ".join(p[3:]) if len(p) > 3 else None
            out.append((p[0][0].upper(), p[1], p[2], spice_float(val)))
    return out


def _label(e, values: bool):
    v = e[3]
    if isinstance(v, float):
        v = float(f"{v:.6g}")
    return (e[0], v) if values else (e[0],)


def isomorphic(pred: str, ref: str, values: bool = True) -> bool:
    """Exact graph isomorphism of two netlists (elements = labelled edges)."""
    A, B = elements(pred), elements(ref)
    if Counter(_label(e, values) for e in A) != Counter(_label(e, values) for e in B):
        return False

    def incidence(E):
        inc = {}
        for i, e in enumerate(E):
            for k, n in enumerate((e[1], e[2])):
                inc.setdefault(n, []).append((i, k))
        return inc

    ia, ib = incidence(A), incidence(B)
    if sorted(len(v) for v in ia.values()) != sorted(len(v) for v in ib.values()):
        return False

    def sig(E, inc, n):    # labels of incident elements, with terminal side for polarized parts
        return Counter(_label(E[i], values) + ((k,) if E[i][0] in POLAR else ()) for i, k in inc[n])

    sa = {n: sig(A, ia, n) for n in ia}
    sb = {n: sig(B, ib, n) for n in ib}
    order = sorted(ia, key=lambda n: -len(ia[n]))
    mapping, used = {}, set()

    def edges_ok():
        # every element whose both nets are mapped must exist in B with the same label
        want = Counter()
        for e in A:
            if e[1] in mapping and e[2] in mapping:
                a, b = mapping[e[1]], mapping[e[2]]
                key = (a, b) if e[0] in POLAR else tuple(sorted((a, b)))
                want[(_label(e, values), key)] += 1
        have = Counter()
        mapped = set(mapping.values())
        for e in B:
            if e[1] in mapped and e[2] in mapped:
                key = (e[1], e[2]) if e[0] in POLAR else tuple(sorted((e[1], e[2])))
                have[(_label(e, values), key)] += 1
        return all(have[k] >= c for k, c in want.items())

    def solve(i):
        if i == len(order):
            return edges_ok()
        n = order[i]
        for m in ib:
            if m not in used and sb[m] == sa[n]:
                mapping[n] = m
                used.add(m)
                if edges_ok() and solve(i + 1):
                    return True
                del mapping[n]
                used.discard(m)
        return False

    return solve(0)


def element_voltages(text: str, op: dict) -> dict:
    """{label: sorted voltages across elements with that label} from an operating point.
    Polarized parts keep the sign; others use magnitude (orientation-free)."""
    out = {}
    for e in elements(text):
        va = 0.0 if e[1] == "0" else op.get(f"v({e[1].lower()})")
        vb = 0.0 if e[2] == "0" else op.get(f"v({e[2].lower()})")
        if va is None or vb is None:
            return {}
        d = va - vb
        out.setdefault(_label(e, True), []).append(d if e[0] in POLAR else abs(d))
    return {k: sorted(v) for k, v in out.items()}


def voltages_match(a: dict, b: dict, rel: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    """ngspice prints ~6 significant digits, so compare with a tolerance."""
    if not a or a.keys() != b.keys():
        return False
    return all(len(a[k]) == len(b[k]) and
               all(abs(x - y) <= abs_tol + rel * max(abs(x), abs(y)) for x, y in zip(a[k], b[k]))
               for k in a)


def compare(pred: str, ref: str, simulate_op: bool = True) -> dict:
    r = {"topology_match": isomorphic(pred, ref, values=False),
         "exact_match": isomorphic(pred, ref, values=True), "op_match": None}
    if simulate_op:
        from . import simulate
        if simulate.available():
            try:
                vp = element_voltages(pred, simulate.operating_point(pred))
                vr = element_voltages(ref, simulate.operating_point(ref))
                r["op_match"] = voltages_match(vp, vr)
            except (RuntimeError, OSError):
                r["op_match"] = False
    return r
