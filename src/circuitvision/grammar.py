"""Netlist validation: a context-free grammar for our SPICE subset, an Earley
recognizer, and semantic (constraint) checks on top of the parse.

Terminals are predicates on raw tokens rather than fixed token types, so an
ambiguous token like "1" can be a NODE or a VALUE; Earley keeps both readings
alive and the grammar decides.
"""
import re
from collections import Counter

NUM = r"\d+(\.\d+)?"
TERMINALS = {
    "rname": re.compile(r"^r\w+$", re.I),
    "cname": re.compile(r"^c\w+$", re.I),
    "lname": re.compile(r"^l\w+$", re.I),
    "vname": re.compile(r"^v\w+$", re.I),
    "dname": re.compile(r"^d\w+$", re.I),
    "node": re.compile(r"^\d+$"),
    "value": re.compile(rf"^{NUM}(meg|[fpnumkg])?\w*$", re.I),
    "ident": re.compile(r"^[a-z_]\w*$", re.I),
    "dc": re.compile(r"^dc$", re.I),
    "dotmodel": re.compile(r"^\.model$", re.I),
    "dotop": re.compile(r"^\.op$", re.I),
    "dotend": re.compile(r"^\.end$", re.I),
    "comment": re.compile(r"^\*"),
    "EOL": re.compile(r"^\n$"),
}

GRAMMAR = {
    "S":        [("TITLE", "EOL", "BODY", "ENDLINE"), ("TITLE", "EOL", "ENDLINE")],
    "TITLE":    [("comment",)],
    "BODY":     [("LINE",), ("LINE", "BODY")],
    "LINE":     [("STMT", "EOL")],
    "STMT":     [("R",), ("C",), ("L",), ("V",), ("D",), ("MODEL",), ("dotop",), ("comment",)],
    "R":        [("rname", "node", "node", "value")],
    "C":        [("cname", "node", "node", "value")],
    "L":        [("lname", "node", "node", "value")],
    "V":        [("vname", "node", "node", "SRC")],
    "SRC":      [("value",), ("dc", "value")],
    "D":        [("dname", "node", "node", "ident")],
    "MODEL":    [("dotmodel", "ident", "ident")],
    "ENDLINE":  [("dotend", "EOL")],
}


def tokenize(text: str) -> list[tuple[str, int]]:
    """-> [(token, line_no)], one EOL token per non-empty line."""
    toks = []
    for no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        toks += [(line, no)] if line.startswith("*") else [(t, no) for t in line.split()]
        toks.append(("\n", no))
    return toks


def earley(tokens: list[str], grammar=GRAMMAR, start="S"):
    """Earley recognizer. Returns (accepted, index of first failing token).

    Chart items: (lhs, rhs, dot, origin). No epsilon rules in the grammar,
    which keeps predict/complete simple.
    """
    n = len(tokens)
    chart = [set() for _ in range(n + 1)]
    for rhs in grammar[start]:
        chart[0].add((start, rhs, 0, 0))

    for i in range(n + 1):
        agenda = list(chart[i])
        while agenda:
            lhs, rhs, dot, org = agenda.pop()
            if dot < len(rhs):
                sym = rhs[dot]
                if sym in grammar:                                  # predict
                    for r in grammar[sym]:
                        it = (sym, r, 0, i)
                        if it not in chart[i]:
                            chart[i].add(it); agenda.append(it)
                elif i < n and TERMINALS[sym].match(tokens[i]):     # scan
                    chart[i + 1].add((lhs, rhs, dot + 1, org))
            else:                                                   # complete
                for l2, r2, d2, o2 in list(chart[org]):
                    if d2 < len(r2) and r2[d2] == lhs:
                        it = (l2, r2, d2 + 1, o2)
                        if it not in chart[i]:
                            chart[i].add(it); agenda.append(it)
        if i < n and not chart[i + 1]:
            return False, i
    ok = any(l == start and d == len(r) and o == 0 for l, r, d, o in chart[n])
    return ok, (None if ok else n)


def elements(text: str):
    """-> [(ref, [nodes], rest)] for element lines."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if p and p[0][0].upper() in "RCLVD" and len(p) >= 3:
            out.append((p[0].upper(), p[1:3], p[3:]))
    return out


def validate(text: str) -> list[str]:
    """Syntax (CFG) + semantic checks. Empty list = valid netlist."""
    toks = tokenize(text)
    ok, at = earley([t for t, _ in toks])
    if not ok:
        where = toks[at] if at is not None and at < len(toks) else ("<eof>", "end")
        return [f"syntax error at line {where[1]} near {where[0]!r}"]

    errs = []
    els = elements(text)
    refs = Counter(r for r, _, _ in els)
    errs += [f"duplicate reference {r}" for r, k in refs.items() if k > 1]

    degree = Counter(n for _, nodes, _ in els for n in nodes)
    if "0" not in degree:
        errs.append("no ground node (0)")
    errs += [f"node {n} is floating (1 connection)" for n, k in degree.items() if k < 2]
    errs += [f"{r} is shorted (both terminals on node {a})"
             for r, (a, b), _ in els if a == b]

    models = {l.split()[1].upper() for l in text.splitlines()
              if l.lower().startswith(".model") and len(l.split()) > 1}
    errs += [f"{r} uses undefined model {rest[0]}" for r, _, rest in els
             if r.startswith("D") and rest and rest[0].upper() not in models]
    return errs
