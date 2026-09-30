"""Series-parallel circuit trees: the semantic representation for text2netlist.

A tree is ("E", cls, value) | ("S", [children]) | ("P", [children]).
"""


def E(cls, value=None):
    return ("E", cls, value)


def op(kind, children):
    """Build S/P node, flattening nested nodes of the same kind."""
    flat = []
    for c in children:
        flat += c[1] if c[0] == kind else [c]
    return flat[0] if len(flat) == 1 else (kind, flat)


def canonical(t) -> str:
    """Order-independent string form, for exact-match evaluation.

    Parallel branches are unordered; series order is kept, since it decides
    which node a part sits on.
    """
    if t[0] == "E":
        return f"{t[1]}:{t[2] or '-'}"
    parts = [canonical(c) for c in t[1]]
    if t[0] == "P":
        parts.sort()
    return f"{t[0]}({','.join(parts)})"


def to_netlist(load, source=None, title="text2netlist") -> str:
    """Source across the whole load: + on node 1, - on ground (0)."""
    from ..classes import SPICE
    counters, lines, next_node = {}, [], [2]

    def ref(cls):
        p = SPICE[cls][0]
        counters[p] = counters.get(p, 0) + 1
        return f"{p}{counters[p]}"

    def place(t, a, b):
        if t[0] == "E":
            lines.append(f"{ref(t[1])} {a} {b} {t[2] or SPICE[t[1]][1]}")
        elif t[0] == "P":
            for c in t[1]:
                place(c, a, b)
        else:
            nodes = [a]
            for _ in t[1][:-1]:
                nodes.append(next_node[0]); next_node[0] += 1
            nodes.append(b)
            for c, x, y in zip(t[1], nodes, nodes[1:]):
                place(c, x, y)

    src_val = source[2] if source and source[2] else SPICE["voltage_source"][1]
    lines.append(f"{ref('voltage_source')} 1 0 {src_val}")
    place(load, 1, 0)
    out = [f"* {title}"] + lines
    if any(l.startswith("D") for l in lines):
        out.append(".model DMOD D")
    return "\n".join(out + [".op", ".end"]) + "\n"
