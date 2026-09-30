"""Tagged tokens -> (source entity, series-parallel load tree).

1. Chunking: each component noun takes the nearest free value on its left
   (no other noun in between), else the nearest on its right. A UNIT token
   is glued onto the value before it ("4.7" + "k" -> "4.7k").
2. Structure:
   - list form  "A , B and C in series"   -> S(A, B, C)
   - chain form "A in series with B which is in parallel with C"
                -> S(A, P(B, C))   (each operator attaches to the nearest
                   right-hand phrase: right attachment, like PP-attachment)
The source is pulled out wherever it appears; it always sits across the load.
"""
from .lexicon import COMPONENT_TAGS, class_from_unit, spice_value
from .tree import E, op


def chunk(words, tags):
    """-> list of items: ("ENT", cls, value, pos) | ("OP", "S"/"P", pos) | ("SEP", pos)."""
    vals = []                                         # (pos, text)
    i = 0
    while i < len(words):
        if tags[i] == "VAL":
            text, j = words[i], i + 1
            while j < len(words) and tags[j] == "UNIT":
                text += words[j]; j += 1
            vals.append((i, text))
            i = j
        else:
            i += 1
    nouns = [i for i, t in enumerate(tags) if t in COMPONENT_TAGS]
    used, items = set(), []

    def free_between(a, b):
        lo, hi = sorted((a, b))
        return not any(lo < n < hi for n in nouns)

    for n in nouns:
        cls = COMPONENT_TAGS[tags[n]]
        left = [v for v in vals if v[0] < n and v[0] not in used and free_between(v[0], n)]
        right = [v for v in vals if v[0] > n and v[0] not in used and free_between(v[0], n)]
        pick = left[-1] if left else (right[0] if right else None)
        # Semantic constraints: an explicit unit outranks the tagger's noun
        # class ("a 1mH choke" is an inductor whatever the noun was tagged as),
        # and a source must be in volts.
        if pick and cls != "diode":
            unit_cls = class_from_unit(pick[1])
            if unit_cls:
                cls = unit_cls
            elif cls == "voltage_source":
                cls = "resistor"
        value = None
        if pick and cls != "diode":
            used.add(pick[0])
            value = spice_value(pick[1], cls)
        items.append(("ENT", cls, value, n))
    # Back-off: a value no noun claimed still implies a component
    # (unseen noun, e.g. "a 10k res"): class from its unit, bare numbers -> resistor
    for pos, text in vals:
        if pos not in used and not any(abs(pos - n) == 1 and tags[n] == "DIO" for n in nouns):
            cls = class_from_unit(text) or "resistor"
            items.append(("ENT", cls, spice_value(text, cls), pos))
    for i, t in enumerate(tags):
        if t in ("SER", "PAR"):
            items.append(("OP", "S" if t == "SER" else "P", i))
        elif t == "SEP":
            items.append(("SEP", i))
    return sorted(items, key=lambda x: x[-1])


def parse(words, tags):
    """-> (source E-node or None, load tree). Raises ValueError if no load."""
    items = chunk(words, tags)
    src = next((E(it[1], it[2]) for it in items if it[0] == "ENT" and it[1] == "voltage_source"), None)
    items = [it for it in items if not (it[0] == "ENT" and it[1] == "voltage_source")]
    ents = [it for it in items if it[0] == "ENT"]
    if not ents:
        raise ValueError("no components found")
    leaves = [E(e[1], e[2]) for e in ents]
    if len(leaves) == 1:
        return src, leaves[0]

    last_ent = ents[-1][-1]
    trailing = [it for it in items if it[0] == "OP" and it[-1] > last_ent]
    has_sep = any(it[0] == "SEP" and ents[0][-1] < it[-1] < last_ent for it in items)
    if has_sep and trailing:                          # list form
        return src, op(trailing[0][1], leaves)

    # chain form: operator between consecutive entities, default series
    ops = []
    for a, b in zip(ents, ents[1:]):
        between = [it[1] for it in items if it[0] == "OP" and a[-1] < it[-1] < b[-1]]
        ops.append(between[-1] if between else "S")
    tree = leaves[-1]
    for k, leaf in zip(reversed(ops), reversed(leaves[:-1])):
        tree = op(k, [leaf, tree])
    return src, tree
