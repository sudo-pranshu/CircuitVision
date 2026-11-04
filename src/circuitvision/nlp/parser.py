"""Tagged tokens -> (source entity, series-parallel load tree).

1. Chunking: each component noun takes the nearest free value on its left
   (no other noun in between), else the nearest on its right. UNIT tokens are
   glued onto the value before them ("4.7" + "k" -> "4.7k"). A count word
   directly before a valued plural noun ("two 1k resistors") repeats it.
   Values no noun claimed still imply a component (unseen nouns: "a 10k res").
2. Structure, from the outside in:
   - parentheses group explicitly:  "A in series with (B in parallel with C)"
   - a comma followed by an operator closes everything before it into one
     operand:  "A and B in parallel, in series with C" -> S(P(A, B), C)
               "A in series with B, all in parallel with C" -> P(S(A, B), C)
   - inside a clause, list form  "A, B and C in series"  -> S(A, B, C)
   - otherwise chain form, right attachment (like PP-attachment):
       "A in series with B which is in parallel with C" -> S(A, P(B, C))
The source is pulled out wherever it appears; it always sits across the load.
"""
from .lexicon import COMPONENT_TAGS, class_from_unit, spice_value
from .tree import E, op

COUNTS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "both": 2}
OPENERS, CLOSERS = {"("}, {")"}


def chunk(words, tags):
    """-> sorted items: ("ENT", cls, value, pos) | ("OP", "S"/"P", pos) | ("SEP", pos)
    | ("LP", pos) | ("RP", pos)."""
    vals = []                                         # (pos, end, text)
    i = 0
    while i < len(words):
        if tags[i] == "VAL":
            text, j = words[i], i + 1
            # glue UNIT tokens only while the result is still a value: an unseen
            # noun mis-tagged as UNIT ("4k7 res") must not swallow the value
            while j < len(words) and tags[j] == "UNIT" and spice_value(text + words[j], "resistor"):
                text += words[j]; j += 1
            vals.append((i, j, text))
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
        # Semantic constraints: an explicit unit outranks the tagger's noun class
        # ("a 1mH choke" is an inductor), and a source must be in volts.
        if pick and cls != "diode":
            unit_cls = class_from_unit(pick[2])
            if unit_cls:
                cls = unit_cls
            elif cls == "voltage_source":
                cls = "resistor"
        value = None
        if pick and cls != "diode":
            used.add(pick[0])
            value = spice_value(pick[2], cls)
        copies = 1
        if pick and pick[0] < n and pick[0] > 0 and words[n].lower().endswith("s"):
            copies = COUNTS.get(words[pick[0] - 1].lower(), 1)      # "two 1k resistors"
        for k in range(copies):
            items.append(("ENT", cls, value, n + k / 10))
    # Back-off: a value no noun claimed still implies a component, if it is a value
    for pos, _, text in vals:
        if pos in used or any(abs(pos - n) == 1 and tags[n] == "DIO" for n in nouns):
            continue
        cls = class_from_unit(text) or "resistor"
        value = spice_value(text, cls)
        if value is not None:
            items.append(("ENT", cls, value, pos))
    for i, (w, t) in enumerate(zip(words, tags)):
        if w in OPENERS:
            items.append(("LP", i))
        elif w in CLOSERS:
            items.append(("RP", i))
        elif t in ("SER", "PAR"):
            items.append(("OP", "S" if t == "SER" else "P", i))
        elif t == "SEP":
            items.append(("SEP", i))
    return sorted(items, key=lambda x: x[-1])


def _group_parens(items):
    """Replace each (...) span by one ("ENT", tree, None, pos) item, recursively."""
    out, stack = [], []
    for it in items:
        if it[0] == "LP":
            stack.append(out)
            out = []
        elif it[0] == "RP" and stack:
            inner = _structure(out)
            out = stack.pop()
            if inner is not None:
                out.append(("TREE", inner, None, it[-1]))
        elif it[0] != "RP":
            out.append(it)
    while stack:                                       # unbalanced "(": keep the contents
        out = stack.pop() + out
    return out


def _leaf(it):
    return it[1] if it[0] == "TREE" else E(it[1], it[2])


def _clause(items):
    """Flat clause -> tree (list form or right-attached chain), or None."""
    ents = [it for it in items if it[0] in ("ENT", "TREE")]
    if not ents:
        return None
    leaves = [_leaf(e) for e in ents]
    if len(leaves) == 1:
        return leaves[0]
    last = ents[-1][-1]
    trailing = [it for it in items if it[0] == "OP" and it[-1] > last]
    has_sep = any(it[0] == "SEP" and ents[0][-1] < it[-1] < last for it in items)
    if has_sep and trailing:                           # list form
        return op(trailing[0][1], leaves)
    # no operator between two parts: the clause's trailing operator if any
    # ("three 1k resistors in parallel"), else series
    default = trailing[0][1] if trailing else "S"
    ops = []
    for a, b in zip(ents, ents[1:]):
        between = [it[1] for it in items if it[0] == "OP" and a[-1] < it[-1] < b[-1]]
        ops.append(between[-1] if between else default)
    tree = leaves[-1]
    for k, leaf in zip(reversed(ops), reversed(leaves[:-1])):
        tree = op(k, [leaf, tree])
    return tree


def _structure(items):
    """Split at 'comma + operator' boundaries; fold clauses left to right."""
    items = _group_parens(items)
    clauses, cur, pending = [], [], []
    for i, it in enumerate(items):
        nxt = next((x for x in items[i + 1:] if x[0] != "SEP"), None)
        if it[0] == "SEP" and nxt is not None and nxt[0] == "OP" and \
                any(x[0] in ("ENT", "TREE") for x in cur):
            clauses.append((pending, cur))
            cur, pending = [], []
            continue
        if not cur and it[0] == "OP" and clauses:
            pending.append(it[1])                      # operator that opens a new clause
            continue
        cur.append(it)
    clauses.append((pending, cur))
    tree = None
    for ops_, cl in clauses:
        sub = _clause(cl)
        if sub is None:
            continue
        tree = sub if tree is None else op(ops_[-1] if ops_ else "S", [tree, sub])
    return tree


def parse(words, tags):
    """-> (source E-node or None, load tree). Raises ValueError if no load."""
    items = chunk(words, tags)
    src = next((E(it[1], it[2]) for it in items if it[0] == "ENT" and it[1] == "voltage_source"), None)
    items = [it for it in items if not (it[0] == "ENT" and it[1] == "voltage_source")]
    tree = _structure(items)
    if tree is None:
        raise ValueError("no components found")
    return src, tree
