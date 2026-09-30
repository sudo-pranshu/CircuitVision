"""Synthetic corpus: English circuit descriptions with gold tags and gold trees.

Each sample: (tokens, tags, source_entity, load_tree).
Phrasings are drawn from templates with lexical variation; `held_out=True`
switches to connector words and value formats never used in training, to
measure generalisation rather than memorisation.
"""
import random

from .lexicon import spice_value
from .tree import E, op

NOUNS = {
    "resistor": (["resistor", "resistance"], "RES"),
    "capacitor": (["capacitor", "cap"], "CAP"),
    "inductor": (["inductor", "coil"], "IND"),
    "diode": (["diode", "led"], "DIO"),
}
VALUES = {
    "resistor": [("10k", None), ("1k", None), ("220", "ohm"), ("4.7", "k"), ("100k", None), ("330", "ohms")],
    "capacitor": [("10uF", None), ("100nF", None), ("47", "uf"), ("1uF", None)],
    "inductor": [("1mH", None), ("10mH", None), ("100", "uh")],
}
SRC_NOUNS = ["battery", "supply", "source", "cell"]
SRC_VALUES = [("5V", None), ("12", "volt"), ("9", "v"), ("3.3V", None), ("24", "volts")]
PRE_VERBS = [["connected", "to"], ["driving"], ["powering"], ["feeding"]]
POST_VERBS = [["connected", "across"], ["powered", "by"], ["driven", "by"], ["fed", "from"]]
SER = [["in", "series", "with"], ["followed", "by"]]
PAR = [["in", "parallel", "with"], ["parallel", "to"]]
REL = [["which", "is"], ["that", "is"]]

# Only used for the held-out split
HO_PRE_VERBS = [["supplying"], ["across"]]
HO_POST_VERBS = [["attached", "to"], ["across"]]
HO_SER = [["series", "with"]]
HO_PAR = [["parallel", "with"]]
HO_NOUNS = {"resistor": ["res"], "capacitor": ["capacitance"], "inductor": ["choke"]}

TAG_OF = {"series": "SER", "followed": "SER", "parallel": "PAR", ",": "SEP", "and": "SEP"}


def _tagged(words):
    return [(w, TAG_OF.get(w, "O")) for w in words]


def _value_tokens(rng, choices):
    num, unit = rng.choice(choices)
    toks = [(num, "VAL")] + ([(unit, "UNIT")] if unit else [])
    return toks, num + (unit or "")


def component(rng, held_out):
    cls = rng.choice(list(NOUNS))
    words, tag = NOUNS[cls]
    if held_out and cls in HO_NOUNS:
        words = HO_NOUNS[cls]
    noun = (rng.choice(words), tag)
    if cls == "diode":
        art = "an" if noun[0] == "led" else "a"
        return [(art, "O"), noun], E(cls)
    val, text = _value_tokens(rng, VALUES[cls])
    form = rng.randrange(3)
    if form == 0:
        toks = [("a", "O")] + val + [noun]
    elif form == 1:
        toks = [("a", "O"), noun, ("of", "O")] + val
    else:
        toks = val + [noun]
    return toks, E(cls, spice_value(text, cls))


def source(rng):
    val, text = _value_tokens(rng, SRC_VALUES)
    noun = (rng.choice(SRC_NOUNS), "SRC")
    toks = ([("a", "O")] + val + [noun]) if rng.random() < 0.7 else \
        [("a", "O"), ("voltage", "O"), noun, ("of", "O")] + val
    return toks, E("voltage_source", spice_value(text, "voltage_source"))


def load(rng, held_out):
    ser, par = (HO_SER, HO_PAR) if held_out else (SER, PAR)
    kind = rng.choice(["single", "chain2", "chain3", "list"])
    if kind == "single":
        return component(rng, held_out)
    if kind == "list":
        n = rng.randint(2, 3)
        comps = [component(rng, held_out) for _ in range(n)]
        k = rng.choice(["S", "P"])
        toks = []
        for i, (t, _) in enumerate(comps):
            if i:
                toks += _tagged([","] if i < n - 1 else ["and"])
            toks += t
        toks += _tagged(["in", "series" if k == "S" else "parallel"])
        return toks, op(k, [c for _, c in comps])
    n = 2 if kind == "chain2" else 3
    comps = [component(rng, held_out) for _ in range(n)]
    ops = [rng.choice(["S", "P"]) for _ in range(n - 1)]
    toks = comps[0][0]
    for i, k in enumerate(ops):
        if i:
            toks += _tagged(rng.choice(REL))
        toks += _tagged(rng.choice(ser if k == "S" else par)) + comps[i + 1][0]
    tree = comps[-1][1]
    for k, (_, c) in zip(reversed(ops), reversed(comps[:-1])):   # right attachment
        tree = op(k, [c, tree])
    return toks, tree


def sample(rng: random.Random, held_out=False):
    ld_toks, tree = load(rng, held_out)
    pre, post = (HO_PRE_VERBS, HO_POST_VERBS) if held_out else (PRE_VERBS, POST_VERBS)
    r = rng.random()
    if r < 0.45:
        s_toks, src = source(rng)
        toks = s_toks + _tagged(rng.choice(pre)) + ld_toks
    elif r < 0.9:
        s_toks, src = source(rng)
        toks = ld_toks + _tagged(rng.choice(post)) + s_toks
    else:
        toks, src = ld_toks, None
    words, tags = [w for w, _ in toks], [t for _, t in toks]
    return words, tags, src, tree


def corpus(n: int, seed: int = 0, held_out=False):
    rng = random.Random(seed)
    return [sample(rng, held_out) for _ in range(n)]
