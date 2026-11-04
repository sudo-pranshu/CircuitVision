"""English circuit description -> SPICE netlist.

  python -m circuitvision.nlp.text2netlist train                 # fit + evaluate tagger
  python -m circuitvision.nlp.text2netlist "a 5V battery driving a 10k resistor in series with a 1k resistor" [--simulate]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from ..grammar import validate
from .corpus import corpus
from .hmm import HMMTagger, tag_report
from .lexicon import tokenize
from .parser import COUNTS, parse
from .tree import canonical, to_netlist

MODEL = Path(__file__).with_name("hmm_tagger.json")


def train(n=3000, seed=0) -> HMMTagger:
    data = corpus(n, seed)
    return HMMTagger().fit([(w, t) for w, t, _, _ in data])


def load_tagger() -> HMMTagger:
    if MODEL.exists():
        return HMMTagger.load(MODEL)
    tagger = train()
    tagger.save(MODEL)
    return tagger


def text_to_netlist(text: str, tagger: HMMTagger | None = None):
    """-> (netlist, tags, errors)."""
    net, tagged, errors, _ = text_to_netlist_with_notes(text, tagger)
    return net, tagged, errors


def text_to_netlist_with_notes(text: str, tagger: HMMTagger | None = None):
    """-> (netlist, tags, errors, notes). notes lists everything that was assumed rather
    than read from the text, so a user is never handed a guess as if it were their circuit.
    Raises ValueError if the text describes no component."""
    tagger = tagger or load_tagger()
    words = tokenize(text)
    tags = tagger.tag(words)
    src, tree = parse(words, tags)
    net = to_netlist(tree, src, title=text[:60])
    return net, list(zip(words, tags)), validate(net), assumptions(words, tags, src, tree)


def assumptions(words, tags, src, tree) -> list[str]:
    from ..classes import SPICE
    notes = []
    if src is None:
        notes.append(f"No voltage source was described; a default {SPICE['voltage_source'][1]} V source "
                     "was placed across the circuit.")
    elif src[2] is None:
        notes.append(f"The source has no voltage; {SPICE['voltage_source'][1]} V was assumed.")
    missing = Counter(leaf[1] for leaf in _leaves(tree) if leaf[2] is None and leaf[1] != "diode")
    for cls, n in sorted(missing.items()):
        notes.append(f"{n} {cls}{'s' if n > 1 else ''} had no value; the default {SPICE[cls][1]} was used.")
    counted = any(w.lower() in COUNTS or w.lower() == "pair" for w in words)
    for w, t in zip(words, tags):
        if t in ("RES", "CAP", "IND", "DIO") and w.lower().endswith("s") and not counted \
                and sum(tg == "VAL" for tg in tags) < 2:
            notes.append(f"'{w}' is plural but no number of parts was given; only one was built.")
    return notes


def _leaves(t):
    return [t] if t[0] == "E" else [x for c in t[1] for x in _leaves(c)]


def evaluate(tagger, data):
    gold = [t for _, t, _, _ in data]
    pred = [tagger.tag(w) for w, _, _, _ in data]
    acc, per = tag_report(gold, pred)
    exact = 0
    for (w, _, src, tree), p in zip(data, pred):
        try:
            ps, pt = parse(w, p)
            exact += canonical(pt) == canonical(tree) and ps == src
        except ValueError:
            pass
    return acc, per, exact / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", help='description in quotes, or "train"')
    ap.add_argument("--simulate", action="store_true", help="run ngspice .op on the netlist")
    a = ap.parse_args()

    if a.text == "train":
        tagger = train()
        tagger.save(MODEL)
        for name, data in (("test (same templates)", corpus(500, seed=1)),
                           ("held-out phrasings", corpus(500, seed=2, held_out=True))):
            acc, per, exact = evaluate(tagger, data)
            print(f"\n{name}: token accuracy {acc:.3f}, exact circuit match {exact:.3f}")
            for t, (p, r, f) in per.items():
                print(f"  {t:<5} P {p:.2f}  R {r:.2f}  F1 {f:.2f}")
        return

    try:
        net, tagged, errors, notes = text_to_netlist_with_notes(a.text)
    except ValueError:
        sys.exit("No circuit components were found in that description. Name at least one part, "
                 'e.g. "a 9V battery driving a 1k resistor in series with a 2k resistor".')
    print(" ".join(f"{w}/{t}" for w, t in tagged), "\n")
    print(net)
    print("validation:", "OK" if not errors else "; ".join(errors))
    for n in notes:
        print("note:", n)
    if a.simulate and not errors:
        from .. import simulate
        try:
            for k, v in simulate.operating_point(net).items():
                print(f"  {k:<12} {v:.4g}")
        except RuntimeError as e:
            print("simulation:", e)


if __name__ == "__main__":
    main()
