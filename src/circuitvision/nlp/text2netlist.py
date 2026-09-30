"""English circuit description -> SPICE netlist.

  python -m circuitvision.nlp.text2netlist train                 # fit + evaluate tagger
  python -m circuitvision.nlp.text2netlist "a 5V battery driving a 10k resistor in series with a 1k resistor"
"""
import argparse
from pathlib import Path

from ..grammar import validate
from .corpus import corpus
from .hmm import HMMTagger, tag_report
from .lexicon import tokenize
from .parser import parse
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
    tagger = tagger or load_tagger()
    words = tokenize(text)
    tags = tagger.tag(words)
    src, tree = parse(words, tags)
    net = to_netlist(tree, src, title=text[:60])
    return net, list(zip(words, tags)), validate(net)


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

    net, tagged, errors = text_to_netlist(a.text)
    print(" ".join(f"{w}/{t}" for w, t in tagged), "\n")
    print(net)
    print("validation:", "OK" if not errors else "; ".join(errors))


if __name__ == "__main__":
    main()
