from circuitvision.nlp.text2netlist import load_tagger, text_to_netlist
from circuitvision.nlp.tree import canonical
from circuitvision.nlp.parser import parse
from circuitvision.nlp.lexicon import tokenize, spice_value, class_from_unit

# Written by hand, not produced by the corpus generator
CASES = [
    ("a 5V battery driving a 10k resistor in series with a 1k resistor",
     "DC 5", "S(resistor:10k,resistor:1k)"),
    ("a 220 ohm resistor in series with an LED powered by a 9 V battery",
     "DC 9", "S(resistor:220,diode:-)"),
    ("a 1k resistor , a 2.2k resistor and a 4.7 k resistor in parallel",
     None, "P(resistor:1k,resistor:2.2k,resistor:4.7k)"),
    ("a 10uF capacitor in parallel with a 1k resistor which is in series with a 1mH inductor connected across a 12 volt supply",
     "DC 12", "P(S(resistor:1k,inductor:1m),capacitor:10u)"),
    ("the 3.3V supply feeds a 330 ohm resistor followed by a red LED",
     "DC 3.3", "S(resistor:330,diode:-)"),
    ("two resistors of 100k and 47k in series across a 24 volt source",
     "DC 24", "S(resistor:100k,resistor:47k)"),
    ("a 1mH coil in parallel with a 100nF cap",
     None, "P(capacitor:100n,inductor:1m)"),
    ("a 4.7uF capacitance in series with a 2.2k res powered by a 5V cell",
     "DC 5", "S(capacitor:4.7u,resistor:2.2k)"),
]


def test_handwritten_cases():
    tagger = load_tagger()
    correct = 0
    for text, src_val, gold in CASES:
        src, tree = parse(tokenize(text), tagger.tag(tokenize(text)))
        correct += canonical(tree) == gold and (src[2] if src else None) == src_val
    assert correct == len(CASES), f"{correct}/{len(CASES)}"


def test_netlist_valid_and_simulatable_shape():
    net, _, errors = text_to_netlist(CASES[0][0])
    assert errors == []
    assert "V1 1 0 DC 5" in net and "R1 1 2 10k" in net and "R2 2 0 1k" in net


def test_value_normalisation():
    assert spice_value("4.7uF", "capacitor") == "4.7u"
    assert spice_value("1M", "resistor") == "1meg"
    assert spice_value("12volt", "voltage_source") == "DC 12"
    assert class_from_unit("1mH") == "inductor"
    assert class_from_unit("10k") == "resistor"
    assert class_from_unit("10") is None


# Development sentences for nested grouping, counts, RKM codes and part numbers.
# Written separately from the frozen evaluation set (nlp/data/handwritten_eval.jsonl).
DEV = [
    ("a 10k resistor in parallel with a 10k resistor, then in series with a 5k resistor",
     None, "S(P(resistor:10k,resistor:10k),resistor:5k)"),
    ("(a 100 ohm resistor in parallel with a 100 ohm resistor) in series with a 50 ohm resistor",
     None, "S(P(resistor:100,resistor:100),resistor:50)"),
    ("a 2k resistor in series with (a 3k resistor in parallel with a 6k resistor) powered by a 9V battery",
     "DC 9", "S(resistor:2k,P(resistor:3k,resistor:6k))"),
    ("three 1k resistors in parallel", None, "P(resistor:1k,resistor:1k,resistor:1k)"),
    ("a 1N4007 diode in series with a 4k7 resistor", None, "S(diode:-,resistor:4.7k)"),
    ("a 2R2 resistor in series with a 1M5 resistor", None, "S(resistor:2.2,resistor:1.5meg)"),
    ("a 5k resistor in series with a 5k resistor, all in parallel with a 10k resistor across a 12V supply",
     "DC 12", "P(S(resistor:5k,resistor:5k),resistor:10k)"),
    ("a 3n3 cap in parallel with a 4k7 res", None, "P(capacitor:3.3n,resistor:4.7k)"),
]


def test_dev_nested_counts_rkm():
    tagger = load_tagger()
    for text, src_val, gold in DEV:
        src, tree = parse(tokenize(text), tagger.tag(tokenize(text)))
        assert (canonical(tree), src[2] if src else None) == (gold, src_val), text


def test_unbalanced_parenthesis_does_not_crash():
    tagger = load_tagger()
    w = tokenize("(a 1k resistor in series with a 2k resistor")
    assert canonical(parse(w, tagger.tag(w))[1]) == "S(resistor:1k,resistor:2k)"


def test_frozen_eval_set_is_well_formed():
    from circuitvision.nlp.evaluate import load
    items = load()
    assert len(items) == 30 and all(len(i["tokens"]) == len(i["tags"]) for i in items)
