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
