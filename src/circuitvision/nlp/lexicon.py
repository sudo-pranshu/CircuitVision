"""Tokenizer, tag set and lexicon shared by the corpus generator, tagger and parser."""
import re

TAGS = ["SRC", "RES", "CAP", "IND", "DIO", "VAL", "UNIT", "SER", "PAR", "SEP", "O"]
COMPONENT_TAGS = {"SRC": "voltage_source", "RES": "resistor", "CAP": "capacitor",
                  "IND": "inductor", "DIO": "diode"}

TOKEN_RE = re.compile(r"\d+(?:\.\d+)?[a-zA-Zµμ]*|[a-zA-Zµμ]+|,")
NUMERIC_RE = re.compile(r"^\d+(\.\d+)?[a-zA-Zµμ]*$")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.replace("Ω", " ohm"))


def shape(token: str) -> str:
    """Back-off class for words the tagger never saw in training."""
    if NUMERIC_RE.match(token):
        return "<NUM>"
    return "<UNK>"


# ---- value normalisation ----
VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(meg|[pnuµμmkKMG])?", re.I)
UNIT_WORDS = {"ohm", "ohms", "v", "volt", "volts", "f", "farad", "h", "henry",
              "k", "kilo", "kiloohm", "kohm", "uf", "nf", "pf", "mh", "uh", "mf"}


def spice_value(text: str, cls: str) -> str | None:
    """'10k' -> '10k', '4.7uF' -> '4.7u', '1 M' (resistor) -> '1meg', '5V' -> 'DC 5'."""
    t = text.replace(" ", "")
    m = VALUE_RE.match(t)
    if not m:
        return None
    num, suf = m.group(1), m.group(2) or ""
    if suf == "M":                       # capital M = mega for resistors, SPICE needs meg
        suf = "meg"
    suf = {"µ": "u", "μ": "u", "K": "k", "G": "g"}.get(suf, suf.lower() if suf != "meg" else suf)
    if cls == "voltage_source":
        return f"DC {num}{suf}"
    return num + suf


UNIT_CLASS = [("ohm", "resistor"), ("f", "capacitor"), ("h", "inductor"), ("v", "voltage_source")]


def class_from_unit(text: str) -> str | None:
    """'4.7uF' -> capacitor, '1mH' -> inductor, '220ohm' -> resistor, '5V' -> source."""
    m = VALUE_RE.match(text)
    tail = text[m.end():].lower() if m else ""
    for unit, cls in UNIT_CLASS:
        if tail.startswith(unit):
            return cls
    if m and m.group(2) and m.group(2).lower() in ("k", "meg"):
        return "resistor"                  # kilo/mega only make sense for resistance
    return None
