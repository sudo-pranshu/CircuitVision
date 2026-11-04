"""Tokenizer, tag set and lexicon shared by the corpus generator, tagger and parser."""
import re

TAGS = ["SRC", "RES", "CAP", "IND", "DIO", "VAL", "UNIT", "SER", "PAR", "SEP", "O"]
COMPONENT_TAGS = {"SRC": "voltage_source", "RES": "resistor", "CAP": "capacitor",
                  "IND": "inductor", "DIO": "diode"}

TOKEN_RE = re.compile(
    r"\d+[pnuµμmkKMGR]\d+(?![\w.])"     # RKM codes: 4k7, 2R2, 1M5 (one token)
    r"|\d+[A-Z]+\d+\w*"                   # part numbers: 1N4148, 2N2222
    r"|\d+(?:\.\d+)?[a-zA-Zµμ]*"           # 10k, 4.7uF, 9V, 220
    r"|[a-zA-Zµμ]+|[,()]")
NUMERIC_RE = re.compile(r"^(\d+(\.\d+)?[a-zA-Zµμ]*|\d+[pnuµμmkKMGR]\d+)$")
# spelled-out multiplier units the tagger never saw: "2 megohms" -> "2M ohm", "4.7 kilohm" -> "4.7k ohm"
SPELLED_UNIT_RE = [(re.compile(r"(\d)\s*(?:mega?\s*ohms?|MΩ)\b", re.I), r"\1M ohm"),
                   (re.compile(r"(\d)\s*kilo?\s*ohms?\b", re.I), r"\1k ohm")]


def tokenize(text: str) -> list[str]:
    for pattern, repl in SPELLED_UNIT_RE:
        text = pattern.sub(repl, text)
    return TOKEN_RE.findall(text.replace("Ω", " ohm"))


def shape(token: str) -> str:
    """Back-off class for words the tagger never saw in training."""
    # RKM codes (4k7, 2R2) are numbers; part numbers (1N4148) are not
    return "<NUM>" if NUMERIC_RE.match(token) else "<UNK>"


# ---- value normalisation ----
VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(meg|[pnuµμmkKMG])?", re.I)
UNIT_WORDS = {"ohm", "ohms", "v", "volt", "volts", "f", "farad", "h", "henry",
              "k", "kilo", "kiloohm", "kohm", "uf", "nf", "pf", "mh", "uh", "mf"}


RKM_RE = re.compile(r"^(\d+)([pnuµμmkKMGR])(\d+)(ohms?|Ω|[fFhHvV])?$")
R_SUFFIX_RE = re.compile(r"^(\d+(?:\.\d+)?)R$")            # 330R = 330 ohm (BS 1852)
VALUE_FULL_RE = re.compile(r"^(\d+(?:\.\d+)?)(meg|[pnuµμmkKMG])?(ohms?|Ω|[fFhHvV]|volts?)?$", re.I)


def spice_value(text: str, cls: str) -> str | None:
    """'10k' -> '10k', '4.7uF' -> '4.7u', '1 M' (resistor) -> '1meg', '5V' -> 'DC 5',
    '4k7' -> '4.7k', '2R2' -> '2.2', '330R' -> '330'. Anything that is not wholly a value -> None."""
    t = text.replace(" ", "").replace(",", ".")
    r = R_SUFFIX_RE.match(t)
    if r:
        return r.group(1) if cls == "resistor" else None
    r = RKM_RE.match(t)
    if r:
        pre = {"R": "", "µ": "u", "μ": "u", "K": "k", "M": "meg", "G": "g"}.get(r.group(2), r.group(2))
        num = f"{r.group(1)}.{r.group(3)}"
        return f"DC {num}{pre}" if cls == "voltage_source" else num + pre
    if not VALUE_FULL_RE.match(t):
        return None
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
    """'4.7uF' -> capacitor, '1mH' -> inductor, '220ohm' -> resistor, '5V' -> source,
    '4k7' / '2R2' / '330R' -> resistor."""
    if R_SUFFIX_RE.match(text.replace(" ", "")):
        return "resistor"
    if RKM_RE.match(text.replace(" ", "")):
        unit = RKM_RE.match(text.replace(" ", "")).group(4)
        unit_cls = {"f": "capacitor", "h": "inductor", "v": "voltage_source"}.get((unit or "").lower()[:1])
        if unit_cls:
            return unit_cls
        pre = RKM_RE.match(text.replace(" ", "")).group(2)
        return "capacitor" if pre in "pnuµμ" else "resistor" if pre in "kKMGR" else None   # 3n3 / 4k7
    m = VALUE_RE.match(text)
    tail = text[m.end():].lower() if m else ""
    for unit, cls in UNIT_CLASS:
        if tail.startswith(unit):
            return cls
    if m and m.group(2) and m.group(2).lower() in ("k", "meg"):
        return "resistor"                  # kilo/mega only make sense for resistance
    return None
