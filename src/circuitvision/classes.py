"""Class set for the mini-project scope, and mapping from CGHD labels.

CGHD has 59 classes; we keep a small subset. Anything not mapped is ignored.
"""

# Canonical classes used by the detector (order = YOLO class id)
CLASSES = [
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "diode",
    "gnd",
    "junction",
    "crossover",
    "text",
]

# CGHD label -> canonical class. Prefix match: "capacitor.polarized" -> "capacitor"
CGHD_MAP = {
    "resistor": "resistor",
    "capacitor": "capacitor",
    "inductor": "inductor",
    "voltage.dc": "voltage_source",
    "voltage.ac": "voltage_source",
    "voltage.battery": "voltage_source",
    "diode": "diode",
    "gnd": "gnd",
    "junction": "junction",
    "crossover": "crossover",
    "text": "text",
}

# Classes that are wiring structure, not netlist elements
STRUCTURAL = {"junction", "crossover", "text", "gnd"}

# SPICE prefix and default value per element class
SPICE = {
    "resistor": ("R", "1k"),
    "capacitor": ("C", "1u"),
    "inductor": ("L", "1m"),
    "voltage_source": ("V", "DC 5"),
    "diode": ("D", "DMOD"),
}


def map_cghd(label: str) -> str | None:
    label = label.lower()
    if label in CGHD_MAP:
        return CGHD_MAP[label]
    for key, cls in CGHD_MAP.items():
        if label.startswith(key + "."):
            return cls
    return None
